import asyncio
import base64
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Coroutine, Dict, List, Optional
from uuid import UUID

from fenic_cloud.hasura_client.generated_graphql_client.client import (
    CatalogDispatchInput,
)
from fenic_cloud.hasura_client.generated_graphql_client.enums import (
    CatalogDatasetTypeReferenceEnum,
    FileFormat,
    TypedefCatalogTypeReferenceEnum,
)
from fenic_cloud.hasura_client.generated_graphql_client.input_types import (
    CreateTableInput,
    NestedFieldInput,
    SchemaInput,
)
from fenic_cloud.hasura_client.generated_graphql_client.list_catalogs_for_organization import (
    ListCatalogsForOrganization,
)
from fenic_cloud.hasura_client.generated_graphql_client.load_table import (
    LoadTableSimpleCatalogLoadTable,
)

from fenic._backends.cloud.cloud_catalog_utils import convert_custom_dtype_to_pyarrow
from fenic._backends.cloud.manager import CloudSessionManager
from fenic._backends.local.catalog import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_DATABASE_NAME,
)
from fenic._backends.utils.catalog_utils import (
    DBIdentifier,
    TableIdentifier,
    compare_object_names,
)
from fenic.core._interfaces import BaseCatalog
from fenic.core._logical_plan.plans import LogicalPlan
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import DataTypeProto
from fenic.core.error import (
    CatalogAlreadyExistsError,
    CatalogError,
    CatalogNotFoundError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    TableAlreadyExistsError,
    TableNotFoundError,
)
from fenic.core.mcp.types import ToolParam, UserDefinedTool
from fenic.core.types import DatasetMetadata, Schema
from fenic.core.types.schema import ColumnField

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogKey:
    catalog_name: str
    catalog_id: UUID


class CloudCatalog(BaseCatalog):
    """A catalog for cloud execution mode. Implements the BaseCatalog -
    all table reads and writes should go through this class for unified table name canonicalization.
    """

    def __init__(self,
        ephemeral_catalog_id: str,
        asyncio_loop: asyncio.AbstractEventLoop,
        cloud_session_manager: CloudSessionManager):
        """Initialize the remote catalog."""
        self.cloud_session_manager = cloud_session_manager
        self.lock = threading.Lock()
        self.asyncio_loop = asyncio_loop
        self.ephemeral_catalog_id: UUID = UUID(ephemeral_catalog_id)
        self.current_catalog_id: UUID = self.ephemeral_catalog_id
        self.current_catalog_name: str = DEFAULT_CATALOG_NAME
        self.current_database_name: str = DEFAULT_DATABASE_NAME
        self.user_id = self.cloud_session_manager._user_id
        self.organization_id = self.cloud_session_manager._organization_id
        self.user_client = self.cloud_session_manager.hasura_user_client

    def does_catalog_exist(self, catalog_name: str) -> bool:
        """Checks if a catalog with the specified name exists."""
        with self.lock:
            return self._does_catalog_exist(catalog_name)

    def get_current_catalog(self) -> str:
        """Get the name of the current catalog."""
        with self.lock:
            return self.current_catalog_name

    def set_current_catalog(self, catalog_name: str) -> None:
        """Set the current catalog."""
        if not catalog_name:
            raise CatalogError("No catalog name provided")

        with self.lock:
            self._set_current_catalog(catalog_name)

    def list_catalogs(self) -> List[str]:
        """Get a list of all catalogs."""
        with self.lock:
            catalogs = self._get_catalogs_for_organization()
            # iterate over catalogs but exclude the ephemeral catalog name,
            # this will be added as the default catalog.
            remote_catalogs = [catalog.name for catalog in catalogs.values()]
            remote_catalogs.append(DEFAULT_CATALOG_NAME)
            return remote_catalogs

    def create_catalog(self, catalog_name: str, ignore_if_exists: bool = True) -> bool:
        """Create a new catalog."""
        if compare_object_names(catalog_name, DEFAULT_CATALOG_NAME):
            raise CatalogError("Cannot create a catalog with the default name")

        with self.lock:
            if self._does_catalog_exist(catalog_name):
                if ignore_if_exists:
                    return False
                else:
                    raise CatalogAlreadyExistsError(catalog_name)

            self._execute_catalog_command(
                self.user_client.create_standard_catalog(
                    catalog_name=catalog_name,
                    catalog_canonical_name=catalog_name.casefold(),
                    created_by_user_id=UUID(self.user_id),
                    parent_organization_id=UUID(self.organization_id),
                    catalog_type=TypedefCatalogTypeReferenceEnum.INTERNAL_TYPEDEF,
                    catalog_warehouse="",
                )
            )
            return True

    def drop_catalog(
        self, catalog_name: str, ignore_if_not_exists: bool = True
    ) -> bool:
        """Drop a catalog."""
        if compare_object_names(catalog_name, DEFAULT_CATALOG_NAME):
            raise CatalogError("Cannot drop the default catalog")

        if compare_object_names(catalog_name, self.current_catalog_name):
            raise CatalogError("Cannot drop the current catalog")

        with self.lock:
            if not self._does_catalog_exist(catalog_name):
                if ignore_if_not_exists:
                    return False
                else:
                    raise CatalogNotFoundError(catalog_name)

            catalog_id = self._get_catalog_id(catalog_name)
            self._execute_catalog_command(
                self.user_client.mark_catalog_as_deleted(
                    catalog_id=catalog_id,
                    deleted_at=datetime.now(),
                )
            )
            return True

    # Database operations
    def does_database_exist(self, database_name: str) -> bool:
        """Checks if a database with the specified name exists in the current catalog."""
        with self.lock:
            db_identifier = DBIdentifier.from_string(database_name).enrich(
                self.current_catalog_name
            )
            return self._does_database_exist(db_identifier.catalog, db_identifier.db)

    def get_current_database(self) -> str:
        """Get the name of the current database in the current catalog."""
        with self.lock:
            return self.current_database_name

    def set_current_database(self, database_name: str) -> None:
        """Sets the current database.
        The database name can be fully qualified or not.
        If it is not fully qualified, the current catalog will be used to resolve the database.
        If it is fully qualified, we'll attempt to resolve it against the catalog provided in the db name.
        """
        with self.lock:
            if not database_name:
                raise ValueError("No database name provided")

            db_identifier = DBIdentifier.from_string(database_name).enrich(
                self.current_catalog_name
            )
            if not self._does_database_exist(db_identifier.catalog, db_identifier.db):
                raise DatabaseNotFoundError(database_name)

            self.current_database_name = db_identifier.db
            if not db_identifier.is_catalog_name_equal(self.current_catalog_name):
                self._set_current_catalog(db_identifier.catalog)

            logger.info(f"current catalog: {self.current_catalog_name}")
            logger.info(f"current database: {self.current_database_name}")

    def list_databases(self) -> List[str]:
        """Get a list of all databases in the current catalog."""
        with self.lock:
            return self._get_databases_for_catalog()

    def create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        with self.lock:
            return self._create_database(database_name, ignore_if_exists)

    def drop_database(
        self,
        database_name: str,
        cascade: bool = False,
        ignore_if_not_exists: bool = True,
    ) -> bool:
        """Drop a database from the current catalog."""
        with self.lock:
            return self._drop_database(database_name, cascade, ignore_if_not_exists)

    # Table operations
    def does_table_exist(self, table_name: str) -> bool:
        """Checks if a table with the specified name exists in the current database."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.current_catalog_name, self.current_database_name
            )
            return self._does_table_exist(
                table_identifier.catalog, table_identifier.db, table_identifier.table
            )

    def list_tables(self) -> List[str]:
        """Get a list of all tables in the current database."""
        with self.lock:
            return self._get_tables_for_database(
                self.current_catalog_name,
                self.current_database_name,
            )

    def describe_table(self, table_name: str) -> DatasetMetadata:
        """Get the schema of the specified table."""
        with self.lock:
            table_identifier = TableIdentifier.from_string(table_name).enrich(
                self.current_catalog_name,
                self.current_database_name,
            )

            if not self._does_table_exist(
                table_identifier.catalog, table_identifier.db, table_identifier.table
            ):
                raise TableNotFoundError(table_identifier.table, table_identifier.db)

            schema =  self._get_table_details(
                table_identifier.catalog,
                table_identifier.db,
                table_identifier.table,
            )
            #TODO(bcallender): Modify fenic_cloud's graphql client to return the description.
            return DatasetMetadata(schema=schema, description=None)

    def set_table_description(self, table_name: str, description: str) -> None:
        """Set the description for a table."""
        raise NotImplementedError(
            "Set table description not implemented for cloud catalog"
        )

    def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a table from the current database."""
        with self.lock:
            return self._drop_table(table_name, ignore_if_not_exists)

    def create_table(
        self,
        table_name: str,
        schema: Schema,
        location: str,
        ignore_if_exists: bool = True,
        file_format: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Create a new table in the current database."""
        with self.lock:
            return self._create_table(
                table_name, schema, location, ignore_if_exists, file_format, description
            )

    def create_view(
        self,
        view_name: str,
        logical_plan: LogicalPlan,
        ignore_if_exists: bool = True,
        description: Optional[str] = None,
    ) -> bool:
        """Create a new view in the current database."""
        # TODO: Implement view creation for the cloud
        raise NotImplementedError(
            "View creation not implemented for cloud catalog"
        )

    def drop_view(self, view_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a view from the current database."""
        # TODO: Implement drop view for the cloud
        raise NotImplementedError(
            "Drop view not implemented for cloud catalog"
        )

    def get_view_plan(self, view_name: str) -> LogicalPlan:
        # TODO: Implement describe view for the cloud
        raise NotImplementedError(
            "get view plan not implemented for cloud catalog"
        )

    def describe_view(self, view_name: str) -> DatasetMetadata:
        """Get the schema and description of the specified view."""
        # TODO: Implement describe view for the cloud
        raise NotImplementedError(
            "Describe view not implemented for cloud catalog"
        )

    def list_views(self) -> List[str]:
        """Get a list of all views in the current database."""
        # TODO: Implement list views for the cloud
        raise NotImplementedError(
            "List views not implemented for cloud catalog"
        )

    def does_view_exist(self, view_name: str) -> bool:
        """Checks if a view with the specified name exists in the current database."""
        # TODO: Implement does view exist for the cloud
        raise NotImplementedError(
            "Method to check if view exists not implemented for cloud catalog"
        )

    def set_view_description(self, view_name: str, description: str) -> bool:
        """Set the description for a view."""
        # TODO: Implement set view description for the cloud
        raise NotImplementedError(
            "Set view description not implemented for cloud catalog"
        )

    def describe_tool(self, tool_name: str) -> UserDefinedTool:
        """Find and return the tool from the current database."""
        # TODO: Implement get tool for the cloud
        raise NotImplementedError(
            "Get tool not implemented for cloud catalog"
        )

    def create_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_params: List[ToolParam],
        tool_query: LogicalPlan,
        result_limit: int = 50,
        ignore_if_exists: bool = True,
    ) -> bool:
        """Create a new tool in the current database."""
        raise NotImplementedError(
            "Create tool not implemented for cloud catalog"
        )

    def drop_tool(self, tool_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a tool from the current database."""
        raise NotImplementedError(
            "Drop tool not implemented for cloud catalog"
        )

    def list_tools(self) -> List[UserDefinedTool]:
        """Get a list of all tools in the current database."""
        raise NotImplementedError(
            "List tools not implemented for cloud catalog"
        )

    def _drop_database(
        self,
        database_name: str,
        cascade: bool = False,
        ignore_if_not_exists: bool = True,
    ) -> bool:
        """Drop a database from the current catalog."""
        db_identifier = DBIdentifier.from_string(database_name).enrich(
            self.current_catalog_name
        )

        if not db_identifier.is_catalog_name_equal(self.current_catalog_name):
            if not self._does_catalog_exist(db_identifier.catalog):
                raise CatalogError(
                    f"Catalog {db_identifier.catalog} referenced by {database_name} does not exist."
                )
        else:
            # we are in the current catalog and this is the current database
            # you can't drop it.
            if db_identifier.is_db_name_equal(self.current_database_name):
                raise CatalogError(
                    f"Cannot drop the current database '{database_name}'. Switch to another database first."
                )

        if db_identifier.is_catalog_name_equal(
            DEFAULT_CATALOG_NAME
        ) and db_identifier.is_db_name_equal(DEFAULT_DATABASE_NAME):
            raise CatalogError(
                f"Cannot drop the default database '{DEFAULT_DATABASE_NAME}'."
            )

        if not self._does_database_exist(db_identifier.catalog, db_identifier.db):
            if ignore_if_not_exists:
                return False
            else:
                raise DatabaseNotFoundError(f"Database {database_name} does not exist")

        catalog_id = self._get_catalog_id(db_identifier.catalog)
        self._execute_catalog_command(
            self.user_client.sc_drop_namespace(
                dispatch=self._get_catalog_dispatch_input(catalog_id),
                name=db_identifier.db,
            )
        )
        return True


    def _create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        db_identifier = DBIdentifier.from_string(database_name).enrich(
            self.current_catalog_name
        )
        if not db_identifier.is_catalog_name_equal(self.current_catalog_name):
            if not self._does_catalog_exist(db_identifier.catalog):
                raise CatalogError(
                    f"Catalog {db_identifier.catalog} referenced by {database_name} does not exist."
                )

        if self._does_database_exist(db_identifier.catalog, db_identifier.db):
            if ignore_if_exists:
                return False
            else:
                raise DatabaseAlreadyExistsError(database_name)

        self._execute_catalog_command(
            self.user_client.sc_create_namespace(
                dispatch=self._get_catalog_dispatch_input(self.current_catalog_id),
                name=db_identifier.db,
                canonical_name=db_identifier.db.casefold(),
                description=None,
                properties=[],
            )
        )
        return True


    def _does_table_exist(
        self, catalog_name: str, db_name: str, table_name: str
    ) -> bool:
        if not compare_object_names(self.current_catalog_name, catalog_name):
            if not self._does_catalog_exist(catalog_name):
                return False

        if not self._does_database_exist(catalog_name, db_name):
            return False

        table = self._get_table(catalog_name, db_name, table_name)
        return table is not None

    def _set_current_catalog(self, catalog_name: str) -> None:
        if not catalog_name:
            raise CatalogError("No catalog name provided")

        if compare_object_names(catalog_name, self.current_catalog_name):
            return

        if compare_object_names(catalog_name, DEFAULT_CATALOG_NAME):
            self.current_catalog_id = self.ephemeral_catalog_id
            self.current_catalog_name = DEFAULT_CATALOG_NAME
            return

        catalog = self._get_catalog_by_name(catalog_name)
        if not catalog:
            raise CatalogError(f"Catalog {catalog_name} does not exist")

        self.current_catalog_id = catalog.catalog_id
        self.current_catalog_name = catalog.catalog_name

    def _does_catalog_exist(self, catalog_name: str) -> bool:
        if compare_object_names(DEFAULT_CATALOG_NAME, catalog_name):
            return True

        if compare_object_names(self.current_catalog_name, catalog_name):
            return True

        catalog = self._get_catalog_by_name(catalog_name)
        return True if catalog else False

    def _get_catalogs_for_organization(self) -> Dict[str, ListCatalogsForOrganization]:
        result = self._execute_catalog_command(
            self.user_client.list_catalogs_for_organization(
                parent_organization_id=self.organization_id
            )
        )
        filtered_catalogs = [
            catalog for catalog in result.catalogs if not catalog.ephemeral
        ]
        return {catalog.name.casefold(): catalog for catalog in filtered_catalogs}

    def _does_database_exist(self, catalog_name: str, database_name: str) -> bool:
        """Checks if a database with the specified name exists in the specified catalog."""
        if not compare_object_names(self.current_catalog_name, catalog_name):
            if not self._does_catalog_exist(catalog_name):
                return False

        databases = self._get_databases_for_catalog(catalog_name)
        return any(
            compare_object_names(database, database_name) for database in databases
        )

    def _execute_catalog_command(self, command: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(
            command, self.asyncio_loop
        ).result()

    def _get_catalog_by_name(self, catalog_name: str) -> Optional[CatalogKey]:
        if compare_object_names(catalog_name, self.current_catalog_name):
            return (self.current_catalog_name, self.current_catalog_id)

        catalogs = self._get_catalogs_for_organization()
        catalog = catalogs.get(catalog_name.casefold(), None)
        if not catalog:
            return None
        return CatalogKey(catalog.name.casefold(), catalog.catalog_id)

    def _get_catalog_id(self, catalog_name: str) -> UUID:
        # this takes care of both the default catalog and the current catalog.
        if compare_object_names(catalog_name, self.current_catalog_name):
            return self.current_catalog_id

        catalog = self._get_catalog_by_name(catalog_name)
        if not catalog:
            raise ValueError(f"Catalog {catalog_name} does not exist")
        return catalog.catalog_id

    def _get_databases_for_catalog(
        self,
        catalog_name: Optional[str] = None,
    ) -> List[str]:
        if not catalog_name:
            catalog_name = self.current_catalog_name
        catalog_id = self._get_catalog_id(catalog_name)
        result = self._execute_catalog_command(
            self.user_client.list_namespaces(
                dispatch=self._get_catalog_dispatch_input(catalog_id)
            )
        )
        databases = [
            namespace.name for namespace in result.simple_catalog.list_namespaces
        ]

        # in case we are in the current catalog, always return the default database.
        if self.current_catalog_name == DEFAULT_CATALOG_NAME:
            databases.append(DEFAULT_DATABASE_NAME)

        return databases

    def _get_tables_for_database(
        self,
        catalog_name: str,
        db_name: str,
    ) -> List[str]:
        catalog_id = self._get_catalog_id(catalog_name)
        result = self._execute_catalog_command(
            self.user_client.get_dataset_names_for_catalog_namespace(
                catalog_id=catalog_id,
                namespace=db_name,
                dataset_type=CatalogDatasetTypeReferenceEnum.TABLE,
            )
        )
        return [dataset.name for dataset in result.catalog_dataset]

    def _get_table(
            self,
            catalog_name: str,
            db_name: str,
            table_name: str,
            ignore_if_not_exists: bool = True,
    ) -> LoadTableSimpleCatalogLoadTable:
        catalog_id = self._get_catalog_id(catalog_name)
        try:
            result = self._execute_catalog_command(
                self.user_client.load_table(
                    dispatch=self._get_catalog_dispatch_input(catalog_id),
                    namespace=db_name,
                    name=table_name,
                )
            )
            return result.simple_catalog.load_table
        except Exception as e:
            if ignore_if_not_exists:
                return None
            logger.debug(f"Error getting table {table_name} from catalog {catalog_name} and database {db_name}: {e}")
            raise e


    def _get_table_details(
        self, catalog_name: str, db_name: str, table_name: str
    ) -> Schema:
        load_table = self._get_table(catalog_name, db_name, table_name, ignore_if_not_exists=False)
        return self._get_table_schema(load_table)

    def _get_catalog_dispatch_input(
        self,
        catalog_id: Optional[UUID] = None,
    ) -> CatalogDispatchInput:
        """Builds a CatalogDispatchInput object for the current catalog or the specified catalog."""
        return CatalogDispatchInput(
            catalog_id=catalog_id if catalog_id else self.current_catalog_id,
            parent_organization_id=self.organization_id,
            requested_by_user_id=self.user_id,
        )

    def _create_table(
        self,
        table_name: str,
        schema: Schema,
        location: str,
        ignore_if_exists: bool = True,
        file_format: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.current_catalog_name, self.current_database_name
        )

        if not table_identifier.is_catalog_name_equal(DEFAULT_CATALOG_NAME):
            if not self._does_catalog_exist(table_identifier.catalog):
                raise CatalogError(
                    f"Catalog {table_identifier.catalog} referenced by {table_name} does not exist."
                )

        if not table_identifier.is_db_name_equal(self.current_database_name):
            if not self._does_database_exist(
                table_identifier.catalog, table_identifier.db
            ):
                raise CatalogError(
                    f"Database {table_identifier.db} referenced by {table_name} does not exist."
                )

        if self._does_table_exist(
            table_identifier.catalog, table_identifier.db, table_identifier.table
        ):
            if ignore_if_exists:
                return False
            else:
                raise TableAlreadyExistsError(table_identifier.table, table_identifier.db)

        catalog_id = self._get_catalog_id(table_identifier.catalog)
        fixed_file_format = (
            FileFormat.PARQUET
            if file_format is None
            else FileFormat(file_format.upper())
        )
        self._execute_catalog_command(
            self.user_client.sc_create_table(
                dispatch=self._get_catalog_dispatch_input(catalog_id),
                namespace=table_identifier.db,
                table=CreateTableInput(
                    name=table_identifier.table,
                    canonical_name=table_identifier.table.casefold(),
                    description=description,
                    external=False,
                    location=location,
                    file_format=fixed_file_format,
                    partition_field_names=[],
                    schema_=self._get_schema_input_from_schema(schema),
                ),
            )
        )
        return True

    def _drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a table from the current database."""
        table_identifier = TableIdentifier.from_string(table_name).enrich(
            self.current_catalog_name, self.current_database_name
        )

        if not table_identifier.is_catalog_name_equal(self.current_catalog_name):
            if not self._does_catalog_exist(table_identifier.catalog):
                raise CatalogError(
                    f"Catalog {table_identifier.catalog} referenced by {table_name} does not exist."
                )

        if not table_identifier.is_db_name_equal(self.current_database_name):
            if not self._does_database_exist(
                table_identifier.catalog, table_identifier.db
            ):
                raise CatalogError(
                    f"Database {table_identifier.db} referenced by {table_name} does not exist."
                )

        if not self._does_table_exist(
            table_identifier.catalog, table_identifier.db, table_identifier.table
        ):
            if ignore_if_not_exists:
                return False
            else:
                raise TableNotFoundError(table_identifier.table, table_identifier.db)

        catalog_id = self._get_catalog_id(table_identifier.catalog)
        self._execute_catalog_command(
            self.user_client.sc_drop_dataset(
                dispatch=self._get_catalog_dispatch_input(catalog_id),
                namespace=table_identifier.db,
                dataset_name=table_identifier.table,
            )
        )
        return True

    @staticmethod
    def _get_schema_input_from_schema(schema: Schema) -> SchemaInput:
        fields: List[NestedFieldInput] = []
        context = SerdeContext()
        for column_field in schema.column_fields:
            data_type_proto = context.serialize_data_type(
                "data_type",
                column_field.data_type
            ).SerializeToString()
            fields.append(
                NestedFieldInput(
                    name=column_field.name,
                    data_type=base64.b64encode(data_type_proto).decode("utf-8"),
                    arrow_data_type=str(convert_custom_dtype_to_pyarrow(column_field.data_type)),
                    doc=None,
                    nullable=False,
                    identifier_field=None,
                    metadata=None,
                )
            )
        return SchemaInput(fields=fields)

    @staticmethod
    def _get_table_schema(table_details: LoadTableSimpleCatalogLoadTable) -> Schema:
        context = SerdeContext()
        column_fields = []
        for field in table_details.schema_.fields:
            decoded_data = base64.b64decode(field.data_type)
            column_fields.append(
                ColumnField(
                    name=field.name,
                    data_type=context.deserialize_data_type(
                        "data_type",
                        DataTypeProto.FromString(decoded_data)
                    ),
                )
            )
        return Schema(column_fields=column_fields)