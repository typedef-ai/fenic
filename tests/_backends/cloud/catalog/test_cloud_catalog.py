import asyncio
import os
import threading
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fenic_cloud")
from fenic_cloud.hasura_client.generated_graphql_client import Client
from fenic_cloud.hasura_client.generated_graphql_client.create_namespace import (
    CreateNamespace,
    CreateNamespaceInsertCatalogNamespaceOne,
)
from fenic_cloud.hasura_client.generated_graphql_client.enums import (
    FileFormat,
    TypedefObjectStateReferenceEnum,
)
from fenic_cloud.hasura_client.generated_graphql_client.fragments import (
    SimpleCatalogSchemaDetailsFields,
    SimpleCatalogTableDetailsSchema,
)
from fenic_cloud.hasura_client.generated_graphql_client.get_dataset_names_for_catalog_namespace import (
    GetDatasetNamesForCatalogNamespace,
    GetDatasetNamesForCatalogNamespaceCatalogDataset,
    GetDatasetNamesForCatalogNamespaceCatalogDatasetNamespace,
)
from fenic_cloud.hasura_client.generated_graphql_client.list_catalogs_for_organization import (
    ListCatalogsForOrganization,
    ListCatalogsForOrganizationCatalogs,
)
from fenic_cloud.hasura_client.generated_graphql_client.list_namespaces import (
    ListNamespaces,
    ListNamespacesSimpleCatalog,
    ListNamespacesSimpleCatalogListNamespaces,
)
from fenic_cloud.hasura_client.generated_graphql_client.load_table import (
    LoadTable,
    LoadTableSimpleCatalog,
    LoadTableSimpleCatalogLoadTable,
)
from fenic_cloud.hasura_client.generated_graphql_client.sc_create_table import (
    ScCreateTable,
    ScCreateTableSimpleCatalog,
    ScCreateTableSimpleCatalogCreateTable,
)
from fenic_cloud.hasura_client.generated_graphql_client.sc_drop_dataset import (
    ScDropDataset,
    ScDropDatasetSimpleCatalog,
)
from fenic_cloud.hasura_client.generated_graphql_client.sc_drop_namespace import (
    ScDropNamespace,
    ScDropNamespaceSimpleCatalog,
)

from fenic import ColumnField, IntegerType, Schema, StringType
from fenic._backends.cloud.catalog import CloudCatalog
from fenic._backends.cloud.manager import CloudSessionManager
from fenic._backends.cloud.session_state import CloudSessionState
from fenic._backends.local.catalog import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_DATABASE_NAME,
)
from fenic.core.error import (
    CatalogAlreadyExistsError,
    CatalogError,
    CatalogNotFoundError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    TableAlreadyExistsError,
    TableNotFoundError,
)

pytestmark = pytest.mark.cloud

TEST_DEFAULT_USER_ID = "e7b92407-d203-49dd-8864-c6d1141389ae"
TEST_DEFAULT_ORGANIZATION_ID = "1d0ee92b-ed90-4299-9563-48c447a21630"
TEST_DEFAULT_APP_NAME = "test_app"
TEST_CATALOG_NAME = "test_catalog"
TEST_CATALOG_UUID = "123e4567-e89b-12d3-a456-426614174000"
TEST_DATABASE_NAME = "test_database"
TEST_UNIQUE_IDENTIFIER = "2c323605-4dc0-4a0d-9d6a-e7476d1ed9d7"
TEST_NONEXISTENT_CATALOG_NAME = "nonexistent_catalog"
TEST_TABLE_NAME_1 = "table1"
TEST_TABLE_NAME_1_UUID = "123e4567-e89b-12d3-a456-426614174001"
TEST_TABLE_NAME_2 = "table2"
TEST_TABLE_NAME_2_UUID = "123e4567-e89b-12d3-a456-426614174002"
TEST_TABLE_NAME_DEFAULT_CATALOG = "table_table_default_catalog"
TEST_TABLE_UUID_DEFAULT_CATALOG = "123e4567-e89b-12d3-a456-426614174011"

TEST_COMPOSED_TABLE_NAME_1 = (
    f"{TEST_CATALOG_NAME}.{TEST_DATABASE_NAME}.{TEST_TABLE_NAME_1}"
)
TEST_COMPOSED_TABLE_NAME_2 = (
    f"{TEST_CATALOG_NAME}.{TEST_DATABASE_NAME}.{TEST_TABLE_NAME_2}"
)
TEST_NEW_DATABASE_UUID = "123e4567-e89b-12d3-a456-426614174003"
TEST_NEW_DATABASE_NAME = "new_database"
TEST_VALID_COMPOSED_DB_NAME = f"{TEST_CATALOG_NAME}.{TEST_DATABASE_NAME}"
TEST_EPHMERAL_CATALOG_ID = "123e4567-e89b-12d3-a456-426614174010"
TEST_NEW_TABLE_NAME = "new_table"
TEST_SAMPLE_LOCATION = "s3://test-bucket/test-path"
TEST_NEW_CATALOG_NAME = "new_catalog"


@pytest.fixture(scope="function")
def typedef_event_loop():
    ev_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ev_loop)
    yield ev_loop
    ev_loop.stop()

@pytest.fixture
def schema(): # noqa: D103
    return Schema(
        column_fields=[
            ColumnField(name="id", data_type=IntegerType),
            ColumnField(name="name", data_type=StringType),
        ]
    )


@pytest.fixture
def mock_user_client(schema):
    user_client = MagicMock(spec=Client)

    user_client.list_catalogs_for_organization.return_value = (
        ListCatalogsForOrganization(
            catalogs=[
                ListCatalogsForOrganizationCatalogs(
                    typename__="TypedefCatalogDetails",
                    catalog_id=uuid.UUID(str(TEST_CATALOG_UUID)),
                    name=TEST_CATALOG_NAME,
                    canonical_name=TEST_CATALOG_NAME,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    description="",
                    ephemeral=False,
                )
            ]
        )
    )

    user_client.list_namespaces.return_value = ListNamespaces(
        simple_catalog=ListNamespacesSimpleCatalog(
            list_namespaces=[
                ListNamespacesSimpleCatalogListNamespaces(
                    typename__="SimpleCatalogNamespaceDetails",
                    name=TEST_DATABASE_NAME,
                    description="",
                    catalog_id=uuid.UUID(str(TEST_CATALOG_UUID)),
                    catalog_name=TEST_CATALOG_NAME,
                    canonical_name=TEST_DATABASE_NAME,
                    unique_identifier=TEST_UNIQUE_IDENTIFIER,
                    properties=[],
                )
            ]
        )
    )

    user_client.get_dataset_names_for_catalog_namespace.return_value = (
        GetDatasetNamesForCatalogNamespace(
            catalog_dataset=[
                GetDatasetNamesForCatalogNamespaceCatalogDataset(
                    name=TEST_TABLE_NAME_1,
                    namespace=GetDatasetNamesForCatalogNamespaceCatalogDatasetNamespace(
                        name=TEST_DATABASE_NAME,
                    ),
                    dataset_id=uuid.UUID(str(TEST_TABLE_NAME_1_UUID)),
                ),
                GetDatasetNamesForCatalogNamespaceCatalogDataset(
                    name=TEST_TABLE_NAME_2,
                    namespace=GetDatasetNamesForCatalogNamespaceCatalogDatasetNamespace(
                        name=TEST_DATABASE_NAME,
                    ),
                    dataset_id=uuid.UUID(str(TEST_TABLE_NAME_2_UUID)),
                ),
            ]
        )
    )

    user_client.load_table.return_value = LoadTable(
        simple_catalog=LoadTableSimpleCatalog(
            load_table=LoadTableSimpleCatalogLoadTable(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                schema_=SimpleCatalogTableDetailsSchema(
                    schema_id=1,
                    identifier_field_ids=[1, 2, 3],
                    fields=[
                        SimpleCatalogSchemaDetailsFields(
                            id=1,
                            name="id",
                            data_type="int64",
                            arrow_data_type="int64",
                            nullable=False,
                            metadata=None,
                        ),
                        SimpleCatalogSchemaDetailsFields(
                            id=2,
                            name="name",
                            data_type="string",
                            arrow_data_type="string",
                            nullable=False,
                            metadata=None,
                        ),
                        SimpleCatalogSchemaDetailsFields(
                            id=3,
                            name="account_balance",
                            data_type="Decimal128",
                            arrow_data_type="Decimal128",
                            nullable=False,
                            metadata=None,
                        ),
                    ],
                ),
                name=TEST_TABLE_NAME_1,
                location=None,
                external=True,
                file_format=None,
                partition_field_names=None,
            ),
        ),
    )

    user_client.create_namespace.return_value = CreateNamespace(
        insert_catalog_namespace_one=CreateNamespaceInsertCatalogNamespaceOne(
            namespace_id=uuid.UUID(str(TEST_NEW_DATABASE_UUID)),
            name=TEST_NEW_DATABASE_NAME,
            created_at=datetime.now(),
            object_state=TypedefObjectStateReferenceEnum.ACTIVE,
        )
    )

    user_client.sc_drop_namespace.return_value = ScDropNamespace(
        simple_catalog=ScDropNamespaceSimpleCatalog(
            drop_namespace=True,
        )
    )

    user_client.sc_create_table.return_value = ScCreateTable(
        simple_catalog=ScCreateTableSimpleCatalog(
            create_table=ScCreateTableSimpleCatalogCreateTable(
                created_at=datetime.now(),
                updated_at=datetime.now(),
                name=TEST_NEW_TABLE_NAME,
                location=None,
                external=False,
                file_format=FileFormat.PARQUET,
                partition_field_names=None,
                schema_=SimpleCatalogTableDetailsSchema(
                    schema_id=1,
                    identifier_field_ids=None,
                    fields=[
                        SimpleCatalogSchemaDetailsFields(
                            id=1,
                            name="id",
                            data_type="int64",
                            arrow_data_type="int64",
                            nullable=False,
                            metadata=None,
                        ),
                    ],
                ),
            ),
        ),
    )

    user_client.sc_drop_dataset.return_value = ScDropDataset(
        simple_catalog=ScDropDatasetSimpleCatalog(
            drop_dataset=True,
        ),
    )

    return user_client

@pytest.fixture
def mock_user_client_no_catalogs():
    user_client = MagicMock(spec=Client)
    user_client.list_catalogs_for_organization.return_value = (
        ListCatalogsForOrganization(catalogs=[])
    )
    user_client.get_dataset_names_for_catalog_namespace.return_value = (
        GetDatasetNamesForCatalogNamespace(
            catalog_dataset=[
                GetDatasetNamesForCatalogNamespaceCatalogDataset(
                    name=TEST_TABLE_NAME_DEFAULT_CATALOG,
                    namespace=GetDatasetNamesForCatalogNamespaceCatalogDatasetNamespace(
                        name=DEFAULT_DATABASE_NAME,
                    ),
                    dataset_id=uuid.UUID(str(TEST_TABLE_UUID_DEFAULT_CATALOG)),
                ),
            ]
        )
    )
    return user_client


@pytest.fixture
def mock_session_state(typedef_event_loop):
    yield _init_mock_state(typedef_event_loop)

@pytest.fixture
def mock_cloud_session_manager():
    cloud_session_manager = MagicMock(spec=CloudSessionManager)
    cloud_session_manager.user_id = TEST_DEFAULT_USER_ID
    cloud_session_manager.organization_id = TEST_DEFAULT_ORGANIZATION_ID
    cloud_session_manager.hasura_user_client = None
    return cloud_session_manager

@pytest.fixture
def cloud_catalog(mock_session_state, mock_user_client, mock_cloud_session_manager):
    return _init_cloud_catalog(mock_session_state, mock_user_client, mock_cloud_session_manager)


@pytest.fixture
def cloud_catalog_default_catalog(mock_session_state, mock_user_client_no_catalogs, mock_cloud_session_manager):
    return _init_cloud_catalog(mock_session_state, mock_user_client_no_catalogs, mock_cloud_session_manager)


def test_does_catalog_exist(cloud_catalog):
    assert cloud_catalog.does_catalog_exist(TEST_CATALOG_NAME) is True


def test_list_catalogs(cloud_catalog):
    assert cloud_catalog.list_catalogs() == [
        TEST_CATALOG_NAME,
        DEFAULT_CATALOG_NAME,
    ]


def test_list_catalogs_use_default_catalog(cloud_catalog_default_catalog):
    assert cloud_catalog_default_catalog.list_catalogs() == [DEFAULT_CATALOG_NAME]


def test_set_current_catalog(cloud_catalog):
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    assert cloud_catalog.get_current_catalog() == TEST_CATALOG_NAME

    with pytest.raises(CatalogError):
        cloud_catalog.set_current_catalog(TEST_NONEXISTENT_CATALOG_NAME)


def test_list_databases(cloud_catalog):
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    result = cloud_catalog.list_databases()
    assert result is not None
    assert len(result) > 0
    assert any(database == TEST_DATABASE_NAME for database in result)


def test_list_databases_use_default_catalog(cloud_catalog):
    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    result = cloud_catalog.list_databases()
    assert result is not None
    assert len(result) > 0
    assert any(database == DEFAULT_DATABASE_NAME for database in result)


def test_set_current_database(cloud_catalog):
    # default catalog is set at this point.
    with pytest.raises(DatabaseNotFoundError) as e:
        cloud_catalog.set_current_database("SOME_OTHER_DB")
    assert str(e.value) == "Database 'SOME_OTHER_DB' does not exist"

    # test setting the current database using a fully qualified name.
    cloud_catalog.set_current_database(TEST_VALID_COMPOSED_DB_NAME)
    assert cloud_catalog.get_current_database() == TEST_DATABASE_NAME
    assert cloud_catalog.get_current_catalog() == TEST_CATALOG_NAME

    with pytest.raises(DatabaseNotFoundError) as e:
        cloud_catalog.set_current_database("SOME_CATALOG.SOME_OTHER_DB")
    assert str(e.value) == "Database 'SOME_CATALOG.SOME_OTHER_DB' does not exist"

    # test setting the current database using a non-fully qualified name.
    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    cloud_catalog.set_current_database(DEFAULT_DATABASE_NAME)
    assert cloud_catalog.get_current_database() == DEFAULT_CATALOG_NAME
    assert cloud_catalog.get_current_catalog() == DEFAULT_DATABASE_NAME

    # fail to set the current database to a nonexistent database.
    with pytest.raises(DatabaseNotFoundError) as e:
        cloud_catalog.set_current_database("nonexistent_database")
    assert str(e.value) == "Database 'nonexistent_database' does not exist"

    assert cloud_catalog.get_current_database() == DEFAULT_DATABASE_NAME


def test_does_database_exist(cloud_catalog):
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    assert cloud_catalog.does_database_exist(TEST_DATABASE_NAME)
    assert not cloud_catalog.does_database_exist("nonexistentdatabase")
    assert cloud_catalog.does_database_exist(TEST_DATABASE_NAME.lower())
    assert cloud_catalog.does_database_exist(TEST_DATABASE_NAME.upper())
    assert cloud_catalog.does_database_exist(TEST_VALID_COMPOSED_DB_NAME)
    assert not cloud_catalog.does_database_exist(
        "nonexistent_catalog." + TEST_DATABASE_NAME
    )


def test_does_database_exist_use_default_catalog(cloud_catalog):
    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    assert cloud_catalog.does_database_exist(DEFAULT_DATABASE_NAME)
    assert not cloud_catalog.does_database_exist("nonexistentdatabase")
    assert cloud_catalog.does_database_exist(DEFAULT_DATABASE_NAME.lower())
    assert cloud_catalog.does_database_exist(DEFAULT_DATABASE_NAME.upper())


def test_list_tables(cloud_catalog):
    # If not database is selected, then we should raise an error
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    assert cloud_catalog.list_tables() == [
        TEST_TABLE_NAME_1,
        TEST_TABLE_NAME_2,
    ]


def test_list_tables_use_default_catalog(cloud_catalog_default_catalog):
    # If not database is selected, then we should raise an error
    cloud_catalog_default_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    cloud_catalog_default_catalog.set_current_database(DEFAULT_DATABASE_NAME)
    assert cloud_catalog_default_catalog.list_tables() == [
        TEST_TABLE_NAME_DEFAULT_CATALOG
    ]


def test_does_table_exist(cloud_catalog):
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    assert cloud_catalog.does_table_exist(TEST_TABLE_NAME_1)
    assert cloud_catalog.does_table_exist(TEST_TABLE_NAME_2)
    assert cloud_catalog.does_table_exist(TEST_TABLE_NAME_1.lower())
    assert cloud_catalog.does_table_exist(TEST_TABLE_NAME_2.upper())
    assert not cloud_catalog.does_table_exist("nonexistent_table")
    assert cloud_catalog.does_table_exist(TEST_COMPOSED_TABLE_NAME_1)
    assert cloud_catalog.does_table_exist(TEST_COMPOSED_TABLE_NAME_2)

    assert not cloud_catalog.does_table_exist(
        f"nonexistent_catalog.{TEST_DATABASE_NAME}.{TEST_TABLE_NAME_1}"
    )

    assert not cloud_catalog.does_table_exist(
        f"{TEST_CATALOG_NAME}.nonexistent_database.{TEST_TABLE_NAME_1}"
    )


def test_get_table_schema(cloud_catalog):
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    schema = cloud_catalog.describe_table(TEST_TABLE_NAME_1)
    assert schema is not None
    assert len(schema.column_names()) == 3
    assert schema.column_names() == ["id", "name", "account_balance"]

    with pytest.raises(TableNotFoundError):
        cloud_catalog.describe_table("nonexistent_table")

def test_create_database(cloud_catalog): # noqa: D103
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    assert cloud_catalog.create_database(TEST_NEW_DATABASE_NAME)

    # The database already exists, so we should return False (default for ignore_if_exists is True)
    assert not cloud_catalog.create_database(TEST_DATABASE_NAME)

    # The database already exists, so we should raise an error if ignore_if_exists is False
    with pytest.raises(DatabaseAlreadyExistsError):
        cloud_catalog.create_database(TEST_DATABASE_NAME, ignore_if_exists=False)

    with pytest.raises(CatalogError) as e:
        cloud_catalog.create_database("some_catalog.some_database")
    assert (
        "Catalog some_catalog referenced by some_catalog.some_database does not exist."
        in str(e.value)
    )

    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    with pytest.raises(DatabaseAlreadyExistsError) as e:
        cloud_catalog.create_database(DEFAULT_DATABASE_NAME, ignore_if_exists=False)
    assert str(e.value) == "Database 'typedef_default' already exists"


def test_drop_database(cloud_catalog): # noqa: D103
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    assert cloud_catalog.drop_database(TEST_DATABASE_NAME)

    # Dropping a database that doesn't exist should return False if ignore_if_exists is True
    assert not cloud_catalog.drop_database(TEST_NEW_DATABASE_NAME)

    with pytest.raises(DatabaseNotFoundError):
        cloud_catalog.drop_database(TEST_NEW_DATABASE_NAME, ignore_if_not_exists=False)

    # Can't drop the current database
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    with pytest.raises(CatalogError):
        cloud_catalog.drop_database(TEST_DATABASE_NAME)

    # Can't drop the default database
    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    with pytest.raises(CatalogError):
        cloud_catalog.drop_database(DEFAULT_DATABASE_NAME)

    # Can't drop the default database using a fully qualified name.
    with pytest.raises(CatalogError):
        cloud_catalog.drop_database(f"{DEFAULT_CATALOG_NAME}.{DEFAULT_DATABASE_NAME}")

def test_create_table(cloud_catalog, schema): # noqa: D103
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    assert cloud_catalog.create_table(
        TEST_NEW_TABLE_NAME, schema, TEST_SAMPLE_LOCATION
    )
    # TEST_TABLE_NAME_1 already exists, either can return False or raise an error.
    assert not cloud_catalog.create_table(
        TEST_TABLE_NAME_1, schema, TEST_SAMPLE_LOCATION
    )
    with pytest.raises(TableAlreadyExistsError):
        cloud_catalog.create_table(
            TEST_TABLE_NAME_1,
            schema=schema,
            location=TEST_SAMPLE_LOCATION,
            ignore_if_exists=False,
        )
    with pytest.raises(CatalogError):
        cloud_catalog.create_table(
            "some_catalog.some_database.some_table",
            schema=schema,
            location=TEST_SAMPLE_LOCATION,
        )


def test_drop_table(cloud_catalog): # noqa: D103
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    cloud_catalog.set_current_database(TEST_DATABASE_NAME)
    assert cloud_catalog.drop_table(TEST_TABLE_NAME_1)
    assert not cloud_catalog.drop_table(TEST_NEW_TABLE_NAME)

    with pytest.raises(TableNotFoundError):
        cloud_catalog.drop_table(TEST_NEW_TABLE_NAME, ignore_if_not_exists=False)

    with pytest.raises(CatalogError):
        cloud_catalog.drop_table(
            f"some_catalog.{TEST_DATABASE_NAME}.{TEST_TABLE_NAME_1}",
            ignore_if_not_exists=False,
        )

    with pytest.raises(CatalogError):
        cloud_catalog.drop_table(
            f"{TEST_CATALOG_NAME}.some_db.{TEST_TABLE_NAME_1}",
            ignore_if_not_exists=False,
        )


def test_create_catalog(cloud_catalog): # noqa: D103
    with pytest.raises(CatalogError):
        cloud_catalog.create_catalog(DEFAULT_CATALOG_NAME)

    assert cloud_catalog.create_catalog(TEST_NEW_CATALOG_NAME)

    # The catalog already exists, so we should return False (default for ignore_if_exists is True)
    assert not cloud_catalog.create_catalog(TEST_CATALOG_NAME)

    # The catalog already exists, so we should raise an error if ignore_if_exists is False
    with pytest.raises(CatalogAlreadyExistsError):
        cloud_catalog.create_catalog(TEST_CATALOG_NAME, ignore_if_exists=False)


def test_drop_catalog(cloud_catalog): # noqa: D103
    # can't drop the default catalog
    with pytest.raises(CatalogError):
        cloud_catalog.drop_catalog(DEFAULT_CATALOG_NAME)

    # can't drop a catalog that doesn't exist
    assert not cloud_catalog.drop_catalog(TEST_NEW_CATALOG_NAME)
    with pytest.raises(CatalogNotFoundError):
        cloud_catalog.drop_catalog(TEST_NEW_CATALOG_NAME, ignore_if_not_exists=False)

    # can't drop the current catalog
    cloud_catalog.set_current_catalog(TEST_CATALOG_NAME)
    with pytest.raises(CatalogError):
        cloud_catalog.drop_catalog(TEST_CATALOG_NAME)

    cloud_catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    assert cloud_catalog.drop_catalog(TEST_CATALOG_NAME)

def _init_cloud_catalog(
    session_state: Any, client: Client, cloud_session_manager: Any
) -> Any:
    os.environ["TYPEDEF_USER_ID"] = "mock_user_id"
    os.environ["TYPEDEF_USER_SECRET"] = "mock_user_secret"  # nosec B105
    os.environ["HASURA_GRAPHQL_ADMIN_SECRET"] = "mock_admin_secret"  # nosec B105
    os.environ["REMOTE_SESSION_AUTH_PROVIDER_URI"] = "mock_auth_provider_uri"
    cloud_catalog = CloudCatalog(session_state, cloud_session_manager)
    cloud_catalog.user_client = client
    cloud_catalog.user_id = TEST_DEFAULT_USER_ID
    cloud_catalog.organization_id = TEST_DEFAULT_ORGANIZATION_ID
    return cloud_catalog


def _init_mock_state(ev_loop: asyncio.AbstractEventLoop) -> Any:
    """Initialize a mock session state with a given event loop."""
    mock_state = MagicMock(spec=CloudSessionState)
    mock_state.app_name = TEST_DEFAULT_APP_NAME
    mock_state.ephemeral_catalog_id = TEST_EPHMERAL_CATALOG_ID
    mock_state.asyncio_loop = ev_loop
    mock_state.lock = threading.Lock()

    # global background_thread
    background_thread = threading.Thread(target=ev_loop.run_forever, daemon=True)
    background_thread.start()

    return mock_state
