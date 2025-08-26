
import pytest

from fenic import (
    ColumnField,
    IntegerType,
    Schema,
    Session,
    StringType,
    StructField,
    StructType,
)
from fenic._backends.local.catalog import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_DATABASE_NAME,
)
from fenic.core.error import (
    CatalogError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    TableAlreadyExistsError,
    TableNotFoundError,
)

NON_EXISTING_CATALOG_NAME = "non_existing_catalog"
NON_EXISTING_DATABASE_NAME = "non_existing_db"
A_DATABASE_NAME = "test_db"
ANOTHER_DATABASE_NAME = "another_existing_db"
NON_EXISTING_TABLE_NAME = "non_existing_table"
TABLE_NAME_T1 = "test_table_t1"
TABLE_NAME_T2 = "test_table_t2"
TABLE_NAME_QUALIFIED = f"{DEFAULT_DATABASE_NAME}.qualified_table"
TABLE_NAME_WITH_CATALOG = f"not_typedef_default.{DEFAULT_DATABASE_NAME}.catalog_table"
TABLE_NAME_STRUCT = "test_struct_table"
SIMPLE_TABLE_SCHEMA = Schema([ColumnField("id", IntegerType)])
SIMPLE_TABLE_SCHEMA_2 = Schema([ColumnField("name", StringType)])
STRUCT_TABLE_SCHEMA = Schema(
    [
        ColumnField(
            "s1",
            StructType(
                [
                    StructField("i", IntegerType),
                    StructField("j", IntegerType),
                ]
            ),
        )
    ]
)


def test_does_catalog_exist(local_session: Session):
    assert local_session.catalog.does_catalog_exist(DEFAULT_CATALOG_NAME)
    assert not local_session.catalog.does_catalog_exist(NON_EXISTING_CATALOG_NAME)
    assert local_session.catalog.does_catalog_exist(DEFAULT_CATALOG_NAME.upper())

def test_get_current_catalog(local_session: Session):
    assert local_session.catalog.get_current_catalog() == DEFAULT_CATALOG_NAME


def test_set_current_catalog(local_session: Session):
    with pytest.raises(
        CatalogError,
        match="Invalid catalog name 'non_existing_catalog'. Only the default catalog 'typedef_default' is supported in local execution mode.",
    ):
        local_session.catalog.set_current_catalog(NON_EXISTING_CATALOG_NAME)
    # No actual change expected for the default catalog
    local_session.catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    assert local_session.catalog.get_current_catalog() == DEFAULT_CATALOG_NAME


def test_list_catalogs(local_session: Session):
    assert local_session.catalog.list_catalogs() == [DEFAULT_CATALOG_NAME]


def test_does_database_exist(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(NON_EXISTING_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME.upper())
    assert not local_session.catalog.does_database_exist(NON_EXISTING_DATABASE_NAME.upper())


def test_create_database(local_session: Session):
    assert local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.create_database(
        A_DATABASE_NAME, ignore_if_exists=True
    )
    assert not local_session.catalog.create_database(
        A_DATABASE_NAME.upper(), ignore_if_exists=True
    )
    with pytest.raises(
        DatabaseAlreadyExistsError, match=f"Database '{A_DATABASE_NAME}' already exists"
    ):
        local_session.catalog.create_database(A_DATABASE_NAME, ignore_if_exists=False)
    with pytest.raises(
        DatabaseAlreadyExistsError, match=f"Database '{A_DATABASE_NAME.upper()}' already exists"
    ):
        local_session.catalog.create_database(A_DATABASE_NAME.upper(), ignore_if_exists=False)


def test_drop_database(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert local_session.catalog.drop_database(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(A_DATABASE_NAME.upper())
    assert not local_session.catalog.drop_database(NON_EXISTING_DATABASE_NAME)
    assert not local_session.catalog.drop_database(
        NON_EXISTING_DATABASE_NAME, ignore_if_not_exists=True
    )
    with pytest.raises(
        DatabaseNotFoundError,
        match=f"Database '{NON_EXISTING_DATABASE_NAME}' does not exist",
    ):
        local_session.catalog.drop_database(
            NON_EXISTING_DATABASE_NAME, ignore_if_not_exists=False
        )

    local_session.catalog.create_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(
        CatalogError,
        match=f"Cannot drop the current database '{DEFAULT_DATABASE_NAME}'. Switch to another database first.",
    ):
        local_session.catalog.drop_database(DEFAULT_DATABASE_NAME)

    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(CatalogError):
        local_session.catalog.drop_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=True)
    assert not local_session.catalog.does_database_exist(ANOTHER_DATABASE_NAME)


    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    local_session.create_dataframe({"a": [1, 2, 3]}).write.save_as_view("df1")
    assert local_session.catalog.does_view_exist("df1")
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(CatalogError, match="Cannot drop database 'another_existing_db' because it contains views. Use CASCADE to drop the database and all its views."):
        local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=False)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    assert local_session.catalog.does_view_exist("df1")
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=True)
    assert not local_session.catalog.does_database_exist(ANOTHER_DATABASE_NAME)

def test_list_databases(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    databases = local_session.catalog.list_databases()
    assert A_DATABASE_NAME in databases
    assert ANOTHER_DATABASE_NAME in databases
    assert DEFAULT_DATABASE_NAME in databases
    assert len(databases) == 3


def test_get_and_set_current_database(local_session: Session):
    # create databases
    local_session.catalog.create_database(A_DATABASE_NAME)
    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)

    local_session.catalog.set_current_database(A_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == A_DATABASE_NAME
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == DEFAULT_DATABASE_NAME
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == ANOTHER_DATABASE_NAME


def test_does_table_exist(local_session: Session):
    # create table
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    # assert table exists
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}"
    )
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{TABLE_NAME_T1}"
    )
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{TABLE_NAME_T1}".upper()
    )
    assert not local_session.catalog.does_table_exist(NON_EXISTING_TABLE_NAME)
    assert not local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}"
    )

def test_does_view_exist(local_session: Session):
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)

    view_name = "df1"
    non_existing_view_name = "df2"

    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view(view_name)

    assert local_session.catalog.does_view_exist(view_name)
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME}.{view_name}"
    )
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{view_name}"
    )
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{view_name}".upper()
    )
    assert not local_session.catalog.does_view_exist(non_existing_view_name)
    assert not local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME}.{non_existing_view_name}"
    )

def test_list_tables(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    local_session.catalog.create_table(TABLE_NAME_T2, SIMPLE_TABLE_SCHEMA_2)
    tables = local_session.catalog.list_tables()
    assert TABLE_NAME_T1 in tables
    assert TABLE_NAME_T2 in tables

def test_list_views(local_session: Session):
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view("df1")

    df2 = local_session.create_dataframe({"b": [1, 2, 3]})
    df2.write.save_as_view("df2")

    views = local_session.catalog.list_views()
    assert "df1" in views
    assert "df2" in views

def test_describe_view(local_session: Session):
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view("df1")
    view_df1 = local_session._session_state.catalog.describe_view("df1")
    assert view_df1.schema().column_names() == ["a"]

def test_describe_table(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    schema = local_session.catalog.describe_table(TABLE_NAME_T1)
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "id"
    assert schema.column_fields[0].data_type == IntegerType

    schema = local_session.catalog.describe_table(
        f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}"
    )
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "id"
    assert schema.column_fields[0].data_type == IntegerType

    with pytest.raises(
        TableNotFoundError,
        match=f"Table 'typedef_default.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.describe_table(NON_EXISTING_TABLE_NAME)
    with pytest.raises(
        TableNotFoundError,
        match=f"Table '{A_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.describe_table(
            f"{A_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}"
        )


def test_describe_table_struct(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_STRUCT, STRUCT_TABLE_SCHEMA)
    schema = local_session.catalog.describe_table(TABLE_NAME_STRUCT)
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "s1"
    assert isinstance(schema.column_fields[0].data_type, StructType)
    inner_schema = schema.column_fields[0].data_type
    assert len(inner_schema.struct_fields) == 2
    assert inner_schema.struct_fields[0].name == "i"
    assert inner_schema.struct_fields[0].data_type == IntegerType
    assert inner_schema.struct_fields[1].name == "j"
    assert inner_schema.struct_fields[1].data_type == IntegerType


def test_drop_table(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.drop_table(TABLE_NAME_T1)
    assert not local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.drop_table(f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}")
    assert not local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert not local_session.catalog.drop_table(NON_EXISTING_TABLE_NAME)
    assert not local_session.catalog.drop_table(
        NON_EXISTING_TABLE_NAME, ignore_if_not_exists=True
    )
    with pytest.raises(
        TableNotFoundError,
        match=f"Table 'typedef_default.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.drop_table(
            NON_EXISTING_TABLE_NAME, ignore_if_not_exists=False
        )

def test_drop_view(local_session: Session):
    view_name_1 = "df1"
    view_name_2 = "df2"
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view(view_name_1)

    assert local_session.catalog.does_view_exist(view_name_1)
    assert local_session.catalog.drop_view(view_name_1)
    assert not local_session.catalog.does_view_exist(view_name_1)

    df1.write.save_as_view(view_name_2)
    assert local_session.catalog.does_view_exist(view_name_2)
    assert local_session.catalog.drop_view(f"{DEFAULT_DATABASE_NAME}.{view_name_2}")
    assert not local_session.catalog.does_view_exist(view_name_2)

    with pytest.raises(
        TableNotFoundError,
        match="Table 'typedef_default.df3' does not exist",
    ):
        local_session.catalog.drop_view(
            "typedef_default.df3", ignore_if_not_exists=False
        )

def test_create_table(local_session: Session):
    assert local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert not local_session.catalog.create_table(
        TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA, ignore_if_exists=True
    )
    with pytest.raises(
        TableAlreadyExistsError,
        match=f"Table 'typedef_default.{TABLE_NAME_T1}' already exists",
    ):
        local_session.catalog.create_table(
            TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA, ignore_if_exists=False
        )

    assert local_session.catalog.create_table(TABLE_NAME_QUALIFIED, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_QUALIFIED)

    with pytest.raises(
        CatalogError,
        match="Invalid catalog name 'not_typedef_default'",
    ):
        local_session.catalog.create_table(TABLE_NAME_WITH_CATALOG, SIMPLE_TABLE_SCHEMA)
