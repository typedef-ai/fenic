from fenic import ColumnField, JsonType, MarkdownType, Schema, Session

SCHEMA = Schema(
    column_fields=[
        ColumnField(name="c1", data_type=MarkdownType),
        ColumnField(name="c2", data_type=JsonType),
    ],
)


def test_single_table_schema_storage(local_session: Session):
    schema_storage = local_session.catalog.catalog.system_tables
    schema_storage.save_schema("test_database", "test_table", SCHEMA)
    retrieved_schema = schema_storage.get_schema("test_database", "test_table")
    assert retrieved_schema == SCHEMA
    assert schema_storage.delete_schema("test_database", "test_table")
    assert not schema_storage.get_schema("test_database", "test_table")


def test_multiple_table_schema_storage(local_session: Session):
    schema_storage = local_session.catalog.catalog.system_tables
    schema_storage.save_schema("test_database", "test_table_1", SCHEMA)
    schema_storage.save_schema("test_database", "test_table_2", SCHEMA)
    schema_storage.save_schema("test_database2", "test_table_1", SCHEMA)
    assert schema_storage.delete_database_schemas("test_database") == 2
    assert schema_storage.delete_database_schemas("test_database2") == 1
    assert schema_storage.delete_database_schemas("test_database3") == 0
