from fenic.api.session.session import Session
from fenic.core.types.datatypes import IntegerType
from fenic.core.types.schema import ColumnField, Schema


def create_table_with_rows(session: Session, name: str, values: list[int], description: str | None = None) -> None:
    df = session.create_dataframe({"id": values})
    # Persist table and optional description through writer (threads description into TableSink)
    if description is not None:
        df.write.save_as_table(name, mode="overwrite")
        session.catalog.set_table_description(name, description)
    else:
        # No description: create an empty table with schema only
        session.catalog.create_table(name, Schema([ColumnField("id", IntegerType)]))
