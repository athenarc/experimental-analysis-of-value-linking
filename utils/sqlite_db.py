import os
from collections import defaultdict

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text,exc
from sqlalchemy.exc import SQLAlchemyError
from sqlglot import parse_one
from func_timeout import func_timeout, FunctionTimedOut
import time

class DatabaseSqlite:
    def __init__(self, database: str):
        """.
        Initialize the database connector for SQLite by providing the path to the database file.
        Note: SQLite does not have all the functionalities provided by the Database connector.

        Args:
            database: the database path i.e. "path/to/database.db"
        """
        if not os.path.exists(database):
            raise ValueError(f'Sqlite database file "{database}" does not exist.')
        self.connection_uri = f"sqlite:///{database}"
        self.engine = create_engine(self.connection_uri)

    def execute(self, sql: str, limit: int = 500, timeout_seconds: int = 100) -> pd.DataFrame | dict:
        """
        Execute a given SQL query with a timeout using func-timeout.
        Returns an empty DataFrame on timeout instead of an error dictionary.

        Args:
            sql: the sql query
            limit: the limit of the number of rows to return
            timeout_seconds: Timeout in seconds for the query execution using func-timeout

        Returns:
            results: the results of the query, an empty DataFrame on timeout,
                     or a dictionary with an error message for other errors.
        """
        pars = parse_one(sql, dialect="sqlite") # Assuming parse_one is defined elsewhere
        if limit not in (-1, 0):
            pars = pars.limit(limit)
        sql = pars.sql(dialect="sqlite")

        try:
            df = func_timeout(timeout_seconds, self._read_sql_with_conn, args=(sql,)) # Apply timeout here
            return df
        except FunctionTimedOut:
            logger.warning(f"Query timeout after {timeout_seconds} seconds using func-timeout for query: {sql}. Returning empty DataFrame.") # Log as warning, not error, as it's handled
            return pd.DataFrame()  # Return an empty DataFrame here
        except exc.SQLAlchemyError as e:
            logger.error(f"sqlalchemy error {str(e.__dict__['orig'])}")
            return {"error": str(e.__dict__["orig"])}
        except Exception as e:
            logger.error(f"General error: {e} for query {sql}")
            return {"error": "Something went wrong with your query."}

    def _read_sql_with_conn(self, sql):
        """Helper function to encapsulate read_sql and connection logic."""
        try:
            with self.engine.begin() as conn:
                df = pd.read_sql(text(sql), con=conn)
            return df
        finally:
            if 'conn' in locals() and conn:
                conn.close()
            self.engine.dispose()
            
    def get_tables_and_columns(self):
        tables_cols_df = self.execute(
            """
            SELECT m.name as tableName,
                   p.name as columnName
            FROM sqlite_master m
            left outer join pragma_table_info((m.name)) p
                 on m.name <> p.name
            order by tableName, columnName
            ;
        """
        )

        res = {"tables": tables_cols_df["tableName"].unique(), "columns": []}
        for _, row in tables_cols_df.iterrows():
            if row["columnName"] and row["tableName"]:
                res["columns"].append(row["tableName"] + "." + row["columnName"])

        return res

    def get_types_of_db(self) -> dict:
        """
        Return the types of the columns of the database
        """
        types_table = self.execute(
            """
            SELECT m.name AS table_name,
              p.name AS column_name, p.type AS data_type
            FROM sqlite_master AS m
              INNER JOIN pragma_table_info(m.name) AS p
            WHERE m.name NOT IN ('sqlite_sequence')
            """
        )
        ret_types = defaultdict(dict)
        for _, row in types_table.iterrows():
            table, column, data_type = row
            ret_types[table][column] = data_type

        return ret_types

    def get_primary_keys(self) -> dict:
        """
        Return the primary keys of the database
        """
        pks_table = self.execute(
            """
            SELECT m.name AS table_name,
              p.name AS column_name
            FROM sqlite_master AS m
              INNER JOIN pragma_table_info(m.name) AS p
            WHERE m.name NOT IN ('sqlite_sequence')
                AND p.pk != 0
            ORDER BY m.name, p.cid
            """
        )

        ret_pks = defaultdict(list)
        for _, row in pks_table.iterrows():
            table, column = row
            ret_pks[table].append(column)

        return ret_pks

    def get_foreign_keys(self) -> dict:
        """
        Return the foreign keys of the database
        """
        foreign_keys = self.execute(
            """
            SELECT
                m.tbl_name AS table_name,
                p.'from' AS column_name,
                p.'table' AS foreign_table_name,
                p.'to' AS foreign_column_name
            FROM sqlite_master AS m
                  INNER JOIN pragma_foreign_key_list(m.name) AS p
            WHERE m.name NOT IN ('sqlite_sequence');
        """
        )
        ret_foreign_keys = {}
        for _, row in foreign_keys.iterrows():
            table, column, foreign_table, foreign_column = row

            if None in (table, column, foreign_table, foreign_column):
                # NOTE: Sometimes foreign_column is None, which cause errors later on
                continue

            if table in ret_foreign_keys and column in ret_foreign_keys[table]:
                ret_foreign_keys[table][column].append(
                    {
                        "foreign_table": foreign_table,
                        "foreign_column": foreign_column,
                    }
                )
            else:
                ret_foreign_keys[table] = {
                    column: [
                        {
                            "foreign_table": foreign_table,
                            "foreign_column": foreign_column,
                        }
                    ]
                }

        return ret_foreign_keys

