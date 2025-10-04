import os
import logging
import psycopg2
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _read_db_params_from_env(schema: str):
    uri = os.getenv("POSTGRESQL_URI")  # should be of format <user>:<password>@<ip>:<port>/<database>
    if schema == "default":
        schema = os.getenv("SCHEMA") or "public"

    if uri is None:
        raise ValueError("POSTGRESQL_URI environment variable is not set")

    return {
        "dbname": uri.split("/")[-1],
        "user": uri.split(":")[0],
        "password": uri.split(":")[1].split("@")[0],
        "host": uri.split("@")[1].split(":")[0],
        "port": uri.split(":")[2].split("/")[0],
        "options": f"-c search_path={schema}",
    }


@contextmanager
def get_db_connection(schema: str = None):
    """
    Provides a database connection as a context manager.
    Ensures the connection is always closed.

    Args:
        schema (str, optional): The schema to use. Defaults to env var 'SCHEMA' or 'public'.

    Yields:
        A database cursor object.
    """
    conn = None
    try:
        db_params = _read_db_params_from_env(schema)
        conn = psycopg2.connect(**db_params)
        yield conn.cursor()
    except Exception as e:
        logging.error(f"Database connection or operation failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
