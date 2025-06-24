import os

import psycopg2


def read_db_params_from_env(schema: str):
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


def get_db_connection(schema: str = "default"):
    """
    Parameters:
        schema (str): The schema to connect to. Defaults to the environment variable SCHEMA, or "public" if not set.
                      Valid schemas are "public" and "berlin"
    """

    # Database credentials
    DB_PARAMS = read_db_params_from_env(schema)

    # Connect to the database
    try:
        conn = psycopg2.connect(**DB_PARAMS)
    except Exception as e:
        print(f"Error Connectiong to DB: {e}")
        exit()

    # Create a cursor to execute queries
    return conn.cursor()
