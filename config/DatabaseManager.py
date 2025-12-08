import configparser
import contextlib
import logging
from pathlib import Path
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursorDict
from mysql.connector import pooling


class DatabaseConnector:
    def __init__(
        self,
        name="DatabaseConnector",
        logger: logging.Logger = None,
        retries=10,
        delay=30,
        connection_timeout=10 * 3600,
    ):
        self.name = name
        self.connection_timeout = connection_timeout
        config_path = Path("config/config.ini")
        config_ = configparser.ConfigParser(interpolation=None)
        config_.read(config_path)
        env = config_["ENV"]["ENV"]
        self.env = env
        env = config_[f"MIDORI-{self.env}"]
        self.db_config = {
            "host": env["HOST"],
            "database": env["MYDB"],
            "user": env["USER"],
            "password": env["PASS"],
            "port": int(env["PORT"]),
            "connection_string": f"mysql://{env['USER']}:{env['PASS']}@{env['HOST']}:{int(env['PORT'])}/{env['MYDB']}",
        }
        self.delay = delay
        self.retries = retries
        self.port = int(env["PORT"])
        self.dbname = env["MYDB"]
        self.passwd = env["PASS"]
        self.host = env["HOST"]
        self.user = env["USER"]
        self.base_url = env["BASE_EXT_URL"]
        self.connection = None
        self.connector = None
        self.connection_string = (
            f"mysql://{self.user}:{self.passwd}@{self.host}:{self.port}/{self.dbname}"
        )

        self.pool = pooling.MySQLConnectionPool(
            pool_name="ned_sql_pool",
            pool_size=5,  # Adjust based on concurrency
            pool_reset_session=True,
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            database=self.dbname,
            port=self.port,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb is None:
            self.connector.commit()
        else:
            self.connector.rollback()
        self.connector.close()

    @contextlib.contextmanager
    def __connect__(self) -> MySQLConnection:
        connection_ = None
        try:
            # Get connection from pool instead of creating new one
            connection_ = self.pool.get_connection()
            if connection_.is_connected():
                print("MYSQL-Connection: acquired from pool")
        except Exception as err:
            print(f"Error getting connection: {err}")
            raise

        try:
            yield connection_
        except Exception:
            if connection_:
                connection_.rollback()
            raise
        else:
            if connection_:
                connection_.commit()
        finally:
            if connection_:
                # This doesn't close TCP, it returns to pool
                connection_.close()

    @contextlib.contextmanager
    def execute_query(self) -> MySQLCursorDict:
        """Yields a cursor, handles connection/commit automatically."""
        # Uses the __connect__ logic internally
        with self.__connect__() as conn:
            cursor: MySQLCursorDict = conn.cursor(dictionary=True, buffered=True)
            try:
                yield cursor
            finally:
                cursor.close()


if __name__ == "__main__":
    DatabaseConnector()
