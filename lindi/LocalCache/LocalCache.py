from typing import Union
import os


class LocalCache:
    def __init__(self, *, cache_dir: Union[str, None] = None):
        if cache_dir is None:
            # use ~/.lindi/cache as default cache directory
            cache_dir = os.path.expanduser("~/.lindi/cache")
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)
        self._sqlite_db_fname = os.path.join(self._cache_dir, "lindi_cache.db")
        self._sqlite_client = LocalCacheSQLiteClient(db_fname=self._sqlite_db_fname)

    def get_remote_chunk(self, *, url: str, offset: int, size: int):
        return self._sqlite_client.get_remote_chunk(url=url, offset=offset, size=size)

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes):
        if len(data) != size:
            raise ValueError("data size does not match size")
        self._sqlite_client.put_remote_chunk(url=url, offset=offset, size=size, data=data)


class ChunkTooLargeError(Exception):
    pass


class LocalCacheSQLiteClient:
    def __init__(self, *, db_fname: str):
        import sqlite3
        self._db_fname = db_fname
        self._conn = sqlite3.connect(self._db_fname)
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            """
            PRAGMA journal_mode=WAL
            """
        )
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS remote_chunks (
                url TEXT,
                offset INTEGER,
                size INTEGER,
                data BLOB,
                PRIMARY KEY (url, offset, size)
            )
            """
        )
        self._conn.commit()

    def get_remote_chunk(self, *, url: str, offset: int, size: int):
        self._cursor.execute(
            """
            SELECT data FROM remote_chunks WHERE url = ? AND offset = ? AND size = ?
            """,
            (url, offset, size),
        )
        row = self._cursor.fetchone()
        if row is None:
            return None
        return row[0]

    def put_remote_chunk(self, *, url: str, offset: int, size: int, data: bytes):
        if size >= 1000 * 1000 * 900:
            # This is a sqlite limitation/configuration
            # https://www.sqlite.org/limits.html
            # For some reason 1000 * 1000 * 1000 seems to be too large, whereas 1000 * 1000 * 900 is fine
            raise ChunkTooLargeError("Cannot store blobs larger than 900 MB in LocalCache")
        self._cursor.execute(
            """
            INSERT OR REPLACE INTO remote_chunks (url, offset, size, data) VALUES (?, ?, ?, ?)
            """,
            (url, offset, size, data),
        )
        self._conn.commit()
