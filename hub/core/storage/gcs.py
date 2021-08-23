import posixpath
from typing import Dict, Union

from google.auth import credentials
from hub.core.storage.provider import StorageProvider
from google.cloud import storage  # type: ignore


class GCSProvider(StorageProvider):
    """Provider class for using GC storage."""

    def __init__(
        self,
        root: str,
        token: Union[str, Dict] = None,
    ):
        """Initializes the GCSProvider

        Example:
            gcs_provider = GCSProvider("snark-test/gcs_ds")

        Args:
            root (str): The root of the provider. All read/write request keys will be appended to root.
            token (str/Dict): GCP token, used for fetching credentials for storage).
        """
        self.root = root
        self.token: Union[str, Dict, None] = token
        self.missing_exceptions = (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            AttributeError,
        )
        self._initialize_provider()

    def _initialize_provider(self):
        self._set_bucket_and_path()
        # self.fs = gcsfs.GCSFileSystem(token=self.token)
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(self.token)

        scoped_credentials = credentials.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.client = storage.Client(credentials=scoped_credentials)
        self.client_bucket = self.client.get_bucket(self.bucket)

    def _set_bucket_and_path(self):
        root = self.root.replace("gcp://", "").replace("gcs://", "")
        self.bucket = root.split("/")[0]
        self.path = root
        if not self.path.endswith("/"):
            self.path += "/"

    def _get_path_from_key(self, key):
        return posixpath.join(self.path, key)

    def _list_keys(self):
        self._blob_objects = self.client_bucket.list_blobs(prefix=self.path)
        return [obj.name for obj in self._blob_objects]

    def clear(self):
        """Remove all keys below root - empties out mapping"""
        self.check_readonly()
        blob_objects = self.client_bucket.list_blobs(prefix=self.path)

        for obj in blob_objects:
            obj.delete()

    def __getitem__(self, key):
        """Retrieve data"""
        try:
            # with self.fs.open(posixpath.join(self.path, key), "rb") as f:
            #     return f.read()
            blob = self.client_bucket.get_blob(self._get_path_from_key(key))
            return blob.download_as_bytes()
        except self.missing_exceptions:
            raise KeyError(key)

    def __setitem__(self, key, value):
        """Store value in key"""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        with blob.open("wb") as f:
            f.write(value)

    def __iter__(self):
        """Iterating over the structure"""
        yield from [f for f in self._list_keys() if not f.endswith("/")]

    def __len__(self):
        """Returns length of the structure"""
        return len(self._list_keys())

    def __delitem__(self, key):
        """Remove key"""
        self.check_readonly()
        blob = self.client_bucket.blob(self._get_path_from_key(key))
        blob.delete()

    def __contains__(self, key):
        """Does key exist in mapping?"""
        path = posixpath.join(self.path, key)
        return self.fs.exists(path)
