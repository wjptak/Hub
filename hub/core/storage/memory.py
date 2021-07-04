from hub.core.storage.provider import StorageProvider

from hub.core.storage.cachable import Cachable


class MemoryProvider(StorageProvider):
    """Provider class for using the memory."""

    def __init__(self, root=""):
        self.dict = {}
        self.root = root

    def __getitem__(
        self,
        path: str,
    ):
        """Gets the object present at the path within the given byte range.

        Example:
            memory_provider = MemoryProvider("xyz")
            my_data = memory_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """
        return self.dict[path]

    def __setitem__(
        self,
        path: str,
        value: bytes,
    ):
        """Sets the object present at the path with the value

        Example:
            memory_provider = MemoryProvider("xyz")
            memory_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self.dict[path] = value

    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:
            memory_provider = MemoryProvider("xyz")
            for my_data in memory_provider:
                pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self.dict

    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:
            memory_provider = MemoryProvider("xyz")
            del memory_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        del self.dict[path]

    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Example:
            memory_provider = MemoryProvider("xyz")
            len(memory_provider)

        Returns:
            int: the number of files present inside the root.
        """
        return len(self.dict)

    def clear(self):
        """Clears the provider."""
        self.check_readonly()
        self.dict = {}

    def get_cachable(self, path: str, expected_class):
        item = self[path]

        if isinstance(item, Cachable):
            if type(item) != expected_class:
                raise ValueError(
                    f"'{path}' was expected to have the class '{expected_class.__name__}'. Instead, got: '{type(item)}'."
                )
            return item

        if isinstance(item, (bytes, memoryview)):
            obj = expected_class.frombuffer(item)
            if len(obj) <= self.cache_size:
                self._insert_in_cache(path, obj)
            return obj

        raise ValueError(f"Item at '{path}' got an invalid type: '{type(item)}'.")
