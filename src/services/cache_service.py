# src/services/cache_service.py
import modal
from typing import Any, Optional

class CacheService:
    def __init__(self):
        self.cache = None

    def get_cache(self, cache_name: str):
        if self.cache is None:
            print("Creating config cache for Agent")
            self.cache = modal.Dict.from_name(cache_name, create_if_missing=True)
        return self.cache

    def get(self, cache_name: str, key: str) -> Optional[Any]:
        cache = self.get_cache(cache_name)
        return cache.get(key)

    def set(self, cache_name: str, key: str, value: Any):
        cache = self.get_cache(cache_name)
        cache[key] = value