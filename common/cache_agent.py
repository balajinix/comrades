"""
Cache Agent for GraphQL Queries
This module provides caching functionality for GraphQL queries.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphQLCacheAgent:
    """
    Agent for caching GraphQL queries.
    """

    def __init__(self, cache_dir: str = "./data/cache", cache_file: str = "graphql_query_cache.json"):
        """
        Initialize the GraphQL Cache Agent.

        Args:
            cache_dir: Directory for the cache file
            cache_file: Name of the cache file
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, cache_file)
        os.makedirs(cache_dir, exist_ok=True)

    def load_cache(self) -> Dict[str, Any]:
        """
        Load the query cache from a JSON file.

        Returns:
            Dictionary of cached queries
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading GraphQL query cache: {e}")
            return {}

    def save_cache(self, cache: Dict[str, Any]) -> None:
        """
        Save the query cache to a JSON file.

        Args:
            cache: Dictionary of cached queries
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving GraphQL query cache: {e}")

    def get_query(self, user_query: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Get a cached GraphQL query for a natural language query.

        Args:
            user_query: Natural language query

        Returns:
            Tuple of (query_text, variables, schema_content)
        """
        cache = self.load_cache()

        # Exact match first
        if user_query in cache:
            entry = cache[user_query]

            # Check if the entry has the expected structure
            if not self.validate_cache_entry(entry):
                logger.warning(f"Invalid cache entry for query: {user_query}")
                return None, None, None

            return (
                entry.get("final_graphql"),
                entry.get("variables", {}),
                entry.get("schema_content")
            )

        # Try to find a close match by checking if query contains similar keywords
        #user_query_words = set(user_query.lower().split())
        #for cached_query, entry in cache.items():
        #    cached_query_words = set(cached_query.lower().split())
        #    # If there's significant overlap in the keywords (>50%)
        #    overlap = len(user_query_words.intersection(cached_query_words))
        #    if overlap > 0 and overlap >= len(cached_query_words) * 0.5:
        #        # Check if the entry has the expected structure
        #        if not self.validate_cache_entry(entry):
        #            logger.warning(f"Invalid cache entry for query: {cached_query}")
        #            continue

        #        return (
        #            entry.get("final_graphql"),
        #            entry.get("variables", {}),
        #            entry.get("schema_content")
        #        )

        return None, None, None

    def validate_cache_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate that a cache entry has the expected structure.

        Args:
            entry: Cache entry to validate

        Returns:
            True if the entry is valid, False otherwise
        """
        required_fields = ["final_graphql"]

        # Check if all required fields are present
        for field in required_fields:
            if field not in entry or entry[field] is None:
                return False

        return True

    def cache_query(
        self,
        user_query: str, 
        final_graphql: str, 
        variables: Dict[str, Any],
        schema_content: Optional[str] = None
    ) -> None:
        """
        Cache a GraphQL query for a natural language query.

        Args:
            user_query: Natural language query
            final_graphql: Final GraphQL query
            variables: Variables for the query
            schema_content: Schema content
        """
        cache = self.load_cache()

        cache[user_query] = {
            "final_graphql": final_graphql,
            "variables": variables,
            "schema_content": schema_content,
            "timestamp": datetime.now().isoformat()
        }

        self.save_cache(cache)

    def list_queries(self) -> List[Dict[str, Any]]:
        """
        List all cached queries.

        Returns:
            List of query entries with metadata
        """
        cache = self.load_cache()
        queries_list = []

        for query, data in cache.items():
            queries_list.append({
                "query": query,
                "timestamp": data.get("timestamp", ""),
                "has_variables": bool(data.get("variables")),
                "schema_file": data.get("schema_file", "")
            })

        return queries_list

    def delete_query(self, query: Optional[str] = None) -> int:
        """
        Delete a specific query or all queries.

        Args:
            query: The query to delete. If None, delete all queries.

        Returns:
            Number of deleted queries
        """
        cache = self.load_cache()
        original_count = len(cache)

        if query:
            # Delete specific query
            if query in cache:
                del cache[query]
                self.save_cache(cache)
                return 1
            else:
                return 0
        else:
            # Delete all queries
            self.save_cache({})
            return original_count

    def get_query_details(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific cached query.

        Args:
            query: The query to get details for

        Returns:
            The query entry or None if not found
        """
        cache = self.load_cache()
        return cache.get(query)

    def invalidate_cache_entry(self, user_query: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            user_query: Natural language query to invalidate

        Returns:
            True if entry was invalidated, False if not found
        """
        cache = self.load_cache()

        if user_query in cache:
            del cache[user_query]
            self.save_cache(cache)
            logger.info(f"Invalidated cache entry for query: {user_query}")
            return True

        return False

    def validate_all_cache_entries(self) -> Tuple[int, int]:
        """
        Validate all cache entries and remove invalid ones.

        Returns:
            Tuple of (total entries, invalid entries removed)
        """
        cache = self.load_cache()
        total_entries = len(cache)
        invalid_entries = []

        for query, entry in cache.items():
            if not self.validate_cache_entry(entry):
                invalid_entries.append(query)

        # Remove invalid entries
        for query in invalid_entries:
            del cache[query]

        if invalid_entries:
            self.save_cache(cache)
            logger.info(f"Removed {len(invalid_entries)} invalid cache entries")

        return total_entries, len(invalid_entries)

    def migrate_cache_format(self) -> int:
        """
        Migrate the cache format to the current structure.
        This is useful for when the cache format changes.

        Returns:
            Number of entries migrated
        """
        cache = self.load_cache()
        migrated_count = 0

        for query, entry in cache.items():
            # Check if entry has schema_file but not schema_content
            schema_file = entry.get("schema_file")
            if schema_file and "schema_content" not in entry:
                # This is a case where we need to migrate
                # In a real implementation, you'd load the schema content from the file
                # But for simplicity, we'll just mark it as migrated
                entry.pop("schema_file", None)
                migrated_count += 1

        if migrated_count > 0:
            self.save_cache(cache)
            logger.info(f"Migrated {migrated_count} cache entries to new format")

        return migrated_count
