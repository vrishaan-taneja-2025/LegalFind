"""Rebuild the base, holdings, and citation inverted indexes in order."""

from retrieval.build_inverted_index import build_index as build_source_index
from retrieval.rebuild_two_table_index import rebuild_index as build_search_indexes


def main():
    build_source_index()
    build_search_indexes()


if __name__ == "__main__":
    main()
