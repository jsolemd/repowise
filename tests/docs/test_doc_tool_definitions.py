"""Tests for the documentation tool surface.

Sourced directly from the docs registry: the docs lane now publishes this
catalog on its own. The definitions were verified byte-identical to the
retired code-lane ones at the point of the split, and
``tests/doc_search/test_docs_transport.py`` keeps the published schemas pinned
to a frozen golden master.
"""

from repowise.docs.catalog import get_doc_tool_definitions
from repowise.docs.server.executor import (
    HIDDEN_DOC_TOOL_NAMES,
    get_doc_tool_schema_by_name,
    get_public_doc_tool_definitions,
)


def test_doc_tool_surface_is_explicit_and_snake_case() -> None:
    tools = get_doc_tool_definitions()
    names = [str(tool.name) for tool in tools]

    assert names == [
        "list_doc_files",
        "resolve_library_id",
        "search_docs",
        "search_docs_multi",
        "expand_doc_chunk",
        "list_doc_libraries",
        "read_doc",
        "update_doc_library",
        "add_doc_library",
        "delete_doc_library",
        "export_doc_bundle",
        "import_doc_bundle",
    ]


def test_doc_tool_schemas_are_closed_and_expose_output() -> None:
    expected_output_enums = {
        "list_doc_files": ["json"],
        "resolve_library_id": ["json"],
        "search_docs": ["markdown", "json"],
        "search_docs_multi": ["markdown", "json"],
        "expand_doc_chunk": ["markdown", "json"],
        "list_doc_libraries": ["json"],
        "read_doc": ["markdown", "json"],
        "update_doc_library": ["json"],
        "add_doc_library": ["json"],
        "delete_doc_library": ["json"],
        "export_doc_bundle": ["json"],
        "import_doc_bundle": ["json"],
    }
    for tool in get_doc_tool_definitions():
        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        output = properties.get("output")

        assert schema.get("additionalProperties") is False, (
            f"{tool.name} should reject unknown args"
        )
        assert output is not None, f"{tool.name} missing output selector"
        assert output.get("enum") == expected_output_enums[str(tool.name)]


def test_doc_tool_specific_defaults_match_contract() -> None:
    tools = {str(tool.name): tool for tool in get_doc_tool_definitions()}

    search_docs_props = (tools["search_docs"].inputSchema or {}).get("properties", {})
    assert search_docs_props["exact_match"]["default"] is False
    assert search_docs_props["limit"]["default"] == 6
    assert search_docs_props["output"]["default"] == "json"
    # search_docs folds in search_docs_multi via library_ids; library_id is no
    # longer required (either selector is accepted).
    assert search_docs_props["library_ids"]["type"] == "array"
    assert (tools["search_docs"].inputSchema or {}).get("required") == ["query"]

    read_doc_props = (tools["read_doc"].inputSchema or {}).get("properties", {})
    assert read_doc_props["max_tokens"]["default"] == 10000
    assert read_doc_props["output"]["default"] == "json"
    assert read_doc_props["file_path"]["deprecated"] is True

    resolve_props = (tools["resolve_library_id"].inputSchema or {}).get("properties", {})
    assert resolve_props["query"]["deprecated"] is True

    add_library_props = (tools["add_doc_library"].inputSchema or {}).get("properties", {})
    assert add_library_props["branch"]["default"] == "main"
    assert add_library_props["output"]["enum"] == ["json"]
    multi_props = (tools["search_docs_multi"].inputSchema or {}).get("properties", {})
    assert multi_props["limit_per_library"]["default"] == 3


def test_the_docs_catalog_hides_only_the_deprecated_alias() -> None:
    # search_docs_multi is a deprecated alias hidden from the advertised list;
    # the full catalog still carries it, and it stays callable by schema.
    all_names = {str(tool.name) for tool in get_doc_tool_definitions()}
    advertised = [str(tool.name) for tool in get_public_doc_tool_definitions()]
    schema_by_name = get_doc_tool_schema_by_name()

    assert all_names == {
        "list_doc_files",
        "resolve_library_id",
        "search_docs",
        "search_docs_multi",
        "expand_doc_chunk",
        "list_doc_libraries",
        "read_doc",
        "update_doc_library",
        "add_doc_library",
        "delete_doc_library",
        "export_doc_bundle",
        "import_doc_bundle",
    }
    assert frozenset({"search_docs_multi"}) == HIDDEN_DOC_TOOL_NAMES
    assert set(advertised) == all_names - HIDDEN_DOC_TOOL_NAMES
    assert set(schema_by_name) == all_names
    assert schema_by_name["search_docs"]["properties"]["library_id"]["type"] == "string"
