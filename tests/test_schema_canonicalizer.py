import json

from cachealign.normalizers.schema import canonicalize_object, canonicalize_tool_schemas


def test_canonicalize_object_dictionary_sorting():
    raw_dict = {
        "zeta": 1,
        "alpha": 2,
        "nested": {
            "gamma": True,
            "beta": False,
        },
    }
    canonical = canonicalize_object(raw_dict)
    keys = list(canonical.keys())
    assert keys == ["alpha", "nested", "zeta"]
    assert list(canonical["nested"].keys()) == ["beta", "gamma"]


def test_canonicalize_tool_schemas_sorting_anthropic_format():
    tools = [
        {
            "name": "search_web",
            "description": "Searches the web",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
        {
            "name": "calculate_math",
            "description": "Calculates math expression",
            "input_schema": {"type": "object", "properties": {"expr": {"type": "string"}}},
        },
    ]
    canonical = canonicalize_tool_schemas(tools)
    assert canonical is not None
    assert canonical[0]["name"] == "calculate_math"
    assert canonical[1]["name"] == "search_web"
    assert list(canonical[1]["input_schema"]["properties"].keys()) == ["limit", "query"]


def test_canonicalize_tool_schemas_byte_identical():
    # Two tools with identical semantics but different key serialization order
    t1 = [
        {"name": "tool_a", "description": "desc", "parameters": {"y": 1, "x": 2}},
        {"name": "tool_b", "description": "desc", "parameters": {"b": 1, "a": 2}},
    ]
    t2 = [
        {"description": "desc", "name": "tool_b", "parameters": {"a": 2, "b": 1}},
        {"parameters": {"x": 2, "y": 1}, "description": "desc", "name": "tool_a"},
    ]

    res1 = canonicalize_tool_schemas(t1)
    res2 = canonicalize_tool_schemas(t2)

    assert json.dumps(res1) == json.dumps(res2)
