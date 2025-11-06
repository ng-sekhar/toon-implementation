import os
import json
from openai import OpenAI
from src.toon_encoder import encode_toon
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- TEST CASES ----------
test_cases = {
    # 1️⃣ Simple flat dictionary
    "simple_flat_dict": {
        "id": 1,
        "name": "John",
        "age": 30
    },

    # 2️⃣ Basic nested dictionary
    "basic_nested_dict": {
        "user": {"id": 1, "name": "Alice", "active": True},
        "meta": {"country": "USA", "signup_date": "2024-03-14"}
    },

    # 3️⃣ Boolean variations
    "boolean_values": {
        "flag_true": True,
        "flag_false": False,
        "nested": {"ok": True, "fail": False}
    },

    # 4️⃣ Null values
    "null_values": {
        "none_field": None,
        "empty_string": "",
        "nested_null": {"value": None}
    },

    # 5️⃣ Inline primitives (array)
    "inline_primitives": {
        "colors": ["red", "green", "blue"]
    },

    # 6️⃣ Empty containers
    "empty_values": {
        "none_field": None,
        "empty_list": [],
        "empty_dict": {},
        "zero_value": 0
    },

    # 7️⃣ Tabular data — consistent dicts
    "tabular_array": {
        "employees": [
            {"id": 1, "name": "Tom", "role": "Engineer"},
            {"id": 2, "name": "Jane", "role": "Designer"}
        ]
    },

    # 8️⃣ Mixed nested lists
    "mixed_nested_lists": {
        "teams": [
            {"name": "Backend", "members": ["Tom", "Alice", "Rob"]},
            {"name": "Frontend", "members": ["Jane", "Eve"]}
        ]
    },

    # 9️⃣ Deep nesting (hierarchical)
    "deeply_nested": {
        "org": {
            "departments": [
                {"name": "AI", "projects": [{"code": "A1", "budget": 50000}]},
                {"name": "ML", "projects": [{"code": "M1", "budget": 75000}]}
            ]
        }
    },

    # 🔟 Unicode & special characters
    "unicode_and_specials": {
        "city": "München",
        "emoji": "🚀",
        "quote_test": 'She said, "Hello, world!"',
        "newline": "Line1\nLine2\nLine3"
    },

    # 11️⃣ Numbers with precision
    "numeric_precision": {
        "int": 42,
        "float": 3.14159,
        "large": 12345678901234567890,
        "small_float": 0.00000012345
    },

    # 12️⃣ List of empty dicts
    "list_of_empty_dicts": {
        "items": [{}, {}, {}]
    },

    # 13️⃣ Dict of lists
    "dict_of_lists": {
        "grades": [98, 85, 92],
        "tags": ["A", "B", "C"]
    },

    # 14️⃣ List of dicts with nested dicts
    "list_of_nested_dicts": {
        "students": [
            {"id": 1, "info": {"name": "John", "age": 21}},
            {"id": 2, "info": {"name": "Sara", "age": 22}}
        ]
    },

    # 15️⃣ Mixed types in same list
    "mixed_type_list": {
        "data": [1, "two", True, None, 3.5]
    },

    # 16️⃣ Deeply recursive nesting (stress)
    "recursive_nesting": {
        "level1": {"level2": {"level3": {"level4": {"key": "value"}}}}
    },

    # 17️⃣ Empty root dict
    "empty_root": {}
    ,

    # 18️⃣ Array of arrays
    "array_of_arrays": {
        "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    },

    # 19️⃣ Long text fields
    "long_text_fields": {
        "paragraph": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3
    },

    # 20️⃣ Special characters and escapes
    "special_escapes": {
        "tab": "Column1\tColumn2",
        "quote": "\"Escaped\"",
        "backslash": "C:\\Users\\Alice"
    },

    # 21️⃣ Heterogeneous dict keys
    "heterogeneous_keys": {
        "1": "numeric_key",
        "True": "boolean_key",
        "None": "none_key"
    },

    # 22️⃣ Nested tabular arrays
    "nested_tabular": {
        "departments": [
            {
                "name": "Engineering",
                "employees": [
                    {"id": 1, "name": "Tom"},
                    {"id": 2, "name": "Jane"}
                ]
            },
            {
                "name": "Design",
                "employees": [
                    {"id": 3, "name": "Eve"},
                    {"id": 4, "name": "Rob"}
                ]
            }
        ]
    },

    # 23️⃣ Boolean edge inside arrays
    "bool_array": {
        "flags": [True, False, True]
    },

    # 24️⃣ None mixed in list
    "none_mixed_list": {
        "values": [1, None, "x", False]
    },

    # 25️⃣ Complex real-world config example
    "complex_config": {
        "service": {
            "name": "api-server",
            "enabled": True,
            "replicas": 3,
            "endpoints": [
                {"path": "/health", "method": "GET", "auth": False},
                {"path": "/data", "method": "POST", "auth": True}
            ],
            "metadata": {
                "labels": {"env": "prod", "region": "us-east-1"},
                "version": "v1.2.3"
            }
        }
    }
}

# ---------- RUN TESTS ----------
for name, data in test_cases.items():
    print(f"\n🧩 TEST: {name}")
    encoded = encode_toon(data)
    print(encoded)

    prompt = f"""
You are a JSON-to-TOON validator.
Compare the JSON object and the TOON encoding.
If the TOON encoding fully and losslessly represents the JSON, reply exactly with:
✅ Correct

If there is any issue, reply concisely with:
❌ [short reason only — 1 line]

JSON:
{json.dumps(data, indent=2)}

TOON:
{encoded}
"""

    response = client.responses.create(
        model="gpt-4o",
        input=prompt
    )

    print(response.output_text.strip())
