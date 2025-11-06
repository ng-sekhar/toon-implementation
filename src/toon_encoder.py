import json
from typing import Any

def encode_toon(data: Any, indent: int = 0) -> str:
    """TOON encoder with tabular array, empty container, and nested dict support."""
    spaces = "  " * indent

    # Handle empty root dict
    if isinstance(data, dict) and not data:
        return f"{spaces}{{}}"

    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            # --- Handle empty dicts explicitly ---
            if isinstance(v, dict):
                if not v:
                    lines.append(f"{spaces}{k}: {{}}")
                else:
                    lines.append(f"{spaces}{k}:")
                    lines.append(encode_toon(v, indent + 1))

            # --- Handle lists ---
            elif isinstance(v, list):
                # Empty list
                if len(v) == 0:
                    lines.append(f"{spaces}{k}: []")

                # Tabular array: uniform dicts with same keys
                elif all(isinstance(x, dict) and x.keys() == v[0].keys() for x in v):
                    headers = ",".join(v[0].keys())
                    lines.append(f"{spaces}{k}[{len(v)}]{{{headers}}}:")
                    for row in v:
                        row_values = ",".join(json.dumps(vv, ensure_ascii=False) for vv in row.values())
                        lines.append(f"{spaces}  {row_values}")

                # Inline array of primitives
                elif all(not isinstance(x, (dict, list)) for x in v):
                    joined = ",".join(json.dumps(x, ensure_ascii=False) for x in v)
                    lines.append(f"{spaces}{k}[{len(v)}]: {joined}")

                # List of nested/mixed objects
                else:
                    lines.append(f"{spaces}{k}[{len(v)}]:")
                    for item in v:
                        lines.append(encode_toon(item, indent + 1))

            # --- Scalars: handle bools and None explicitly ---
            else:
                if isinstance(v, bool):
                    val = "true" if v else "false"
                elif v is None:
                    val = "None"
                elif isinstance(v, str):
                    val = json.dumps(v, ensure_ascii=False)
                else:
                    val = v
                lines.append(f"{spaces}{k}: {val}")

        return "\n".join(lines)

    elif isinstance(data, list):
        # fallback for direct list inputs
        return f"{spaces}[{len(data)}]: {','.join(map(str, data))}"

    else:
        return str(data)
