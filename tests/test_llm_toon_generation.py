import os
import json
from datetime import datetime
from src.llm_toon_generator import generate_in_toon

# ---------- Test Data ----------
TEST_CASES = [
    {
        "name": "Student Performance",
        "instruction": "Summarize this dataset with top performer and average score.",
        "data": json.dumps({
            "students": [
                {"id": 1, "name": "Alice", "score": 95},
                {"id": 2, "name": "Bob", "score": 88},
                {"id": 3, "name": "Cara", "score": 92}
            ]
        }, indent=2)
    },
    {
        "name": "Department Summary",
        "instruction": "List each department, its head, and total employees in TOON format.",
        "data": json.dumps({
            "departments": [
                {"name": "R&D", "head": "Dr. Collins", "employees": 12},
                {"name": "Sales", "head": "Ms. Patel", "employees": 20},
                {"name": "HR", "head": "Mr. Lee", "employees": 6}
            ]
        }, indent=2)
    },
    {
        "name": "Company Metrics",
        "instruction": "Summarize company performance and include key metrics in TOON format.",
        "data": json.dumps({
            "company": {"name": "TechNova", "year": 2024, "departments": 5},
            "metrics": {"revenue_million": 25.4, "growth_percent": 12.8, "customer_satisfaction": 89.5}
        }, indent=2)
    },
    {
        "name": "Product Sales",
        "instruction": "Analyze this sales data and summarize top-selling product and total revenue.",
        "data": json.dumps({
            "sales": [
                {"product": "Phone", "units": 320, "price": 600},
                {"product": "Laptop", "units": 150, "price": 1200},
                {"product": "Tablet", "units": 180, "price": 400}
            ]
        }, indent=2)
    },
    {
        "name": "Weather Averages",
        "instruction": "Summarize average temperature and rainfall across all cities in TOON format.",
        "data": json.dumps({
            "weather": [
                {"city": "London", "temp": 16, "rainfall": 85},
                {"city": "Paris", "temp": 18, "rainfall": 70},
                {"city": "Rome", "temp": 22, "rainfall": 60}
            ]
        }, indent=2)
    }
]


# ---------- Setup ----------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL = "gpt-4o-mini"

all_results = []

print("\n🚀 Starting TOON format evaluation...\n")

# ---------- Run Each Test ----------
for test in TEST_CASES:
    print(f"🧩 Running test: {test['name']}")
    toon_output = generate_in_toon(MODEL, test["instruction"], test["data"])

    # Measure token (character) efficiency
    toon_tokens = len(toon_output)
    json_tokens = len(test["data"])
    compression_ratio = round(toon_tokens / json_tokens, 3)

    print(f"✅ TOON generated ({toon_tokens} chars vs JSON {json_tokens})\n")

    all_results.append({
        "test_name": test["name"],
        "instruction": test["instruction"],
        "toon_output": toon_output,
        "toon_tokens": toon_tokens,
        "json_tokens": json_tokens,
        "compression_ratio": compression_ratio
    })


# ---------- Summary Table ----------
print("\n📊 Summary Report\n" + "-"*50)
print(f"{'Test Case':30} | {'TOON':>6} | {'JSON':>6} | {'Ratio':>6}")
print("-"*50)
for r in all_results:
    print(f"{r['test_name'][:28]:30} | {r['toon_tokens']:>6} | {r['json_tokens']:>6} | {r['compression_ratio']:>6}")
print("-"*50)

# ---------- Save All Results to One File ----------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_path = os.path.join(RESULTS_DIR, f"toon_generation_results_{timestamp}.json")

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ All results saved in one file: '{summary_path}'\n")
