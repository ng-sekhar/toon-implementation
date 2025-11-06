import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.toon_encoder import encode_toon
from src.llm_toon_generator import generate_in_toon
import tiktoken

# ---------- Load Environment ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")
client = OpenAI(api_key=api_key)

# ---------- Import Data ----------
data = {
    "employees": [
        {"id": 1, "name": "Alice", "role": "Engineer", "exp": 5, "salary": 72000, "projects": ["Aurora", "Nebula"], "department": "R&D", "performance": 91},
        {"id": 2, "name": "Bob", "role": "Manager", "exp": 8, "salary": 90000, "projects": ["Aurora", "Horizon"], "department": "Operations", "performance": 87},
        {"id": 3, "name": "Cara", "role": "Analyst", "exp": 3, "salary": 60000, "projects": ["Nova"], "department": "Finance", "performance": 84},
        {"id": 4, "name": "Dan", "role": "Designer", "exp": 6, "salary": 65000, "projects": ["Horizon", "Aurora"], "department": "Design", "performance": 88},
        {"id": 5, "name": "Eve", "role": "Engineer", "exp": 4, "salary": 70000, "projects": ["Aurora", "Nova"], "department": "R&D", "performance": 90},
        {"id": 6, "name": "Frank", "role": "HR Specialist", "exp": 7, "salary": 58000, "projects": [], "department": "HR", "performance": 86},
        {"id": 7, "name": "Grace", "role": "Data Scientist", "exp": 5, "salary": 95000, "projects": ["Nova", "Orion"], "department": "AI Research", "performance": 93},
        {"id": 8, "name": "Henry", "role": "DevOps Engineer", "exp": 9, "salary": 98000, "projects": ["Orion", "Nebula"], "department": "Infrastructure", "performance": 89},
        {"id": 9, "name": "Ivy", "role": "QA Engineer", "exp": 4, "salary": 67000, "projects": ["Horizon"], "department": "Quality Assurance", "performance": 85},
        {"id": 10, "name": "Jack", "role": "Product Manager", "exp": 10, "salary": 105000, "projects": ["Aurora", "Orion"], "department": "Product", "performance": 92}
    ],

    "projects": [
        {"title": "Aurora", "status": "active", "budget": 1.25, "teamSize": 5, "priority": "high", "deadline": "2025-03-15"},
        {"title": "Horizon", "status": "inactive", "budget": 0.8, "teamSize": 3, "priority": "medium", "deadline": "2024-12-10"},
        {"title": "Nova", "status": "active", "budget": 0.95, "teamSize": 4, "priority": "low", "deadline": "2025-06-20"},
        {"title": "Nebula", "status": "planning", "budget": 0.6, "teamSize": 2, "priority": "medium", "deadline": "2025-10-01"},
        {"title": "Orion", "status": "active", "budget": 1.8, "teamSize": 6, "priority": "critical", "deadline": "2025-02-01"}
    ],

    "departments": [
        {"name": "R&D", "head": "Dr. Collins", "budget": 2.0, "location": "Building A"},
        {"name": "Operations", "head": "Mr. Patel", "budget": 1.5, "location": "Building B"},
        {"name": "Finance", "head": "Ms. Nguyen", "budget": 1.2, "location": "Building C"},
        {"name": "Design", "head": "Ms. Tanaka", "budget": 1.1, "location": "Building D"},
        {"name": "HR", "head": "Mr. Lee", "budget": 0.9, "location": "Building E"},
        {"name": "AI Research", "head": "Dr. Li", "budget": 2.5, "location": "Building F"},
        {"name": "Infrastructure", "head": "Mr. Zhao", "budget": 1.8, "location": "Building G"},
        {"name": "Quality Assurance", "head": "Ms. Rossi", "budget": 1.0, "location": "Building H"},
        {"name": "Product", "head": "Mr. Singh", "budget": 1.7, "location": "Building I"}
    ],

    "company": {
        "name": "TechNova Global",
        "location": "New York",
        "founded": 2010,
        "revenue": 45.8,
        "public": True,
        "ceo": "Dr. Angela Carter",
        "offices": {
            "US": {"employees": 320, "revenue": 25.4},
            "EU": {"employees": 180, "revenue": 15.2},
            "Asia": {"employees": 100, "revenue": 5.2}
        }
    },

    "metrics": {
        "quarterly_performance": [88, 91, 85, 93],
        "customer_satisfaction": 89.5,
        "employee_retention_rate": 94.2,
        "avg_project_completion_days": 112.4,
        "innovation_index": 9.1
    }
}


questions = [
    # Employee analysis
    "Who is the highest-paid employee and what department do they belong to?",
    "Which employee has the best performance score in the R&D department?",
    "List all employees working on more than one project.",
    "Find the average salary per department and identify the highest-paying one.",
    "Who are the top 3 employees by performance score?",
    "Which employees contribute to active projects only?",
    "What is the average experience of employees working on 'Orion'?",
    "Which department has the lowest total salary expense?",

    # Project insights
    "List all projects sorted by priority and budget descending.",
    "Which active project has the closest upcoming deadline?",
    "Find all employees who are part of critical or high-priority projects.",
    "Calculate the total combined budget of all active projects.",
    "Which project has the largest team size and who are its members?",

    # Department reasoning
    "Which department head manages the largest budget?",
    "Identify departments located in Building A or Building F.",
    "Which department has employees with average performance above 90?",

    # Company-level reasoning
    "What is the total number of employees across all regions?",
    "Which office generates the highest revenue per employee?",
    "Is the company’s innovation index above 8.5, and what does it imply?",
    "If the company grows revenue by 10%, what will the new revenue be?",

    # Cross-domain reasoning
    "Which projects are managed by employees from different departments?",
    "Who among the engineers has the highest performance score?",
    "What percentage of employees work in departments with budgets over 1.5 million?",
    "Among all employees, who has the most projects and how many?",
    "Based on current salaries and experience, estimate which employees are likely to be promoted next year.",
    "Summarize the overall company health using revenue, satisfaction, and innovation index."
]

# ---------- Encode in JSON ----------
json_input = json.dumps(data, indent=2)

# ---------- Encode in TOON (deterministic) ----------
toon_input_encoder = encode_toon(data)

# ---------- Encode in TOON (LLM-generated) ----------
print("⏳ Generating TOON via LLM...")
toon_input_llm = generate_in_toon("gpt-4o-mini", "Convert this JSON data into TOON format.", json_input)
print("✅ LLM TOON Generated Successfully!\n")

# ---------- Token Stats ----------
enc = tiktoken.get_encoding("o200k_base")
def token_count(txt): return len(enc.encode(txt))
print("🔹 Token Usage Comparison")
print(f"JSON: {token_count(json_input)} tokens")
print(f"TOON Encoder: {token_count(toon_input_encoder)} tokens")
print(f"TOON LLM: {token_count(toon_input_llm)} tokens\n")

# ---------- Query Function ----------
def ask_llm(model, format_type, data_str, question):
    system_prompt = f"You are an expert analyst. Parse the {format_type} data carefully and answer accurately."
    user_prompt = f"Data:\n{data_str}\n\nQuestion: {question}\nAnswer:"
    start = time.time()
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    latency = time.time() - start
    return res.choices[0].message.content.strip(), latency

# ---------- Run Tests ----------
model = "gpt-4o-mini"
results = []

for q in questions:
    print(f"\n🔍 Question: {q}")

    json_ans, json_lat = ask_llm(model, "JSON", json_input, q)
    toon_enc_ans, toon_enc_lat = ask_llm(model, "TOON (encoded)", toon_input_encoder, q)
    toon_llm_ans, toon_llm_lat = ask_llm(model, "TOON (LLM-generated)", toon_input_llm, q)

    match_enc = json_ans.lower().strip() == toon_enc_ans.lower().strip()
    match_llm = json_ans.lower().strip() == toon_llm_ans.lower().strip()

    results.append({
        "question": q,
        "json_answer": json_ans,
        "toon_encoder_answer": toon_enc_ans,
        "toon_llm_answer": toon_llm_ans,
        "json_latency": round(json_lat, 3),
        "toon_encoder_latency": round(toon_enc_lat, 3),
        "toon_llm_latency": round(toon_llm_lat, 3),
        "encoder_match": match_enc,
        "llm_match": match_llm
    })

    print(f"  JSON ({json_lat:.2f}s) → {json_ans[:80]}...")
    print(f"  ENCODER ({toon_enc_lat:.2f}s) → {toon_enc_ans[:80]}... [Match: {match_enc}]")
    print(f"  LLM TOON ({toon_llm_lat:.2f}s) → {toon_llm_ans[:80]}... [Match: {match_llm}]")

    time.sleep(1)

# ---------- Save Results ----------
os.makedirs("results", exist_ok=True)
with open("results/all_toon_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n🗂 Results saved to results/all_toon_comparison.json")

# ---------- Summary ----------
avg_json = sum(r["json_latency"] for r in results) / len(results)
avg_enc = sum(r["toon_encoder_latency"] for r in results) / len(results)
avg_llm = sum(r["toon_llm_latency"] for r in results) / len(results)

print("\n📊 Average Latency Summary")
print("-" * 50)
print(f"JSON Avg Latency        : {avg_json:.2f}s")
print(f"TOON Encoder Avg Latency: {avg_enc:.2f}s")
print(f"TOON LLM Avg Latency    : {avg_llm:.2f}s")
print(f"Encoder Match Rate      : {sum(r['encoder_match'] for r in results)/len(results)*100:.1f}%")
print(f"LLM TOON Match Rate     : {sum(r['llm_match'] for r in results)/len(results)*100:.1f}%")
