import json
import time
import os
from openai import OpenAI
from src.toon_encoder import encode_toon
from dotenv import load_dotenv
import tiktoken

# ---------- Load Environment Variables ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ---------- Data ----------
data = {
    "employees": [
        {"id": 1, "name": "Alice", "role": "Engineer", "exp": 5, "salary": 72000, "projects": ["Aurora", "Nebula"], "department": "R&D"},
        {"id": 2, "name": "Bob", "role": "Manager", "exp": 8, "salary": 90000, "projects": ["Aurora", "Horizon"], "department": "Operations"},
        {"id": 3, "name": "Cara", "role": "Analyst", "exp": 3, "salary": 60000, "projects": ["Nova"], "department": "Finance"},
        {"id": 4, "name": "Dan", "role": "Designer", "exp": 6, "salary": 65000, "projects": ["Horizon"], "department": "Design"},
        {"id": 5, "name": "Eve", "role": "Engineer", "exp": 4, "salary": 70000, "projects": ["Aurora", "Nova"], "department": "R&D"},
        {"id": 6, "name": "Frank", "role": "HR Specialist", "exp": 7, "salary": 58000, "projects": [], "department": "HR"}
    ],

    "projects": [
        {"title": "Aurora", "status": "active", "budget": 1.25, "teamSize": 4, "priority": "high"},
        {"title": "Horizon", "status": "inactive", "budget": 0.8, "teamSize": 3, "priority": "medium"},
        {"title": "Nova", "status": "active", "budget": 0.95, "teamSize": 2, "priority": "low"},
        {"title": "Nebula", "status": "planning", "budget": 0.6, "teamSize": 2, "priority": "medium"}
    ],

    "departments": [
        {"name": "R&D", "head": "Dr. Collins", "budget": 2.0},
        {"name": "Operations", "head": "Mr. Patel", "budget": 1.5},
        {"name": "Finance", "head": "Ms. Nguyen", "budget": 1.2},
        {"name": "Design", "head": "Ms. Tanaka", "budget": 1.1},
        {"name": "HR", "head": "Mr. Lee", "budget": 0.9}
    ],

    "company": {
        "name": "TechNova",
        "location": "New York",
        "founded": 2010,
        "revenue": 25.4,
        "public": True
    },

    "metrics": {
        "quarterly_performance": [88, 91, 85, 93],
        "customer_satisfaction": 89.5,
        "employee_retention_rate": 94.2
    }
}

questions = [
    # Employee-level reasoning
    "Who has the most experience?",
    "List all employee names and their roles.",
    "What is the average employee salary?",
    "Which department has the most employees?",
    "Which employees work on the 'Aurora' project?",
    "Who has more experience, Bob or Alice?",
    "List all employees from the R&D department.",
    "What is the highest salary among Engineers?",
    "How many employees have more than 5 years of experience?",
    
    # Project-level reasoning
    "Which projects are currently active?",
    "Which project has the highest budget?",
    "Which project has the smallest team size?",
    
    # Company-level reasoning
    "What is the company’s name and location?",
    "Is the company publicly listed?",
    "What is the average quarterly performance score?",
    "What is the overall customer satisfaction percentage?"
]

# ---------- Generate Inputs ----------
json_input = json.dumps(data, indent=2)
toon_input = encode_toon(data)

# ---------- Token Counting ----------
enc = tiktoken.get_encoding("o200k_base")
json_tokens = len(enc.encode(json_input))
toon_tokens = len(enc.encode(toon_input))
token_savings = (1 - toon_tokens / json_tokens) * 100

print("🔹 Token Comparison")
print(f"   JSON Tokens: {json_tokens}")
print(f"   TOON Tokens: {toon_tokens}")
print(f"   ➡️  {token_savings:.2f}% token savings\n")

# ---------- Function to Query LLM ----------
def ask_llm(model: str, format_type: str, data_str: str, question: str):
    system_prompt = f"""
You are an expert data analyst. The following data is provided in {format_type} format.
Parse it carefully and answer the question accurately.
"""
    user_prompt = f"Data:\n{data_str}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ---------- Run Tests ----------
model = "gpt-4o-mini"
results = []

for q in questions:
    print(f"🔍 Question: {q}")
    json_ans = ask_llm(model, "JSON", json_input, q)
    toon_ans = ask_llm(model, "TOON", toon_input, q)
    print(f"   JSON → {json_ans}")
    print(f"   TOON → {toon_ans}\n")

    # Compare answers (simple normalized string match)
    match = json_ans.lower().strip() == toon_ans.lower().strip()

    results.append({
        "question": q,
        "json_answer": json_ans,
        "toon_answer": toon_ans,
        "match": match
    })
    time.sleep(1)

# ---------- Save & Summarize ----------
os.makedirs("results", exist_ok=True)
with open("results/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"🗂 Results saved to results/comparison_results.json")
