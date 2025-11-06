import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Load environment ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


# ---------- Helper: Ask LLM for TOON output ----------
def generate_in_toon(model: str, instruction: str, data: str = None) -> str:
    """
    Generic function to make LLM generate structured output in TOON format.
    :param model: Model name (e.g., 'gpt-4o-mini')
    :param instruction: The user instruction or question.
    :param data: Optional context or data (JSON, text, etc.).
    :return: LLM-generated TOON-formatted string.
    """

    base_prompt = """
You are a structured data generator.
Your task is to analyze user input and produce the result STRICTLY in TOON format.

⚙️ TOON FORMAT RULES:
- Use indentation for nested objects.
- Lists use [N]: item1,item2,...
- Tabular data uses [N]{columns}: followed by rows.
- Each key/value pair is on a new line.
- No markdown, quotes, or explanations.
- Respond ONLY in valid TOON format. Do not use JSON or natural language.

📘 Example:
students[2]{id,name,grade}:
  1,Alice,A+
  2,Bob,B
summary:
  total_students: 2
  top_grade: A+
"""

    if data:
        user_prompt = f"{instruction}\n\nHere is the input data:\n{data}\n\nOutput ONLY in TOON format:"
    else:
        user_prompt = f"{instruction}\n\nOutput ONLY in TOON format:"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": base_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ---------- Example: Run Interactively ----------
if __name__ == "__main__":
    print("🧠 TOON Generator (Generic) — enter any task below")
    print("---------------------------------------------------")

    model = input("Enter model (default: gpt-4o-mini): ").strip() or "gpt-4o-mini"
    instruction = input("Enter your instruction/task: ").strip()

    print("\n(Optional) Enter input data (press Enter twice to finish):")
    data_lines = []
    while True:
        line = input()
        if line == "":
            break
        data_lines.append(line)
    data = "\n".join(data_lines).strip() or None

    print("\n⏳ Generating TOON output...\n")
    toon_output = generate_in_toon(model, instruction, data)

    print("✅ TOON Output:\n")
    print(toon_output)
