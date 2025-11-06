# 🧠 TOON Format Encoder & LLM Evaluation Suite

This repository provides:
- A **TOON encoder** that converts JSON into a compact, readable format.
- **LLM validation tests** to verify structural accuracy of encoded data.
- Tools for generating, testing, and comparing **TOON vs JSON** reasoning.

## 🧩 Modules
- `src/toon_encoder.py` — Core encoder logic.
- `src/llm_toon_generator.py` — Generates TOON data using OpenAI models.
- `tests/test_encoder_llm_validation.py` — Runs 25 structural validation tests via GPT.
- `tests/test_llm_reasoning_accuracy.py` — Compares JSON vs TOON reasoning results.
- `tests/test_toon_generation.py` — Measures compression & decoding accuracy.


