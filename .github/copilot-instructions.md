"""
filename.py - Brief description of what this file does

SDK Version: google-genai==0.8.3  # Or relevant SDK
Last Updated: 2024-12-04

Purpose:
  - What problem does this solve?
  - What concepts does it demonstrate?

Usage:
  python filename.py [args]
  
Prerequisites:
  pip install google-genai tiktoken python-dotenv
  export GEMINI_API_KEY=your_key_here

Notes:
  - Important gotchas or limitations
  - Links to relevant documentation
"""

# ✅ Good - Clear SDK version and purpose
"""
gemini_25_flash_vs_pro_examples_WORKING.py

✅ COMPLETELY WORKING VERSION - All issues resolved!

Prereqs:
  pip install -U google-genai pydantic python-dotenv

Auth:
  export GOOGLE_API_KEY="YOUR_KEY"   # or GEMINI_API_KEY
"""

# ✅ Good - Explains WHY config is structured this way
def generate_with_thinking(client: genai.Client, model: str) -> None:
    # FIXED: Thinking config must be inside GenerateContentConfig
    # Don't pass as separate parameter or it will fail
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=5000)
        )
    )
