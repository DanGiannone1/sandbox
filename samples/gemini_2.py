
"""
gemini_25_flash_vs_pro_examples_WORKING.py

✅ COMPLETELY WORKING VERSION - All issues resolved!

This version fixes ALL identified issues:
1. ✅ Thinking config parameter placement 
2. ✅ Function calling response structure (fc.args vs fc.function_call.args)
3. ✅ Code execution tool instantiation (added parentheses)
4. ✅ Automatic function calling thinking interference 
5. ✅ Manual function calling conversation turn order (CRITICAL FIX)

The manual function calling now properly maintains conversation history to avoid:
"Please ensure that function call turn comes immediately after a user turn or after a function response turn"

Prereqs:
  pip install -U google-genai pydantic python-dotenv

Auth:
  export GOOGLE_API_KEY="YOUR_KEY"   # or GEMINI_API_KEY
"""

from __future__ import annotations

from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Utilities
# -----------------------------

def make_client() -> genai.Client:
    """Create a Client. Reads GOOGLE_API_KEY / GEMINI_API_KEY env vars."""
    return genai.Client()

# -----------------------------
# 1) Minimal text generation
# -----------------------------

def minimal_generation_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Minimal generation with {model} ===")
    prompt = "In one sentence, explain the difference between a comet and an asteroid."
    resp = client.models.generate_content(model=model, contents=prompt)
    print(resp.text)

# -----------------------------
# 2) Streaming
# -----------------------------

def streaming_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Streaming with {model} ===")
    for chunk in client.models.generate_content_stream(
        model=model,
        contents="Write a short limerick about databases.",
    ):
        print(chunk.text, end="")
    print()  # newline

# -----------------------------
# 3) Thinking configuration - FIXED
# -----------------------------

def thinking_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Thinking config with {model} ===")

    if "flash" in model:
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,        # disable thinking for speed
                include_thoughts=True     # FIXED: belongs in ThinkingConfig
            )
        )
        prompt = "Plan a 3-step to-do list to prepare tea."
    else:
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=512,      # set thinking budget
                include_thoughts=True     # FIXED: belongs in ThinkingConfig
            )
        )
        prompt = "Plan a 3-step to-do list to prepare coffee."

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg,
    )

    print("Main response:")
    print(resp.text)

    # Check for thought summaries
    thought_found = False
    for part in resp.candidates[0].content.parts:
        if part and hasattr(part, 'thought') and part.thought:
            print("\nThought summary found:")
            print(part.text)
            thought_found = True

    if not thought_found:
        print("\n(No thought summaries - expected with budget=0 for Flash)")

# -----------------------------
# 4) Structured output (JSON) using Pydantic
# -----------------------------

class Book(BaseModel):
    title: str
    author: str
    year: int

def structured_output_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Structured JSON with {model} ===")
    resp = client.models.generate_content(
        model=model,
        contents="Return exactly two science fiction books as JSON.",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[Book],
        ),
    )
    print(resp.text)

# -----------------------------
# 5) Function calling - ALL FIXED
# -----------------------------

# A) Automatic function calling - FIXED
def get_weather_forecast(location: str) -> dict:
    """Mock weather tool that gets weather forecast for a location."""
    print(f"[tool] get_weather_forecast(location={location!r})")
    return {"temperature": 25, "unit": "celsius", "condition": "sunny"}

def set_thermostat_temperature(temperature: int) -> dict:
    """Mock thermostat tool that sets temperature."""
    print(f"[tool] set_thermostat_temperature(temperature={temperature})")
    return {"status": "success", "message": f"Thermostat set to {temperature}°C"}

def automatic_function_calling_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Automatic function calling with {model} ===")

    # FIXED: No thinking config to avoid interference
    cfg = types.GenerateContentConfig(
        tools=[get_weather_forecast, set_thermostat_temperature]
    )

    resp = client.models.generate_content(
        model=model,
        contents=(
            "Check the weather in London. If it's warmer than 20°C, set the thermostat to 20°C, "
            "otherwise set it to 18°C."
        ),
        config=cfg,
    )

    # Handle response properly
    response_text = ""
    for part in resp.candidates[0].content.parts:
        if part.text and not (hasattr(part, 'thought') and part.thought):
            response_text += part.text

    print("Response:", response_text if response_text else resp.text)

# B) Manual function calling - COMPLETELY FIXED WITH CONVERSATION HISTORY
def manual_function_calling_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Manual function calling loop with {model} ===")
    function = types.FunctionDeclaration(
        name="get_current_temperature",
        description="Get the current temperature for a given location.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "location": types.Schema(type="STRING", description="City name, e.g. London")
            },
            required=["location"],
        ),
    )
    tool = types.Tool(function_declarations=[function])

    # Initial USER turn
    initial_prompt = "What's the temperature in London?"

    resp = client.models.generate_content(
        model=model,
        contents=initial_prompt,
        config=types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )

    print("Checking for function calls...")

    # FIXED: Check resp.function_calls and use fc.args directly
    if hasattr(resp, 'function_calls') and resp.function_calls:
        fc = resp.function_calls[0]
        print(f"✅ Model requested: {fc.name}")
        print(f"   Arguments: {dict(fc.args)}")

        # Simulate function execution
        result = {"temperature": 25, "unit": "celsius"}
        print(f"   Simulated result: {result}")

        # CRITICAL FIX: Proper conversation history structure
        # Must follow: USER → MODEL (function call) → USER (function response) → MODEL (final)
        conversation_history = [
            types.Content(role="user", parts=[types.Part(text=initial_prompt)]),
            types.Content(role="model", parts=[
                types.Part.from_function_call(name=fc.name, args=fc.args)
            ]),
            types.Content(role="user", parts=[
                types.Part.from_function_response(name=fc.name, response=result)
            ])
        ]

        # Send complete conversation history for final response
        followup = client.models.generate_content(
            model=model,
            contents=conversation_history,  # Complete history, not individual parts
            config=types.GenerateContentConfig(tools=[tool]),
        )

        print("✅ Final response:")
        print(followup.text)

    else:
        # Fallback: check response parts
        function_call_found = False
        for part in resp.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                print(f"Found function call: {part.function_call.name}")
                print(f"Arguments: {dict(part.function_call.args)}")

                result = {"temperature": 25, "unit": "celsius"}

                # Same conversation history fix
                conversation_history = [
                    types.Content(role="user", parts=[types.Part(text=initial_prompt)]),
                    types.Content(role="model", parts=[part]),
                    types.Content(role="user", parts=[
                        types.Part.from_function_response(name=part.function_call.name, response=result)
                    ])
                ]

                followup = client.models.generate_content(
                    model=model,
                    contents=conversation_history,
                    config=types.GenerateContentConfig(tools=[tool]),
                )
                print("Final response:", followup.text)
                function_call_found = True
                break

        if not function_call_found:
            print("❌ No function calls found")
            print("Response text:", resp.text)

# -----------------------------
# 6) Code execution tool - FIXED
# -----------------------------

def code_execution_example(client: genai.Client, model: str) -> None:
    print(f"\n=== Code execution with {model} ===")
    resp = client.models.generate_content(
        model=model,
        contents=(
            "What is the sum of the first 50 prime numbers? "
            "Generate and run Python to compute it."
        ),
        config=types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]  # FIXED: added ()
        ),
    )

    # Process all response parts
    parts = resp.candidates[0].content.parts
    for i, p in enumerate(parts):
        if getattr(p, "text", None):
            print(f"[model text {i+1}]", p.text[:200] + "..." if len(p.text) > 200 else p.text)
        if getattr(p, "executable_code", None):
            print(f"[code {i+1}]", p.executable_code.code)
        if getattr(p, "code_execution_result", None):
            print(f"[result {i+1}]", p.code_execution_result.output)

# -----------------------------
# Main execution with comprehensive error handling
# -----------------------------

def main():
    try:
        client = make_client()
        print("✅ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        print("💡 Make sure GOOGLE_API_KEY or GEMINI_API_KEY is set")
        return

    MODEL = "gemini-2.5-flash"

    print(f"🚀 Running all examples with {MODEL}")
    print("=" * 60)

    examples = [
        ("Minimal Generation", minimal_generation_example),
        ("Streaming", streaming_example), 
        ("Thinking Configuration", thinking_example),
        ("Structured JSON Output", structured_output_example),
        ("Automatic Function Calling", automatic_function_calling_example),
        ("Manual Function Calling", manual_function_calling_example),
        ("Code Execution", code_execution_example),
    ]

    success_count = 0
    for name, func in examples:
        try:
            func(client, MODEL)
            success_count += 1
            print(f"✅ {name} - SUCCESS")
        except Exception as e:
            print(f"❌ {name} - FAILED: {e}")
            print(f"   Error type: {type(e).__name__}")
            if "function call turn" in str(e):
                print("   💡 This is a conversation turn order issue - check the manual function calling fix")
            elif "include_thoughts" in str(e):
                print("   💡 This is a thinking config placement issue - check ThinkingConfig")
            elif "ToolCodeExecution" in str(e):
                print("   💡 This is a code execution instantiation issue - add parentheses")

    print("\n" + "=" * 60)
    print(f"🎉 Results: {success_count}/{len(examples)} examples succeeded")

    if success_count == len(examples):
        print("🎉 ALL EXAMPLES WORKING! Your Gemini SDK setup is perfect.")
    else:
        print("⚠️  Some examples failed. Check the error messages above for specific fixes.")

if __name__ == "__main__":
    main()
