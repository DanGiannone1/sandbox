"""
Gemini SDK Quick Reference - Essential examples for the Google GenAI Python SDK

Setup:
  pip install -U google-genai
  export GEMINI_API_KEY=your_api_key_here

Key Models:
  - gemini-2.0-flash: Fast, efficient for most tasks
  - gemini-2.0-flash-thinking: Reasoning-focused
  - gemini-embedding-001: Text embeddings
"""

import os
from typing import List
from pydantic import BaseModel
import requests
from io import BytesIO

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
MODEL = "gemini-2.0-flash"  # Options: "gemini-2.0-flash", "gemini-2.0-flash-thinking"
EMBED_MODEL = "gemini-embedding-001"

# Initialize client (reads GEMINI_API_KEY from environment)
client = genai.Client()

# --- 1. List Available Models ---
def list_models():
    print("\n" + "="*50)
    print("Available Models")
    print("="*50)
    models = list(client.models.list())
    for m in models[:3]:  # Show first 3
        actions = getattr(m, "supported_actions", [])
        print(f"• {m.name} | {actions}")
    print(f"... and {len(models)-3} more models")

# --- 2. Basic Text Generation ---
def text_generation():
    print("\n" + "="*50)
    print("Text Generation")
    print("="*50)
    response = client.models.generate_content(
        model=MODEL,
        contents="Explain machine learning in 2 sentences."
    )
    print("Response:", response.text)
    print("Usage:", getattr(response, "usage_metadata", "N/A"))

# --- 3. Chat Session (Multi-turn) ---
def chat_session():
    print("\n" + "="*50)
    print("Multi-turn Chat")
    print("="*50)
    chat = client.chats.create(
        model=MODEL,
        history=[types.Content(role="user", parts=[types.Part(text="Hi, I'm learning Python.")])]
    )
    
    r1 = chat.send_message("What's a good first project?")
    print("Assistant:", r1.text[:100] + "...")
    
    r2 = chat.send_message("How long would that take?")
    print("Assistant:", r2.text[:100] + "...")
    
    # Show history
    print("\nHistory length:", len(chat.get_history()), "messages")

# --- 4. Streaming Response ---
def streaming_generation():
    print("\n" + "="*50)
    print("Streaming Generation")
    print("="*50)
    print("Stream: ", end="")
    stream = client.models.generate_content_stream(
        model=MODEL,
        contents="Count from 1 to 5 with explanations."
    )
    for chunk in stream:
        print(chunk.text, end="", flush=True)
    print("\n")

# --- 5. Vision (Image Understanding) ---
def vision_example():
    print("\n" + "="*50)
    print("Vision - Image Understanding")
    print("="*50)
    # Download sample image
    image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
    image_bytes = requests.get(image_url).content
    
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    
    response = client.models.generate_content(
        model=MODEL,
        contents=[image_part, "What's in this image? Be brief."]
    )
    print("Image analysis:", response.text)

# --- 6. Function Calling (Tools) ---
def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic math operations: add, subtract, multiply, divide"""
    ops = {
        "add": a + b,
        "subtract": a - b, 
        "multiply": a * b,
        "divide": a / b if b != 0 else float('inf')
    }
    return ops.get(operation, 0)

def function_calling():
    print("\n" + "="*50)
    print("Function Calling")
    print("="*50)
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(tools=[calculate])
    )
    
    response = chat.send_message("Calculate 23 * 47 and then add 100 to the result")
    print("Response:", response.text)

# --- 7. Structured Output (JSON) ---
class TaskList(BaseModel):
    title: str
    tasks: List[str]
    priority: str

def structured_output():
    print("\n" + "="*50)
    print("Structured JSON Output")
    print("="*50)
    response = client.models.generate_content(
        model=MODEL,
        contents="Create a simple daily routine for a software developer",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TaskList
        )
    )
    
    print("Raw JSON:", response.text)
    parsed: TaskList = response.parsed
    print("Parsed title:", parsed.title)
    print("Number of tasks:", len(parsed.tasks))

# --- 8. File Upload (Audio/Documents) ---
def file_upload_example():
    print("\n" + "="*50)
    print("File Upload")
    print("="*50)
    # This would work with actual files
    sample_text = "sample_document.txt"
    if os.path.exists(sample_text):
        uploaded_file = client.files.upload(file=sample_text)
        response = client.models.generate_content(
            model=MODEL,
            contents=["Summarize this document:", uploaded_file]
        )
        print("Summary:", response.text)
    else:
        print("Sample file not found. Upload syntax:")
        print("uploaded_file = client.files.upload(file='path/to/file')")
        print("response = client.models.generate_content(model='...', contents=[prompt, uploaded_file])")

# --- 9. Embeddings ---
def embeddings_example():
    print("\n" + "="*50)
    print("Text Embeddings")
    print("="*50)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents="Machine learning enables computers to learn patterns from data.",
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    
    embedding = result.embeddings[0].values
    print(f"Embedding length: {len(embedding)}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Vector norm: {sum(x**2 for x in embedding)**0.5:.4f}")

# --- 10. Error Handling Pattern ---
def error_handling_example():
    print("\n" + "="*50)
    print("Error Handling")
    print("="*50)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents="Test prompt",
            config=types.GenerateContentConfig(
                max_output_tokens=10,
                temperature=0.7,
                top_p=0.8
            )
        )
        print("Success:", response.text)
        
        # Check for blocked content
        if hasattr(response, 'prompt_feedback'):
            print("Safety ratings:", response.prompt_feedback)
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

def main():
    """Run all examples"""
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment")
        return
    
    print("🚀 Gemini SDK Quick Reference Examples")
    print(f"📋 Using Model: {MODEL}")
    print(f"📋 Using Embed Model: {EMBED_MODEL}")
    
    try:
        list_models()
        text_generation()
        chat_session()
        streaming_generation()
        vision_example()
        function_calling()
        structured_output()
        file_upload_example()
        embeddings_example()
        error_handling_example()
        
        print(f"\n✅ All examples completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
