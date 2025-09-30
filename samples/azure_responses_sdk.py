"""
Azure OpenAI GPT-5 - Minimal Examples via the responses SDK
Bare minimum syntax for each feature
"""

import os
import json
import base64
from time import time, sleep
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = OpenAI(
    api_key=os.getenv("FOUNDRY_API_KEY"),
    base_url=f"{os.getenv('FOUNDRY_PROJECT_ENDPOINT').rstrip('/')}/openai/v1/",
)

LLM_DEPLOYMENT_NAME = os.getenv("FOUNDRY_DEPLOYMENT_NAME")

def simple_llm_call():
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="Explain machine learning in one paragraph"
    )
    print("Full Response Object: ", response)
    print("LLM Response Text: ", response.output_text)

def structured_outputs():
    schema = {
        "format": {
            "type": "json_schema", 
            "name": "event_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "date": {"type": "string"}, 
                    "location": {"type": "string"}
                },
                "required": ["event", "date", "location"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="Plan a tech meetup",
        text=schema
    )
    
    print("Full response object:", response)
    
    # The JSON is in output[1].content[0].text
    json_text = response.output[1].content[0].text
    structured_data = json.loads(json_text)
    print("Parsed structured data:", structured_data)
    return structured_data




def conversation_context():
    response1 = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="What are software design patterns?"
    )
    print("First:", response1.output_text[:100] + "...")
    
    response2 = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        previous_response_id=response1.id,
        input="Give me an example of the first pattern you mentioned"
    )
    print("Follow-up:", response2.output_text[:100] + "...")

def tool_calling():
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather info",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }]

    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        tools=tools,
        input="What's the weather in Tokyo?"
    )
    
    for output in response.output:
        if output.type == "function_call":
            print(f"Called: {output.name} with {output.arguments}")
            
            follow_up = client.responses.create(
                model=LLM_DEPLOYMENT_NAME,
                previous_response_id=response.id,
                input=[{
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": '{"temp": "22°C", "condition": "sunny"}'
                }]
            )
            print("Result:", follow_up.output_text)
            return
    
    print("Direct response:", response.output_text)

def streaming():
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="Write about artificial intelligence",
        stream=True
    )

    for event in response:
        if event.type == 'response.output_text.delta':
            print(event.delta, end='', flush=True)
    print()

def file_processing():
    pdf_path = "../sample_data/337 Goldman Drive Inspection Report 20230730.pdf"
    
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    
    base64_string = base64.b64encode(pdf_data).decode("utf-8")
    #calculate the tokens of the base64 string via tiktoken
    
    # Use o200k_base encoding for GPT-4o and newer models like GPT-5
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except:
        # Fallback to cl100k_base if o200k_base is not available
        enc = tiktoken.get_encoding("cl100k_base")
    
    tokens = enc.encode(base64_string)
    print("Number of tokens:", len(tokens))

    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input=[{
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": "inspection_report.pdf",
                "file_data": f"data:application/pdf;base64,{base64_string}"
            }, {
                "type": "input_text",
                #"text": "Tell me what each page of this report talks about. Do not skip pages. For each page, start with 'Page X:'. Talk about what key info is on the page, the pictures, the insights, etc. Give a few sentence summary per page (if it has info and isnt just a table of contents of something). Describe the pictures if any. Make sure you describe each picture on the page, if there are 3 pictures you must talk about all 3. Clearly state the number of pictures on each page. "
                "text": "Describe each picture in this report, and what page each is on"
            }]
        }]
    )

    print("LLM Response Text: ", response.output_text)

def code_interpreter():
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        instructions="Run Python code to solve problems",
        input="Calculate 10 factorial and then divide by 3. Show the code."
    )
    print(response.output_text)

def reasoning_control():
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="If a store has 20% off sale and I buy a $50 item, and then I resell it for 80% of the original price, then buy it back for 60% of the resale price, what do I pay?",
        reasoning={"effort": "high"},
        text={"verbosity": "high"}
    )
    print(response.output_text)

def background_mode():
    response = client.responses.create(
        model=LLM_DEPLOYMENT_NAME,
        input="Write me a story about a magical kingdom",
        background=True
    )
    
    print(f"Initial status: {response.status}")
    
    # Poll for completion
    while response.status in {"queued", "in_progress"}:
        sleep(2)
        response = client.responses.retrieve(response.id)
        print(f"Current status: {response.status}")
    
    print("Final output:", response.output_text)



if __name__ == "__main__":
    
    
    simple_llm_call()
    #structured_outputs()
    #conversation_context()
    #tool_calling()
    #streaming()
    file_processing()
    #code_interpreter()
    #reasoning_control()
    #background_mode()
