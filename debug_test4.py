import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain in 3-4 sentences why aspirin might be relevant to EGFR, and note any ADMET concerns.",
    config=types.GenerateContentConfig(
        system_instruction="You explain computational drug candidate predictions to a non-specialist reader.",
        max_output_tokens=2048,
        temperature=0.4,
    ),
)

print("=== TEXT ===")
print(response.text)
print()
print("=== FINISH REASON ===")
print(response.candidates[0].finish_reason)
print()
print("=== USAGE ===")
print(response.usage_metadata)