import os
from google import genai
from google.genai import types

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain in 3-4 sentences why aspirin might be relevant to EGFR, and note any ADMET concerns.",
    config=types.GenerateContentConfig(
        system_instruction="You explain computational drug candidate predictions to a non-specialist reader.",
        max_output_tokens=300,
        temperature=0.4,
    ),
)

print("=== FULL RESPONSE OBJECT ===")
print(response)
print()
print("=== CANDIDATES ===")
for cand in response.candidates:
    print("finish_reason:", cand.finish_reason)
    print("content parts:", cand.content.parts)
print()
print("=== USAGE ===")
print(response.usage_metadata)