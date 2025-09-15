import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # loads .env file
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is CLIP model?. Explain it in one sentence.",
)

print(response.text)

# text = "Hello World!"
# result = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents=text,
#     config=types.EmbedContentConfig(output_dimensionality=10),
# )
# print(result.embeddings)