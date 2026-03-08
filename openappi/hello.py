import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize client with SumoPod AI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# Make a chat completion request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello in a creative way"}
    ],
    max_tokens=150,
    temperature=0.7
)

print(response.choices[0].message.content)