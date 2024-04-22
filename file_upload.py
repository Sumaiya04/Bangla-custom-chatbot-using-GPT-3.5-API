import os
import openai
from openai import OpenAI
#client = OpenAI()
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY


# Ask the user to upload a file
user_input = input("Please upload a file (PDF, Excel, TXT): ")

# Process the user's input (you can validate the file type, etc.)
# For demonstration purposes, let's assume the user uploaded a PDF file

response = openai.chat.completions.create(
    messages= [
        {"role": "user", "content": f"User: {user_input}"},
        ],
    model="gpt-4",  # Use "model" instead of "engine"
    max_tokens=50
)
print(f"Assistant: {response.choices[0].message.content.strip()}")
