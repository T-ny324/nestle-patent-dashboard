"""
Example: Use GPT-5 to generate cluster labels and summaries
"""

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Create a prompt to analyze the cluster.
# The more specific the prompt, the better the results.
# Here, it is quite general so the response is open-ended and verbose.
prompt = """
What are the common themes among the following patents?

1. A method for packaging food products using biodegradable materials
2. System and apparatus for beverage carbonation and dispensing
3. Novel chocolate manufacturing process with improved texture
"""

response = client.chat.completions.create(
    model="gpt-5.1",  # Can also use "gpt-5-mini" for a smaller, quicker model
    messages=[
        {
            "role": "system",
            "content": "You are a patent analysis expert.",
        },  # System prompt
        {"role": "user", "content": prompt},  # User prompt
    ],
    max_completion_tokens=500,  # Optionally, limit the response length
)

result = response.choices[0].message.content  # Extract the generated content

print(f"Initial prompt to LLM: {prompt}")

print("--------------------------------")

print(f"LLM response: {result}")
