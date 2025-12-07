"""
Example: Generate embeddings for patent text using Azure OpenAI
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

# Example patent texts, stored in a list for batch processing
patent_texts = [
    "A method for packaging food products using biodegradable materials",
    "System and apparatus for beverage carbonation and dispensing",
    "Novel chocolate manufacturing process with improved texture",
]

# Generate embeddings for each patent text individually
print("Generating embeddings for individual patent texts...")
for i, text in enumerate(patent_texts, 1):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")

    embedding = response.data[0].embedding

    print(f"Patent {i}:")
    print(f"  Text: {text}")
    print(f"  Embedding dimension: {len(embedding)}")  # 1536 for text-embedding-3-small
    print(f"  First 3 values: {embedding[:3]}\n")

# For batch processing (more efficient):
print("Batch processing all texts at once...")
response = client.embeddings.create(
    input=patent_texts,  # Send all texts at once
    model="text-embedding-3-small",
)

embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings in one request")
