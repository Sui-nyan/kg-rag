import os
from dotenv import load_dotenv

import tiktoken
from neo4j import GraphDatabase
from openai import OpenAI
from mistralai.client import Mistral


load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI"),
    auth=(os.environ.get("NEO4J_USERNAME"), os.environ.get("NEO4J_PASSWORD")),
    notifications_min_severity="OFF"
)

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

def chunk_text(text, chunk_size, overlap, split_on_whitespace_only=True):
    chunks = []
    index = 0

    while index < len(text):
        if split_on_whitespace_only:
            prev_whitespace = 0
            left_index = index - overlap
            while left_index >= 0:
                if text[left_index] == " ":
                    prev_whitespace = left_index
                    break
                left_index -= 1
            next_whitespace = text.find(" ", index + chunk_size)
            if next_whitespace == -1:
                next_whitespace = len(text)
            chunk = text[prev_whitespace:next_whitespace].strip()
            chunks.append(chunk)
            index = next_whitespace + 1
        else:
            start = max(0, index - overlap + 1)
            end = min(index + chunk_size + overlap, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            index += chunk_size

    return chunks


def num_tokens_from_string(string: str, model: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def embed(texts, model="mistral-embed"):
    return mistral_client.embeddings.create(model=model, inputs=texts)


def chat(messages, model="gpt-4o", temperature=0, config={}):
    response = mistral_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        **config,
    )
    return response.choices[0].message.content


def tool_choice(messages, model="gpt-4o", temperature=0, tools=[], config={}):
    response = mistral_client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        tools=tools or None,
        **config,
    )
    return response.choices[0].message.tool_calls
