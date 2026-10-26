"""Basic RAG lab (3-basic-rag.ipynb) as a script.

Workshop notebook:
https://github.com/Azure/ai-foundry-workshop/blob/273c24c/2-notebooks/1-chat_completion/3-basic-rag.ipynb

This script shows retrieval-augmented generation (RAG), the usual way to steer a model to answer from your own
documents instead of its general knowledge.

What it does:
1. Turns six health tips into embeddings and stores them in an Azure AI Search vector index.
2. For each question, finds the three closest tips and gives only those to the model as context, with an instruction
   to say "I'm not sure" when they don't cover the question.

Connection method:
- Models: the openai package's AzureOpenAI client, created directly with the Azure OpenAI endpoint (OPENAI_ENDPOINT)
  and an Entra ID token from the Azure CLI login.
- Azure AI Search: its own endpoint and API key from .env.
- Twin: 3-basic-rag-project-client.py gets the OpenAI client through the Foundry project SDK instead.

Run from the project root:
    uv run python src/1-models/3-basic-rag-azure-openai.py
"""

import os
import time

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from workshop_utils import http_log
from workshop_utils.settings import load_env

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_INDEX_NAME = "healthtips-index"

HEALTH_TIPS = [
    {
        "id": "doc1",
        "content": "Daily 30-minute walks help maintain a healthy weight and reduce stress.",
        "source": "General Fitness",
    },
    {"id": "doc2", "content": "Stay hydrated by drinking 8-10 cups of water per day.", "source": "General Fitness"},
    {
        "id": "doc3",
        "content": "Consistent sleep patterns (7-9 hours) improve muscle recovery.",
        "source": "General Fitness",
    },
    {"id": "doc4", "content": "For cardio endurance, try interval training like HIIT.", "source": "Workout Advice"},
    {
        "id": "doc5",
        "content": "Warm up with dynamic stretches before running to reduce injury risk.",
        "source": "Workout Advice",
    },
    {
        "id": "doc6",
        "content": "Balanced diets typically include protein, whole grains, fruits, vegetables, and healthy fats.",
        "source": "Nutrition",
    },
]

QUESTIONS = [
    "What's a good short cardio routine for me if I'm busy?",
    "How much water should I drink daily?",
]


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_environment():
    env_path = load_env(required=True)
    print(f"Loaded .env from: {env_path}")


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable is not set: {name}")
    return value


def create_openai_client():
    endpoint = require_env("OPENAI_ENDPOINT")
    api_version = require_env("OPENAI_API_VERSION")

    token_provider = get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SERVICES_SCOPE)
    openai_client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

    print("Connection method: AzureOpenAI (direct)")
    print(f"Endpoint:          {endpoint}")
    print(f"API version:       {api_version}")
    return openai_client


def embed(openai_client, model, text):
    return openai_client.embeddings.create(model=model, input=[text]).data[0].embedding


def create_healthtips_index(index_client, index_name, dimension):
    try:
        index_client.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    except ResourceNotFoundError:
        pass

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=dimension,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    index_client.create_index(SearchIndex(name=index_name, fields=fields, vector_search=vector_search))
    print(f"Created index: {index_name} (vector dimension {dimension})")


def wait_for_documents(search_client, expected, timeout_seconds=30):
    # Uploaded documents become searchable after a short indexing delay.
    deadline = time.time() + timeout_seconds
    while True:
        count = search_client.get_document_count()
        if count >= expected or time.time() > deadline:
            print(f"Searchable documents: {count}/{expected}")
            return count
        time.sleep(1)


def rag_chat(openai_client, search_client, embedding_model, chat_model, query, top_k=3):
    vector_query = VectorizedQuery(
        vector=embed(openai_client, embedding_model, query),
        k_nearest_neighbors=top_k,
        fields="embedding",
    )
    results = search_client.search(search_text="", vector_queries=[vector_query], select=["content", "source"])
    top_docs = [f"Source: {r['source']} => {r['content']}" for r in results]

    print("Retrieved tips:")
    for doc in top_docs:
        print(f"  {doc}")

    system_text = (
        "You are a health & fitness assistant.\n"
        "Answer user questions using ONLY the text from these docs.\n"
        "Docs:\n" + "\n".join(top_docs) + "\nIf unsure, say 'I'm not sure'.\n"
    )
    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    usage = response.usage
    print(f"(tokens: prompt {usage.prompt_tokens}, completion {usage.completion_tokens})")
    return response.choices[0].message.content


def main():
    print_section("BASIC RAG - AzureOpenAI")
    load_environment()
    chat_model = require_env("OPENAI_DEPLOYMENT_NAME")
    embedding_model = require_env("EMBEDDING_MODEL_DEPLOYMENT_NAME")
    search_endpoint = require_env("AZURE_AI_SEARCH_ENDPOINT")
    search_credential = AzureKeyCredential(require_env("AZURE_AI_SEARCH_API_KEY"))
    index_name = os.getenv("SEARCH_INDEX_NAME") or DEFAULT_INDEX_NAME
    print(f"Chat model:        {chat_model}")
    print(f"Embedding model:   {embedding_model}")
    print(f"Search endpoint:   {search_endpoint}")
    openai_client = create_openai_client()

    print_section("1. Build the vector index")
    vectors = [embed(openai_client, embedding_model, tip["content"]) for tip in HEALTH_TIPS]
    print(f"Embedded {len(vectors)} tips, vector length {len(vectors[0])}")

    index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)
    create_healthtips_index(index_client, index_name, len(vectors[0]))

    search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=search_credential)
    documents = [{**tip, "embedding": vector} for tip, vector in zip(HEALTH_TIPS, vectors, strict=True)]
    results = search_client.upload_documents(documents=documents)
    print(f"Uploaded {sum(r.succeeded for r in results)}/{len(documents)} documents to '{index_name}'")
    wait_for_documents(search_client, len(documents))

    for number, question in enumerate(QUESTIONS, start=1):
        print_section(f"{number + 1}. RAG question {number}")
        print(f"User: {question}")
        answer = rag_chat(openai_client, search_client, embedding_model, chat_model, question)
        print(f"Assistant: {answer}")

    print_section("DONE")


if __name__ == "__main__":
    http_log.enable()
    main()
