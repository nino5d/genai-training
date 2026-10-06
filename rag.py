import math
from functools import lru_cache

import requests


EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"


DOCUMENTS = [
    {
        "source": "mission.txt",
        "text": (
            "La mission consiste à industrialiser des POC existants "
            "en intelligence artificielle générative et à les intégrer "
            "dans une solution logicielle plus large."
        ),
    },
    {
        "source": "mission.txt",
        "text": (
            "Le consultant doit développer des briques d'intelligence "
            "artificielle en s'appuyant sur AWS et les modèles de langage "
            "du marché."
        ),
    },
    {
        "source": "mission.txt",
        "text": (
            "Les livrables comprennent le code source, les packages Git, "
            "les pipelines testés et la documentation technique en anglais."
        ),
    },
]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.json()["embeddings"]


@lru_cache(maxsize=1)
def get_document_embeddings() -> list[list[float]]:
    document_texts = [
        document["text"]
        for document in DOCUMENTS
    ]

    return get_embeddings(document_texts)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_context(
    question: str,
    top_k: int = 2,
) -> list[dict]:
    document_embeddings = get_document_embeddings()
    question_embedding = get_embeddings([question])[0]

    results = []

    for document, document_embedding in zip(
        DOCUMENTS,
        document_embeddings,
    ):
        score = cosine_similarity(
            question_embedding,
            document_embedding,
        )

        results.append(
            {
                "source": document["source"],
                "text": document["text"],
                "score": score,
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]