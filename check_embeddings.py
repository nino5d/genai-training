import requests


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"


texts = [
    "La mission consiste à industrialiser des applications d'intelligence artificielle.",
    "Le consultant doit développer des solutions basées sur des modèles de langage.",
    "Le projet utilise Python, AWS et des API REST.",
]


response = requests.post(
    OLLAMA_EMBED_URL,
    json={
        "model": EMBEDDING_MODEL,
        "input": texts,
    },
    timeout=120,
)

response.raise_for_status()

data = response.json()
embeddings = data["embeddings"]

print(f"Nombre de textes : {len(texts)}")
print(f"Nombre de vecteurs : {len(embeddings)}")
print(f"Taille d'un vecteur : {len(embeddings[0])}")
print()
print("Début du premier vecteur :")
print(embeddings[0][:10])