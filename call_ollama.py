import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "/no_think "
                "Explique en deux phrases ce qu'est une API REST."
            ),
        }
    ],
    "think": False,
    "stream": False,
}


response = requests.post(
    OLLAMA_URL,
    json=payload,
    timeout=120,
)

response.raise_for_status()

data = response.json()
content = data["message"]["content"]

# Nettoyage éventuel du raisonnement retourné par certains modèles
if "</think>" in content:
    content = content.split("</think>", 1)[-1].strip()

print("Réponse du modèle :")
print(content)