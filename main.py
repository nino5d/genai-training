import logging
import uuid

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import retrieve_context


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI Question Answering API",
    version="1.0.0",
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )
    use_rag: bool = False


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    request_id: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL,
    }


@app.post("/v1/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    request_id = str(uuid.uuid4())

    logger.info(
        "Processing request %s, use_rag=%s",
        request_id,
        request.use_rag,
    )

    sources = []

    try:
        if request.use_rag:
            retrieved_documents = retrieve_context(
                request.question,
                top_k=2,
            )

            context = "\n\n".join(
                document["text"]
                for document in retrieved_documents
            )

            sources = [
                document["source"]
                for document in retrieved_documents
            ]

            prompt = f"""
You are a technical assistant.

Answer the question only using the context below.
Do not invent information.
If the answer is not present in the context, say:
"I do not have enough information to answer."

Question:
{request.question}

Context:
{context}
"""
        else:
            prompt = f"""
Answer clearly and concisely in French.

Question:
{request.question}
"""

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "/no_think " + prompt,
                }
            ],
            "think": False,
            "stream": False,
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()
        answer = data["message"]["content"]

        if "</think>" in answer:
            answer = answer.split("</think>", 1)[-1].strip()

        return AnswerResponse(
            answer=answer,
            sources=sources,
            request_id=request_id,
        )

    except requests.Timeout:
        logger.exception(
            "Timeout for request %s",
            request_id,
        )

        raise HTTPException(
            status_code=504,
            detail={
                "code": "LLM_TIMEOUT",
                "message": "The language model did not respond in time.",
                "request_id": request_id,
            },
        )

    except requests.RequestException:
        logger.exception(
            "External service unavailable for request %s",
            request_id,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "code": "EXTERNAL_SERVICE_UNAVAILABLE",
                "message": "An external AI service is unavailable.",
                "request_id": request_id,
            },
        )

    except (KeyError, TypeError, ValueError):
        logger.exception(
            "Invalid response for request %s",
            request_id,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "code": "INVALID_RESPONSE",
                "message": "The AI service returned an invalid response.",
                "request_id": request_id,
            },
        )