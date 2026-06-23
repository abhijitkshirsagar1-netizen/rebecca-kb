from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]


@app.route("/")
def home():
    return "Rebecca KB is running"


@app.route("/search", methods=["POST"])
def search():

    data = request.get_json()

    print("REQUEST DATA:", data)

    question = (
        data.get("message", {})
            .get("toolCalls", [{}])[0]
            .get("function", {})
            .get("arguments", {})
            .get("question", "")
    )

    print("QUESTION:", question)

    response = client.responses.create(
    model="gpt-4.1-mini",
    instructions="""
Answer only from DAK IT HUB documents.
Be concise.
Never say you don't have information if the documents contain the answer.
""",
    input=question,
    tools=[
        {
            "type":"file_search",
            "vector_store_ids":[VECTOR_STORE_ID]
        }
    ],
    max_output_tokens=150
)

    answer = response.output_text

    print("ANSWER:", answer)

    return jsonify(
        {
            "answer": answer
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
