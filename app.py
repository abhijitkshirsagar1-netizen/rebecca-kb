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

    tool_calls = data.get("message", {}).get("toolCalls", [])

    question = ""

    if tool_calls:
        question = (
            tool_calls[0]
            .get("function", {})
            .get("arguments", {})
            .get("question", "")
        )

    print("QUESTION:", question)

    if not question:
        return jsonify({
            "results": "Could you please clarify your question?"
        })

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions="""
You are Rebecca from DAK IT HUB.

Answer questions using only the information contained in the DAK IT HUB documents.

Keep answers conversational and concise.

Default to one or two sentences.

Never invent facts.

If information is unavailable, politely suggest that a team member can follow up.
""",
            input=question,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID]
                }
            ],
            max_output_tokens=300
        )

        answer = response.output_text

    except Exception as e:

        print(e)

        answer = (
            "I'm sorry, I couldn't retrieve that information right now. "
            "A team member can follow up with additional details."
        )

    print("ANSWER:", answer)

    return jsonify({
        "results": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
