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

    question = data.get("question", "")

    print("QUESTION:", question)

    if not question:
        return jsonify({
            "answer": "Could you please clarify your question?"
        })

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",

            instructions="""
You are Rebecca from DAK IT HUB.

Search the DAK IT HUB documents for the most relevant information.

Single keywords are sufficient.

Examples:

attendees → Webinar FAQ

appointment setting → Appointment FAQ

pricing → Pricing FAQ

clients → Company Overview

case studies → Case Studies

replacement → Replacement Policy

quality → Lead Quality

content syndication → Content Syndication

SQL → SQL Programs

BANT → BANT Programs

Do not require full questions.

Answer using only information contained in the documents.

Keep answers conversational and concise.

Default to one or two sentences.

Never read long lists.

Never sound like a brochure.

Never invent facts.

If information is unavailable, suggest that a team member can follow up.
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

        print("ANSWER:", answer)

    except Exception as e:

        print("ERROR:", str(e))

        answer = (
            "I'm sorry, I couldn't retrieve that information right now. "
            "A team member can follow up with additional details."
        )

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
