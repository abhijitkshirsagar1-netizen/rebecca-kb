from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# Vector Store ID
VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]


@app.route("/")
def home():
    return "Rebecca KB is running"


@app.route("/search", methods=["POST"])
def search():

    # Get JSON request
    data = request.get_json()

    # Debug logs
    print("REQUEST DATA:", data)

    # Extract question from Vapi
    question = data.get("question", "")

    print("QUESTION:", question)

    # Handle empty question
    if not question:
        return jsonify({
            "answer": "Could you please clarify your question?"
        })

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",

            instructions="""
You are Rebecca from DAK IT HUB.

Search the DAK IT HUB documents semantically.

Infer intent and meaning, not just exact words.

Examples:

attendance, turnout, show rates, people showing up
→ webinar attendees

calendar, meetings, scheduling, appointment setting
→ appointment support

rejects, bad leads, sales pushback, lead quality issues
→ replacement policy

commercials, pricing, cost, CPL, engagement model
→ pricing model

logos, references, IBM, Google, customers
→ clients and case studies

quality, validation, verification, double check
→ lead quality

pilot, trial, proof of concept, POC
→ pilot programs

Chinese, Japanese, Korean, multilingual
→ language support

SQL, BANT, MQL, HQL
→ lead qualification programs

webinar, registrations, audience acquisition
→ webinar programs

content, whitepaper, downloads, syndication
→ content syndication

Answer only from the DAK IT HUB documents.

Keep answers conversational and concise.

Default to one or two sentences.

Never read long lists.

Never sound like a brochure.

Never invent facts.

If information is unavailable, politely suggest that a team member can follow up with additional details.
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

        # Debug logs
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
