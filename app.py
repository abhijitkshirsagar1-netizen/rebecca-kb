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
    question = data.get("question", "").strip()

    print("QUESTION:", question)

    # Handle empty question
    if not question:
        return jsonify({
            "answer": "Could you please clarify your question?"
        })

    try:

        response = client.responses.create(

            model="gpt-4.1-mini",

            temperature=0.2,

            instructions="""
You are Rebecca from DAK IT HUB.

Use only information contained in the DAK IT HUB knowledge base.

Search the documents semantically.

Infer intent and meaning rather than relying on exact wording.

Understand synonyms and different ways prospects naturally ask questions.

Examples:

attendance, turnout, show rates, people showing up
→ webinar attendees

calendar, meetings, scheduling, appointment setting
→ appointment support

rejects, bad leads, sales pushback, lead quality issues
→ replacement policy

commercials, pricing, cost, CPL, investment
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

voice logs, recordings, call logs
→ voice logs

ROI, success rate, hit rate, performance
→ metrics and reporting

reports, dashboards, visibility
→ reporting

customers, references, logos, case studies
→ client references

Answer only from the DAK IT HUB knowledge base.

Answer only the question asked.

Keep answers conversational, warm and professional.

Default to one or two sentences.

Sound like a knowledgeable colleague, not a telemarketer.

Never sound like a brochure.

Never read long lists.

Never volunteer unnecessary information.

Never overwhelm with details.

Never invent facts.

Never guess.

Never make promises or guarantees that are not present in the knowledge base.

Do not guarantee webinar attendees or attendance.

Do not guarantee conversions or ROI.

Do not imply end-to-end calendar ownership.

Do not imply full appointment-setting ownership.

If multiple answers are available, provide the shortest and most relevant answer.

If the question is unclear, politely ask for clarification.

If information is unavailable, say:

"I'm sorry, I don't have that information available right now. A member of the team would be happy to help with that."

Keep answers short unless additional details are requested.

Always sound human, concise, helpful and trustworthy.
""",

            input=question,

            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [VECTOR_STORE_ID]
                }
            ],

            max_output_tokens=500

        )

        answer = response.output_text

        # Debug logs
        print("ANSWER:", answer)

    except Exception as e:

        print("ERROR:", str(e))

        answer = (
            "I'm sorry, I don't have that information available right now. "
            "A member of the team would be happy to help with that."
        )

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
