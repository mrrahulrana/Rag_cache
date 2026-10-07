NO_INFO_ANSWER = "I dont have information about quesry, please provide more information."

ANSWER_SYSTEM = f"""You answer questions using only the document excerpts provided in the user message.
Do not use outside knowledge, prior training facts, or the conversation itself as a source of facts.
If the excerpts do not contain the facts needed to answer, reply with exactly this sentence and nothing else:
{NO_INFO_ANSWER}
When the excerpts do contain the answer, give a concise answer and name the document and page or section you used.
"""

REWRITE_SYSTEM = """You rewrite the latest user question into one standalone question.
Use the summary and recent messages only to resolve pronouns and follow-up references.
Do not answer the question.
Return only the standalone question, with no preamble.
"""

SUMMARY_SYSTEM = """You compress a document-chat transcript into long-term memory.
Keep the user's goals, follow-up references, and document names they asked about.
Do not invent facts that were not stated in the transcript.
Write under 180 words.
"""


def is_no_info(answer: str) -> bool:
    return NO_INFO_ANSWER.casefold() in (answer or "").casefold()
