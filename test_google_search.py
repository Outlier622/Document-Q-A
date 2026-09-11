"""Opt-in live search check; run from the project root with local requirements."""

from pprint import pprint

from app.services.rag_service import extract_grounded_response, invoke_web_search


PROMPT = """
Use Google Search to find the current Gemini API rate-limit documentation.
Answer in two concise sentences and rely on current official sources.
"""


if __name__ == "__main__":
    response = invoke_web_search(PROMPT)
    result = extract_grounded_response(response)

    print("ANSWER:\n", result["answer"])
    print("\nWEB SEARCH USED:", result["web_search_used"])
    print("\nSOURCES:")

    for source in result["web_sources"]:
        print(f"- {source['title']}: {source['url']}")
        if source.get("cited_text"):
            print(f"  Supports: {source['cited_text']}")

    if not result["web_sources"]:
        print("No source URLs were extracted.")
        print("\nRAW RESPONSE METADATA:")
        pprint(getattr(response, "response_metadata", {}))
