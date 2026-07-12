import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_fixed

from app.core.logger import configure_logging
from app.processing.generate_vector_db import load_vector_store


load_dotenv()

logger = configure_logging("GENERATE_RAG_CHAIN")


# Initialize Gemini LLM with retry handling
@retry(stop=stop_after_attempt(5), wait=wait_fixed(3))
def initialize_llm():
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        logger.info(f"Initializing Gemini LLM: {llm_model}")

        return ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            temperature=0,
            max_output_tokens=1024,
        )

    except Exception as e:
        logger.error(f"LLM init error: {e}")
        raise


# Create RAG chain
def create_rag_chain(vector_store):
    try:
        logger.info("Creating RAG chain")
        llm = initialize_llm()

        prompt_template = """
        You are an assistant that answers questions strictly based on the provided document text.

        Rules:
        - Only use the information from the given Context.
        - Do not use outside knowledge.
        - If the answer is not found in the Context, reply exactly: "Information not found in the document."
        - Provide only the answer, without repeating the question or the context.

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            ),
            chain_type_kwargs={"prompt": prompt},
        )

        logger.info("RAG chain created successfully")
        return chain

    except Exception as e:
        logger.error(f"RAG chain error: {e}")
        raise


if __name__ == "__main__":
    saved_vector_store_path = "app/data/vectorstores/faiss_index"

    vector_store = load_vector_store(saved_vector_store_path)
    rag_chain = create_rag_chain(vector_store)

    logger.info("RAG chain is ready for inference")