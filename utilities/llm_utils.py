import os
from google import genai
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import Document
from prompts import BASIC_PROMPT
from utilities.text_utils import format_docs, log_chunks
from dotenv import load_dotenv

load_dotenv()

# all_mini_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
embeddings = bge_embeddings


def create_embeddings(doc_data, file_hash):
    # Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text = doc_data.get("text")
    splits = text_splitter.split_documents(doc_data)
    # docs = [Document(page_content=text)]

    vector_store_path = f"vector_store/{file_hash}"
    if not os.path.exists(vector_store_path):
        # Creating Vector Store
        vector_store = FAISS.from_documents(splits, embedding=embeddings)
        # Store the vector DB locally to save processing time
        vector_store.save_local(vector_store_path)

    print("Embeddings created!")


def answer_from_structured_data(file_hash, questions):
    vector_store = FAISS.load_local(
        f"vector_store/{file_hash}", embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    logged_retriever = retriever | format_docs

    prompt = PromptTemplate.from_template(BASIC_PROMPT)

    rag_chain = (
        {"context": logged_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        answers = []
        for question in questions:
            answer = rag_chain.invoke(question)
            answers.append(answer)
    except Exception as e:
        print(e)

    return answers


def get_document_summary(file_path):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        pdf_file = client.files.upload(file=pdf_bytes)
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=["prompt", pdf_file]
        )
        summary = response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = "Could not generate summary."

    return summary


# def rephrase_query(query):


def answer_using_hyde(file_hash, query):
    # Load vector store and LLM
    vector_store = FAISS.load_local(
        f"vector_store/{file_hash}", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # --- HyDe step: Generate hypothetical answer ---
    hyde_prompt = f"""You are given a user query. Generate a concise, factual, and contextually relevant hypothetical answer that directly addresses the query. 
    This answer will be used for semantic search, so it should resemble the kind of language and format typically found in documents or knowledge bases.

    Focus only on the core information asked in the query. Do not speculate or include unrelated details.

    Query: {query}
    Hypothetical Answer:"""

    try:
        hypo_answer = llm.predict(hyde_prompt).strip()
        if not hypo_answer:
            print(f"Empty hypothetical answer for query: {query}")
            answer = "Could not generate a relevant answer."
            return answer

        # --- Embed the hypothetical answer for retrieval ---
        hypo_embedding = embeddings.embed_query(hypo_answer)
        docs = vector_store.similarity_search_by_vector(hypo_embedding, k=5)

        # --- Build and run RAG chain ---
        prompt = PromptTemplate.from_template(BASIC_PROMPT)
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke({"context": format_docs(docs), "question": query})
    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        answer = "Error occurred while processing the query."

    return answer
