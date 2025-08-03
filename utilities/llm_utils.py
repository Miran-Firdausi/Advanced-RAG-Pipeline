import os
from google import genai
from google.genai import types
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import Document
from prompts import BASIC_PROMPT, SIMPLIFY_USER_QUERY
from utilities.file_utils import is_doc_already_processed, load_pdf_using_PyPDF
from utilities.text_utils import format_docs, log_chunks
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# all_mini_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# bge_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# embeddings = bge_embeddings

index_name = "hackrx-embeddings"


def create_embeddings(doc_data, file_hash):

    if is_doc_already_processed(file_hash):
        print("Already processed. Skipping...")
        return

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )

    # text = doc_data.get("text")
    # docs = [Document(page_content=text)]

    # Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(doc_data)

    namespace = file_hash

    # Store vectors in Pinecone
    records = []
    for i, doc in enumerate(splits):
        records.append(
            {
                "_id": f"{file_hash}-chunk-{i}",
                "chunk_text": doc.page_content,
                "chunk_id": i,
                "file_hash": file_hash,
            }
        )
    batch_size = 96
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        pc.Index(index_name).upsert_records(namespace=namespace, records=batch)

    print("Embeddings created and uploaded to Pinecone!")


def answer_from_structured_data(file_hash, questions, summary):
    # Load vector store from Pinecone
    index = pc.Index(index_name)

    def retrieve(query):
        res = index.search(
            namespace=file_hash, query={"inputs": {"text": query}, "top_k": 5}
        )
        docs = [hit["fields"]["chunk_text"] for hit in res["result"]["hits"]]
        return docs

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = PromptTemplate.from_template(BASIC_PROMPT)

    rag_chain = (
        {
            "context": RunnablePassthrough() | retrieve | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        answers = []
        for question in questions:
            simplified_question = simplify_query(summary, question)
            print(simplified_question)
            answer = rag_chain.invoke(simplified_question)
            answers.append(answer)
    except Exception as e:
        print(e)

    return answers


async def generate_summary(document_file):
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = "Summarize this document"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=document_file,
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        summary = response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = "Could not generate summary."

    return summary


def simplify_query(summary, query):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=os.environ.get("GOOGLE_API_KEY_2"),
    )

    simplify_chain = SIMPLIFY_USER_QUERY | llm | StrOutputParser()

    simplified_question = simplify_chain.invoke({"summary": summary, "question": query})

    return simplified_question


# def answer_using_hyde(file_hash, query):
#     # Load vector store and LLM
#     vector_store = FAISS.load_local(
#         f"vector_store/{file_hash}", embeddings, allow_dangerous_deserialization=True
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )

#     # --- HyDe step: Generate hypothetical answer ---
#     hyde_prompt = f"""You are given a user query. Generate a concise, factual, and contextually relevant hypothetical answer that directly addresses the query.
#     This answer will be used for semantic search, so it should resemble the kind of language and format typically found in documents or knowledge bases.

#     Focus only on the core information asked in the query. Do not speculate or include unrelated details.

#     Query: {query}
#     Hypothetical Answer:"""

#     try:
#         hypo_answer = llm.predict(hyde_prompt).strip()
#         if not hypo_answer:
#             print(f"Empty hypothetical answer for query: {query}")
#             answer = "Could not generate a relevant answer."
#             return answer

#         # --- Embed the hypothetical answer for retrieval ---
#         hypo_embedding = embeddings.embed_query(hypo_answer)
#         docs = vector_store.similarity_search_by_vector(hypo_embedding, k=5)

#         # --- Build and run RAG chain ---
#         prompt = PromptTemplate.from_template(BASIC_PROMPT)
#         rag_chain = (
#             {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )

#         answer = rag_chain.invoke({"context": format_docs(docs), "question": query})
#     except Exception as e:
#         print(f"Error processing query '{query}': {e}")
#         answer = "Error occurred while processing the query."

#     return answer
