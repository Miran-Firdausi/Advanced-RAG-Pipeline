import os
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
    text = doc_data.get("text")
    docs = [Document(page_content=text)]
    splits = text_splitter.split_documents(docs)

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
