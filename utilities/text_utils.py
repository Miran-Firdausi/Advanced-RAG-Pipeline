from langchain.load import dumps, loads


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def log_chunks(docs):
    print("\nRetrieved Chunks:\n" + "\n---\n".join(doc.page_content for doc in docs))
    return docs


def get_unique_union_of_documents(documents: list[list]):
    """Unique union of retrieved docs"""
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]
