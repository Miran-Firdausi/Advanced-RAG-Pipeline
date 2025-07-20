def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def log_chunks(docs):
    print("\nRetrieved Chunks:\n" + "\n---\n".join(doc.page_content for doc in docs))
    return docs
