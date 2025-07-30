import os
from utilities.hashing import calculate_hash

# from utilities.redis_cache import get_cached_summary, cache_summary
from utilities.file_utils import get_doc_by_hash, save_doc_data, extract_using_textract
from utilities.redis_cache import get_cached_data
from utilities.llm_utils import answer_from_structured_data, create_embeddings


def handle_document(file_bytes, filename, questions):
    file_hash = calculate_hash(file_bytes)

    # Store file with filename (hash + file extension)
    os.makedirs("docs", exist_ok=True)
    _, ext = os.path.splitext(filename)
    file_path = os.path.join("docs", file_hash + ext)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file_bytes)
    else:
        print("File already exists!")

    # Step 1: Check cache
    # cached_data = get_cached_data(file_hash)
    # if cached_data:
    #     summary = cached_data.decode()

    doc_data = get_doc_by_hash(file_hash)
    if not doc_data:
        # Step 2: Process new doc
        doc_data = extract_using_textract(file_path, file_hash)
        json_data_path = f"docs/extracted/{file_hash}.json"
        save_doc_data(doc_data, json_data_path)

    # Step 3: Use LLM to answer questions
    create_embeddings(doc_data, file_hash)
    answers = answer_from_structured_data(file_hash, questions)
    return {"answers": answers}


# def process_document(file_bytes: bytes, filename: str):
#     doc_hash = get_sha256(file_bytes)

#     cached = get_cached_summary(doc_hash)
#     if cached:
#         return {"summary": cached.decode()}

#     doc = get_doc_by_hash(doc_hash)
#     if doc:
#         cache_summary(doc_hash, doc.summary)
#         return {"summary": doc.summary}

#     text = extract_text(file_bytes)
#     summary = get_summary(text)
#     table_data = extract_tables(text)
#     vec_path = create_and_save_vectorstore(doc_hash, text)

#     text_path = f"./storage/texts/{doc_hash}.txt"
#     with open(text_path, "w") as f:
#         f.write(text)

#     save_doc(doc_hash, filename, summary, table_data, text_path, vec_path)
#     cache_summary(doc_hash, summary)

#     return {"summary": summary}
