import os
import json
import boto3
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from utilities.textract_utils import (
    get_extracted_text_and_tables,
    start_textract_analysis,
    wait_for_job_completion,
)

# Initialize clients
s3 = boto3.client("s3")
textract = boto3.client("textract")


async def load_pdf_using_PyPDF(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    return pages


def upload_to_s3(file_path: str):
    """
    Uploads a document to S3 and returns the S3 object key.
    """
    bucket_name = "bajaj-hackrx"
    file_name = Path(file_path).name
    s3.upload_file(file_path, bucket_name, file_name)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{file_name}")
    return file_name


def extract_using_textract(file_path: str, file_hash: str):
    """Entire pipeline to upload, analyze, and cache result locally."""
    file_name = upload_to_s3(file_path)

    job_id = start_textract_analysis(file_name)
    if not wait_for_job_completion(job_id):
        raise Exception("Textract job failed.")

    result = get_extracted_text_and_tables(job_id, file_hash)
    return result


def get_doc_by_hash(doc_hash):
    path = os.path.join("docs", "extracted", f"{doc_hash}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_doc_data(result: dict, save_path: str):
    """Stores the result JSON locally for reuse."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[✓] Result cached locally at {save_path}")
