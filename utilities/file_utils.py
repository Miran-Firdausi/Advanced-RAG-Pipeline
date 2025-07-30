import os
import json
import time
import uuid
import boto3
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader

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


def start_textract_analysis(
    document_key: str, features: List[str] = ["TABLES", "FORMS"]
) -> str:
    """
    Starts an asynchronous Textract job for structured document analysis.
    Returns the Job ID.
    """
    bucket_name = "bajaj-hackrx"
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket_name, "Name": document_key}},
        FeatureTypes=features,
    )
    job_id = response["JobId"]
    print(f"Started Textract job: {job_id}")
    return job_id


def wait_for_job_completion(job_id: str, poll_interval=5, timeout=300) -> bool:
    """
    Polls Textract until the job completes or times out.
    """
    elapsed = 0
    while elapsed < timeout:
        response = textract.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        if status in ["SUCCEEDED", "FAILED"]:
            print(f"Textract Job Status: {status}")
            return status == "SUCCEEDED"
        time.sleep(poll_interval)
        elapsed += poll_interval
    print("Timeout waiting for Textract job.")
    return False


def get_full_textract_result(job_id: str):
    """Fetches all paginated results of Textract analysis."""
    pages = []
    response = textract.get_document_analysis(JobId=job_id)
    pages.append(response)

    while "NextToken" in response:
        response = textract.get_document_analysis(
            JobId=job_id, NextToken=response["NextToken"]
        )
        pages.append(response)

    return pages


def extract_text_and_tables(blocks: List[Dict]) -> Dict:
    """
    Parses Textract blocks into a structured format (simple version).
    You can later expand this for full table/form parsing.
    """
    lines = []
    tables = []
    forms = []

    for block in blocks:
        if block["BlockType"] == "LINE":
            lines.append(block["Text"])
        elif block["BlockType"] == "TABLE":
            tables.append(block)
        elif block["BlockType"] == "KEY_VALUE_SET":
            forms.append(block)

    return {
        "lines": lines,
        "tables_raw": tables,  # You can further process this into structured tables
        "forms_raw": forms,  # Can be parsed into key-value pairs
    }


def parse_textract_result(textract_response_pages):
    """Parses Textract result into readable structures: lines, tables, forms."""
    blocks = []
    for page in textract_response_pages:
        blocks.extend(page["Blocks"])

    block_map = {block["Id"]: block for block in blocks}

    lines = []
    tables = []

    for block in blocks:
        if block["BlockType"] == "LINE":
            lines.append(block["Text"])

        if block["BlockType"] == "TABLE":
            table = parse_table(block, block_map)
            tables.append(table)

    return {
        "text": "\n".join(lines),
        "tables": tables,
        # You can add 'forms': parse_forms(block_map) later
    }


def parse_table(table_block, block_map):
    """Reconstructs a table into a 2D list using CELL blocks."""
    cells = [
        block_map[rel["Ids"][0]]
        for rel in table_block.get("Relationships", [])
        if block_map[rel["Ids"][0]]["BlockType"] == "CELL"
    ]
    table = defaultdict(lambda: defaultdict(str))

    max_row, max_col = 0, 0
    for cell in cells:
        row, col = cell["RowIndex"], cell["ColumnIndex"]
        max_row, max_col = max(max_row, row), max(max_col, col)
        text = ""
        if "Relationships" in cell:
            for rel in cell["Relationships"]:
                if rel["Type"] == "CHILD":
                    text = " ".join(
                        [
                            block_map[cid]["Text"]
                            for cid in rel["Ids"]
                            if block_map[cid]["BlockType"] == "WORD"
                        ]
                    )
        table[row][col] = text

    # Convert to list of lists
    structured_table = []
    for row in range(1, max_row + 1):
        structured_table.append([table[row][col] for col in range(1, max_col + 1)])

    return structured_table


def parse_table(table_block, block_map):
    """Reconstructs a table into a 2D list using CELL blocks."""
    cells = [
        block_map[rel["Ids"][0]]
        for rel in table_block.get("Relationships", [])
        if block_map[rel["Ids"][0]]["BlockType"] == "CELL"
    ]
    table = defaultdict(lambda: defaultdict(str))

    max_row, max_col = 0, 0
    for cell in cells:
        row, col = cell["RowIndex"], cell["ColumnIndex"]
        max_row, max_col = max(max_row, row), max(max_col, col)
        text = ""
        if "Relationships" in cell:
            for rel in cell["Relationships"]:
                if rel["Type"] == "CHILD":
                    text = " ".join(
                        [
                            block_map[cid]["Text"]
                            for cid in rel["Ids"]
                            if block_map[cid]["BlockType"] == "WORD"
                        ]
                    )
        table[row][col] = text

    # Convert to list of lists
    structured_table = []
    for row in range(1, max_row + 1):
        structured_table.append([table[row][col] for col in range(1, max_col + 1)])

    return structured_table


def extract_using_textract(file_path: str, file_hash: str):
    """Entire pipeline to upload, analyze, and cache result locally."""
    file_name = upload_to_s3(file_path)

    job_id = start_textract_analysis(file_name)
    if not wait_for_job_completion(job_id):
        raise Exception("Textract job failed.")

    result = get_full_textract_result(job_id)
    return result


def get_doc_by_hash(doc_hash):
    path = os.path.join("docs", f"{doc_hash}.json")
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
