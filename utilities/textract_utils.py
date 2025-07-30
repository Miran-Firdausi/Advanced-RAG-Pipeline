import boto3
import time
from typing import List, Dict
from trp import Document

textract = boto3.client("textract")


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


def get_full_textract_result(job_id: str, file_hash):
    """Fetches all paginated results of Textract analysis."""
    pages = []
    response = textract.get_document_analysis(JobId=job_id)
    pages.append(response)

    while "NextToken" in response:
        response = textract.get_document_analysis(
            JobId=job_id, NextToken=response["NextToken"]
        )
        pages.append(response)

    # json_data_path = f"docs/{file_hash}.json"
    # save_doc_data(pages, json_data_path)

    return pages


def extract_all_text_from_doc(doc_data: list):
    text = ""
    docs = Document(doc_data)
    for page in docs.pages:
        text += page.text

    return text


def extract_all_tables_from_doc(doc_data: list) -> List[Dict]:
    """
    Parses all tables from all pages in the Textract response using TRP,
    and structures them into a list of dictionaries for LLM consumption.
    """
    doc = Document(doc_data)
    parsed_tables = []

    for i, page in enumerate(doc.pages):
        if not page.tables:
            continue

        for table in page.tables:
            # Format table as list of rows -> list of cells (strings)
            structured_table = []
            for row in table.rows:
                structured_row = [cell.text.strip() for cell in row.cells]
                structured_table.append(structured_row)

            parsed_tables.append({"page": i + 1, "table": structured_table})

    return parsed_tables


def format_table_for_llm(table_data: List[Dict]) -> str:
    """
    Converts the list of parsed tables into a markdown string or readable format for LLM.
    """
    formatted_output = ""

    for table_info in table_data:
        formatted_output += f"\n### Table from Page {table_info['page']}\n\n"

        table = table_info["table"]
        if not table:
            continue

        # Format as Markdown table if possible
        headers = table[0]
        formatted_output += "| " + " | ".join(headers) + " |\n"
        formatted_output += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in table[1:]:
            formatted_output += "| " + " | ".join(row) + " |\n"

    return formatted_output


def get_extracted_text_and_tables(job_id, file_hash):
    result = get_full_textract_result(job_id, file_hash)
    text = extract_all_text_from_doc(result)
    tables = extract_all_tables_from_doc(result)

    return {"text": text, "tables": tables}
