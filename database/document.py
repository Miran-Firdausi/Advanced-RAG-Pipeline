from sqlalchemy.orm import Session
from models.document import Document
from .connection import get_db


def get_doc_by_hash(doc_hash: str, db: Session = next(get_db())) -> Document | None:
    """
    Retrieve a document from the database using its hash.
    """
    return db.query(Document).filter(Document.hash == doc_hash).first()


def save_doc(
    doc_hash: str,
    filename: str,
    summary: str,
    table_data: dict,
    text_path: str,
    vectorstore_path: str,
    db: Session = next(get_db()),
) -> None:
    """
    Save a new document and its metadata to the database.
    """
    doc = Document(
        hash=doc_hash,
        filename=filename,
        summary=summary,
        table_data=table_data,
        text_path=text_path,
        vectorstore_path=vectorstore_path,
    )
    db.add(doc)
    db.commit()
