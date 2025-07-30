from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    hash = Column(String, primary_key=True)
    filename = Column(String)
    summary = Column(Text)
    table_data = Column(JSON)
    text_path = Column(String)
    vectorstore_path = Column(String)
