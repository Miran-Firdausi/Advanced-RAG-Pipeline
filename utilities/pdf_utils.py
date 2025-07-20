from langchain_community.document_loaders import PyPDFLoader


async def load_pdf_using_PyPDF(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    return pages
