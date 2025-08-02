import traceback
import requests
from fastapi import FastAPI, APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pipeline import handle_document
from database.connection import get_db
from database.document import get_doc_by_hash

app = FastAPI()

api_router = APIRouter(prefix="/api/v1")


@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.post("/hackrx/run")
async def answer_queries(request: Request):
    body = await request.json()
    doc_url = body.get("documents")
    questions = body.get("questions")

    try:
        # Download the document from the URL
        response = requests.get(doc_url)
        if response.status_code != 200:
            raise Exception("Failed to download the document from the provided URL.")

        file_bytes = response.content
        filename = doc_url.split("/")[-1].split("?")[0]

        # Process the document
        result = await handle_document(file_bytes, filename, questions)
        return JSONResponse(content=result)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


app.include_router(api_router)
