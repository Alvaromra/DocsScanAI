import os
import io
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber

try:
    from transformers import pipeline  # opcional, só se modelos locais estiverem presentes
except Exception:
    pipeline = None

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI(title="Document Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnaliseRequest(BaseModel):
    texto: str
    arquivo: str = "input.txt"


class AnaliseResponse(BaseModel):
    arquivo: str = ""
    tipo_documento: str = ""
    idioma: str = ""
    resumo: str = ""
    palavras_chave: List[str] = []
    entidades: List[str] = []
    datas_importantes: List[str] = []
    valores_monetarios: List[str] = []
    riscos: List[str] = []
    observacoes: str = ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def dummy_analysis(texto: str, arquivo: str) -> AnaliseResponse:
    # Substitua por chamada real a modelo local/remoto
    return AnaliseResponse(
        arquivo=arquivo,
        tipo_documento="",
        idioma="pt",
        resumo=texto[:200],
        palavras_chave=[],
        entidades=[],
        datas_importantes=[],
        valores_monetarios=[],
        riscos=[],
        observacoes="",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analisar", response_model=AnaliseResponse)
def analisar(req: AnaliseRequest):
    return dummy_analysis(req.texto, req.arquivo)


@app.post("/upload", response_model=AnaliseResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    texto = ""
    if file.filename.lower().endswith(".pdf"):
        texto = extract_text_from_pdf(content)
    else:
        try:
            texto = content.decode("utf-8", errors="ignore")
        except Exception:
            texto = ""
    return dummy_analysis(texto, file.filename)
