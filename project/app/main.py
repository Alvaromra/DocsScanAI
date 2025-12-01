import os
import io
import logging
from datetime import datetime
from typing import List, Optional

import pdfplumber
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

try:
    from transformers import pipeline  # opcional, só se modelos locais estiverem presentes
except Exception:
    pipeline = None

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI(title="Document Analyzer API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB setup (Postgres via docker-compose defaults) ---
DB_USER = os.getenv("POSTGRES_USER", "docsia_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "senha_super_secreta")
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "docsia")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    tipo_documento = Column(String, nullable=True)
    idioma = Column(String, nullable=True)
    resumo = Column(Text, nullable=True)
    palavras_chave = Column(Text, nullable=True)
    entidades = Column(Text, nullable=True)
    datas_importantes = Column(Text, nullable=True)
    valores_monetarios = Column(Text, nullable=True)
    riscos = Column(Text, nullable=True)
    observacoes = Column(Text, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.error("DB init failed: %s", exc)


init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


class DocumentOut(BaseModel):
    id: int
    filename: str
    tipo_documento: str = ""
    idioma: Optional[str] = None
    resumo: str = ""
    created_at: datetime

    class Config:
        orm_mode = True


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


def _join_list(values: List[str]) -> str:
    return ", ".join(v for v in values if v) if values else ""


def persist_analysis(
    db: Session,
    filename: str,
    content_type: Optional[str],
    size_bytes: int,
    result: AnaliseResponse,
) -> Optional[Document]:
    try:
        doc = Document(
            filename=filename,
            content_type=content_type,
            tipo_documento=result.tipo_documento,
            idioma=result.idioma,
            resumo=result.resumo,
            palavras_chave=_join_list(result.palavras_chave),
            entidades=_join_list(result.entidades),
            datas_importantes=_join_list(result.datas_importantes),
            valores_monetarios=_join_list(result.valores_monetarios),
            riscos=_join_list(result.riscos),
            observacoes=result.observacoes,
            size_bytes=size_bytes,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    except Exception as exc:
        db.rollback()
        logger.error("Falha ao salvar no banco: %s", exc)
        return None


def render_home(docs: List[Document]) -> str:
    items = []
    for d in docs:
        items.append(
            f"<div style='padding:8px;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:8px;'>"
            f"<div><strong>{d.filename}</strong> "
            f"(tipo: {d.tipo_documento or '—'} | idioma: {d.idioma or '—'})</div>"
            f"<div style='color:#374151;font-size:13px;'>Resumo: {d.resumo or '—'}</div>"
            f"<div style='font-size:12px;color:#6b7280;'>ID {d.id} · {d.created_at}</div>"
            f"</div>"
        )
    items_html = "".join(items) or "<p>Nenhum documento salvo ainda.</p>"
    return f"""
    <html>
      <head>
        <title>Document Analyzer</title>
        <style>
          body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }}
          h1 {{ color: #111827; }}
          form {{ margin-bottom: 16px; padding: 12px; border: 1px solid #e5e7eb; border-radius: 8px; background: #f9fafb; }}
          label {{ display:block; margin-bottom:6px; font-weight:600; }}
          input[type=file] {{ margin-bottom:10px; }}
          button {{ background:#2563eb; color:white; border:none; padding:8px 12px; border-radius:6px; cursor:pointer; }}
          button:hover {{ background:#1d4ed8; }}
        </style>
      </head>
      <body>
        <h1>Document Analyzer</h1>
        <p>Envie um PDF ou texto. O resultado será salvo no Postgres e exibido abaixo.</p>
        <form action="/web/upload" method="post" enctype="multipart/form-data">
            <label for="file">Arquivo:</label>
            <input type="file" id="file" name="file" required />
            <button type="submit">Enviar</button>
        </form>
        <h2>Últimos documentos</h2>
        {items_html}
        <p style="font-size:12px;color:#6b7280;">Para usar via API: POST /upload ou /analisar. Health: /health.</p>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).limit(25).all()
    return HTMLResponse(render_home(docs))


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    try:
        return db.query(Document).order_by(Document.created_at.desc()).limit(100).all()
    except Exception as exc:
        logger.error("Erro ao listar documentos: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao consultar o banco")


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
    except Exception as exc:
        logger.error("Erro ao buscar documento: %s", exc)
        raise HTTPException(status_code=500, detail="Erro ao consultar o banco")
    if not doc:
        raise HTTPException(status_code=404, detail="Documento não encontrado")
    return doc


@app.post("/analisar", response_model=AnaliseResponse)
def analisar(req: AnaliseRequest, db: Session = Depends(get_db), persist: bool = True):
    result = dummy_analysis(req.texto, req.arquivo)
    if persist:
        persist_analysis(db, req.arquivo, content_type="text/plain", size_bytes=len(req.texto.encode("utf-8")), result=result)
    return result


async def _analyze_upload(file: UploadFile, db: Session, persist: bool = True) -> AnaliseResponse:
    content = await file.read()
    texto = ""
    if file.filename.lower().endswith(".pdf"):
        texto = extract_text_from_pdf(content)
    else:
        try:
            texto = content.decode("utf-8", errors="ignore")
        except Exception:
            texto = ""
    result = dummy_analysis(texto, file.filename)
    if persist:
        persist_analysis(
            db,
            filename=file.filename,
            content_type=file.content_type,
            size_bytes=len(content),
            result=result,
        )
    return result


@app.post("/upload", response_model=AnaliseResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await _analyze_upload(file, db, persist=True)


@app.post("/web/upload", response_class=RedirectResponse)
async def web_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    await _analyze_upload(file, db, persist=True)
    return RedirectResponse(url="/", status_code=303)
