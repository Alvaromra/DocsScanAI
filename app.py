#!/usr/bin/env python3
# app.py – Mini Web App Premium (Salary AI Suite style)

from flask import Flask, request, render_template_string, redirect, url_for
from pathlib import Path
import subprocess
import json
import os
import argparse
from collections import Counter
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import requests
from concurrent.futures import ThreadPoolExecutor, Future
import threading
import re
import string
from textwrap import dedent

_OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URL = f"{_OLLAMA_BASE.rstrip('/')}/api/generate"
OLLAMA_MODEL_ANALYSIS = "minha-lora-json"
OLLAMA_MODEL_GENERAL = "minha-lora-docs"
OLLAMA_TIMEOUT = 120
OLLAMA_MODEL = OLLAMA_MODEL_ANALYSIS

app = Flask(__name__)

# Utilidades simples de formato
def format_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def call_local_llm(prompt: str, *, model: Optional[str] = None) -> str:
    """Chama o modelo registrado no Ollama e retorna o texto bruto."""
    target_model = model or OLLAMA_MODEL_ANALYSIS
    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception:
        return ""


def _load_text(file_name: str) -> Optional[str]:
    safe_name = Path(file_name).name
    candidates = [
        TEXT_DIR / safe_name,
        TEXT_DIR / f"{safe_name}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return None


def _parse_json_response(raw: str) -> Optional[dict]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = raw[first:last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
    return None


def get_doc_metadata(file_name: str) -> dict:
    """Busca metadados básicos do scanner para ajustar prompts."""
    meta = {}
    try:
        if INDEX_JSON.exists():
            data = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
            for d in data:
                if d.get("file_name") == file_name:
                    meta = d
                    break
    except Exception:
        meta = {}
    return meta


def validate_json_structure(data: dict) -> bool:
    """Valida campos esperados do JSON."""
    if not isinstance(data, dict):
        return False
    required_keys = [
        "tipo_documento",
        "resumo",
        "palavras_chave",
        "entidades",
        "datas_importantes",
        "valores_monetarios",
        "riscos",
        "observacoes",
    ]
    for k in required_keys:
        if k not in data:
            return False
    return True


def ensure_ia_analysis(file_name: str, model: Optional[str] = None) -> Tuple[dict, str]:
    """Gera análise com o modelo local se ainda não existir."""
    safe_name = Path(file_name).name
    ia_path = IA_DIR / f"{safe_name}.json"
    ia_raw_path = IA_DIR / f"{safe_name}.json.raw.txt"
    try:
        if ia_path.exists():
            data = json.loads(ia_path.read_text(encoding="utf-8"))
            return data, json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        pass

    text = _load_text(file_name)
    if not text:
        return {}, ""

    meta = get_doc_metadata(file_name)
    doc_type_hint = meta.get("doc_type") or meta.get("doc_type_guess")

    def _run_once(with_fix: bool, last_raw: str = "") -> Tuple[Optional[dict], str]:
        if with_fix and last_raw:
            fix_prompt = dedent(f"""
            Você respondeu algo fora do JSON válido. Corrija e devolva APENAS o JSON no schema:
            {{
              "tipo_documento": "...",
              "resumo": "...",
              "palavras_chave": ["..."],
              "entidades": ["..."],
              "datas_importantes": ["..."],
              "valores_monetarios": ["..."],
              "riscos": ["..."],
              "observacoes": "..."
            }}
            Resposta anterior:
            {last_raw}
            """).strip()
            raw = call_local_llm(fix_prompt, model=model or OLLAMA_MODEL_ANALYSIS)
            return _parse_json_response(raw), raw

        prompt = build_focused_prompt(file_name, text, doc_type=doc_type_hint)
        raw = call_local_llm(prompt, model=model or OLLAMA_MODEL_ANALYSIS)
        parsed_local = _parse_json_response(raw)
        return parsed_local, raw

    parsed, raw_response = _run_once(with_fix=False)
    if not parsed or not validate_json_structure(parsed):
        parsed, raw_response = _run_once(with_fix=True, last_raw=raw_response or "")
    if not parsed or not validate_json_structure(parsed):
        # guarda resposta crua para debug
        ia_raw_path.parent.mkdir(parents=True, exist_ok=True)
        ia_raw_path.write_text(raw_response or "", encoding="utf-8")
        return {}, ""

    parsed.setdefault("_model", model or OLLAMA_MODEL_ANALYSIS)
    parsed.setdefault("_timestamp", datetime.utcnow().isoformat())

    ia_path.parent.mkdir(parents=True, exist_ok=True)
    ia_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    if raw_response:
        ia_raw_path.parent.mkdir(parents=True, exist_ok=True)
        ia_raw_path.write_text(raw_response, encoding="utf-8")
    return parsed, json.dumps(parsed, ensure_ascii=False, indent=2)


def list_ia_status():
    entries = []
    pending = set(get_pending_files())
    for path in sorted(IA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        safe = path.name.replace(".json", "")
        entries.append({
            "file": safe,
            "path": path,
            "model": data.get("_model", OLLAMA_MODEL_ANALYSIS),
            "updated": data.get("_timestamp") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "doc_type": data.get("tipo_documento") or data.get("tipo"),
            "pending": safe in pending,
        })
    return entries

BASE_DIR = Path(__file__).parent
SAIDA_DIR = BASE_DIR / "saida"
TEXT_DIR = SAIDA_DIR / "texto_bruto"
IA_DIR = SAIDA_DIR / "ia_local"
UPLOAD_DIR = BASE_DIR / "documentos"
TRAIN_DIR = SAIDA_DIR / "treinamento"
DATASET_JSONL = TRAIN_DIR / "dataset.jsonl"
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_DIR = BASE_DIR / "models/minha-lora"
GGUF_PATH = BASE_DIR / "models/minha-lora-merged.gguf"

for d in [SAIDA_DIR, TEXT_DIR, IA_DIR, UPLOAD_DIR, TRAIN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

INDEX_JSON = SAIDA_DIR / "_docs_index.json"
ANALISE_JSON = SAIDA_DIR / "analise_detalhada.json"

executor = ThreadPoolExecutor(max_workers=2)
pending_tasks: Dict[str, Future] = {}
tasks_lock = threading.Lock()
pipeline_history = []
pipeline_lock = threading.Lock()


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_len: int = 1200) -> List[str]:
    """Quebra texto em pedaços aproximados."""
    parts = []
    buf = []
    count = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count + len(paragraph) > max_len and buf:
            parts.append("\n\n".join(buf))
            buf = []
            count = 0
        buf.append(paragraph)
        count += len(paragraph)
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def score_segment(seg: str) -> int:
    score = 0
    score += 2 * len(re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ]{2,}(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ]{2,})*\b", seg))
    score += len(re.findall(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{4}\b", seg))
    score += len(re.findall(r"R\$|\\$", seg))
    score += len(re.findall(r"%", seg))
    return score + max(1, len(seg) // 500)  # peso leve por tamanho


def select_relevant_sections(text: str, top_k: int = 3) -> List[str]:
    segments = chunk_text(text)
    scored = sorted(segments, key=lambda s: score_segment(s), reverse=True)
    return scored[:top_k]


def extract_focus_sections(text: str, max_len: int = 800) -> Dict[str, List[str]]:
    entities = re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ]{2,}(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ]{2,})*\b", text)
    dates = re.findall(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{4}\b", text)
    money = re.findall(r"\bR\$ ?[\d\.]+,\d{2}\b|\b\$ ?[\d,\.]+\b", text)

    def unique_trim(items):
        seen = set()
        cleaned = []
        for it in items:
            it = it.strip()
            if not it:
                continue
            if it.lower() in seen:
                continue
            seen.add(it.lower())
            cleaned.append(it[:max_len])
        return cleaned[:5]

    return {
        "entities": unique_trim(entities),
        "dates": unique_trim(dates),
        "money": unique_trim(money),
    }


def build_focused_prompt(file_name: str, text: str, doc_type: Optional[str] = None) -> str:
    text = normalize_text(text)
    focus = extract_focus_sections(text)
    trimmed_text = text[:6000]
    top_segments = select_relevant_sections(text)

    parts = ["Você é um analista especializado em documentos.", "Responda APENAS em JSON válido com campos:",
             "{\n  \"tipo_documento\": \"...\",\n  \"resumo\": \"...\",\n  \"palavras_chave\": [...],\n  \"entidades\": [...],\n  \"datas_importantes\": [...],\n  \"valores_monetarios\": [...],\n  \"riscos\": [...],\n  \"observacoes\": \"...\"\n}"]

    parts.append(f"Arquivo: {file_name}")
    if doc_type:
        parts.append(f"Tipo esperado (heurístico): {doc_type}")
    parts.append("CONTEÚDO ORIGINAL:\n" + trimmed_text)

    if any(focus.values()):
        parts.append("FOCOS PRINCIPAIS:")
        if focus["entities"]:
            parts.append("[Foco entidades]: " + " | ".join(focus["entities"]))
        if focus["dates"]:
            parts.append("[Foco datas]: " + " | ".join(focus["dates"]))
        if focus["money"]:
            parts.append("[Foco valores]: " + " | ".join(focus["money"]))

    if top_segments:
        parts.append("TRECHOS RELEVANTES (selecionados automaticamente):")
        for idx, seg in enumerate(top_segments, 1):
            parts.append(f"[Trecho {idx}]: {seg[:800]}")

    parts.append("Instruções finais: responda APENAS o JSON acima, sem texto extra.")
    return "\n".join(parts)


def _normalize_list_field(value) -> List[str]:
    """Converte strings/listas em lista limpa, separando por vírgula, ponto e vírgula ou quebras de linha."""
    if value is None:
        return []
    items: List[str] = []
    if isinstance(value, list):
        for v in value:
            if v is None:
                continue
            items.append(str(v))
    elif isinstance(value, str):
        items = re.split(r"[;,\\n\\|]", value)
    else:
        items = [str(value)]
    cleaned: List[str] = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if it not in cleaned:
            cleaned.append(it)
    return cleaned


def split_metrics_for_view(detailed: dict, ia: dict) -> Tuple[List[dict], List[dict]]:
    """Agrupa métricas em categóricas (nomes/textos) e numéricas."""
    cat: List[dict] = []
    num: List[dict] = []

    def add_cat(label, value, as_list: bool = False):
        if value is None:
            return
        if as_list:
            vals = _normalize_list_field(value)
            if not vals:
                return
            cat.append({"label": label, "value": vals, "is_list": True})
        else:
            text = str(value).strip()
            if text:
                cat.append({"label": label, "value": text, "is_list": False})

    def add_num(label, value):
        if value is None:
            return
        try:
            num.append({"label": label, "value": int(value)})
        except Exception:
            try:
                num.append({"label": label, "value": float(value)})
            except Exception:
                pass

    # Categóricas (nomes/textos)
    add_cat("Tipo (scanner)", detailed.get("doc_type"))
    add_cat("Tipo (IA)", ia.get("tipo_documento"))
    add_cat("Idioma", detailed.get("language_guess") or ia.get("idioma"))
    add_cat("Título detectado", detailed.get("title"))
    add_cat("Headings (amostra)", detailed.get("headings_sample"), as_list=True)
    add_cat("Palavras-chave", ia.get("palavras_chave"), as_list=True)
    add_cat("Entidades", ia.get("entidades"), as_list=True)
    add_cat("Datas importantes", ia.get("datas_importantes"), as_list=True)
    add_cat("Valores monetários", ia.get("valores_monetarios"), as_list=True)
    add_cat("Observações / Riscos", ia.get("observacoes") or ia.get("riscos"))

    # Numéricas
    add_num("Palavras", detailed.get("n_words"))
    add_num("Parágrafos", detailed.get("n_paragraphs"))
    add_num("Bullets", detailed.get("n_bullets"))
    add_num("Datas importantes (qtde)", len(_normalize_list_field(ia.get("datas_importantes"))))
    add_num("Valores monetários (qtde)", len(_normalize_list_field(ia.get("valores_monetarios"))))

    return cat, num


def submit_ia_task(file_name: str, model: Optional[str] = None):
    safe_name = Path(file_name).name

    def _run():
        try:
            ensure_ia_analysis(safe_name, model=model)
        finally:
            with tasks_lock:
                pending_tasks.pop(safe_name, None)

    with tasks_lock:
        fut = executor.submit(_run)
        pending_tasks[safe_name] = fut


def get_pending_files():
    with tasks_lock:
        return list(pending_tasks.keys())


def run_background_action(name: str, cmd: list[str]):
    record = {
        "name": name,
        "command": " ".join(cmd),
        "started": datetime.utcnow().isoformat(),
        "status": "running",
        "output": "",
    }
    with pipeline_lock:
        pipeline_history.insert(0, record)
        if len(pipeline_history) > 10:
            pipeline_history.pop()

    def _work():
        try:
            result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
            record["status"] = "ok" if result.returncode == 0 else "error"
            record["output"] = (result.stdout or "") + "\n" + (result.stderr or "")
        except Exception as exc:
            record["status"] = "error"
            record["output"] = str(exc)
        finally:
            record["finished"] = datetime.utcnow().isoformat()

    executor.submit(_work)


def gather_training_metrics():
    dataset_count = 0
    dataset_size = "0 B"
    dataset_mtime = "—"
    if DATASET_JSONL.exists():
        try:
            dataset_size = format_bytes(DATASET_JSONL.stat().st_size)
            dataset_mtime = datetime.fromtimestamp(DATASET_JSONL.stat().st_mtime).isoformat()
            dataset_count = sum(1 for _ in DATASET_JSONL.open(encoding="utf-8"))
        except Exception:
            pass

    ia_files = len(list(IA_DIR.glob("*.json")))
    pending = len(get_pending_files())
    processed = max(ia_files - pending, 0)

    lora_dir = str(LORA_DIR) if LORA_DIR.exists() else "—"
    gguf_path = str(GGUF_PATH) if GGUF_PATH.exists() else "—"

    models_list = []
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        data = resp.json()
        for m in data.get("models", []):
            models_list.append(m.get("name"))
    except Exception:
        models_list = []

    return {
        "dataset_count": dataset_count,
        "dataset_size": dataset_size,
        "dataset_mtime": dataset_mtime,
        "ia_files": ia_files,
        "pending": pending,
        "processed": processed,
        "lora_dir": lora_dir,
        "gguf_path": gguf_path,
        "ollama_models": models_list or ["—"],
    }

# ==========================
# Templates – Estilo “Salary AI Suite”
# ==========================

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>Central Documental</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --icm-blue-900: #091426;
      --icm-blue-800: #132540;
      --icm-blue-700: #1e3653;
      --icm-blue-600: #27435f;
      --icm-blue-500: #2f4e6d;
      --icm-cloud: #f4f6fb;
      --icm-sand: #fdfdf9;
      --icm-slate: #5c6c85;
      --icm-gold: #f4b844;
      --icm-red: #bb1626;
    }

    body {
      font-family: 'Nunito', 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg, var(--icm-cloud));
      color: var(--text, var(--icm-blue-900));
      margin: 0;
      min-height: 100vh;
    }

    .container-icm {
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 1.5rem;
    }

    .icm-hero {
      background: linear-gradient(135deg, rgba(7,16,30,0.92), rgba(25,46,74,0.88));
      color: var(--hero-text, #fff);
      padding: 3.5rem 0 2.5rem;
      position: relative;
      overflow: hidden;
    }

    .icm-hero:before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top right, rgba(244,184,68,0.18), transparent 55%);
      opacity: 0.9;
    }

    .icm-hero-inner {
      position: relative;
      z-index: 2;
    }

    .icm-breadcrumb {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: rgba(255,255,255,0.7);
      margin-bottom: 1rem;
    }

    .icm-hero h1 {
      font-weight: 700;
      font-size: clamp(2rem, 3vw, 2.8rem);
      margin-bottom: 0.75rem;
      color: var(--hero-text, #fff);
    }

    .icm-hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .icm-meta-card {
      background: rgba(11,33,58,0.65);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      padding: 0.9rem 1.4rem;
      min-width: 150px;
      box-shadow: 0 20px 30px rgba(0,0,0,0.35);
    }

    .icm-meta-card span {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(255,255,255,0.6);
      display: block;
    }

    .icm-meta-card strong {
      font-size: 1.6rem;
      color: #fff;
    }

    .icm-main {
      padding: 3rem 0 4rem;
    }

    .icm-form-card {
      background: var(--card, #fff);
      border-radius: 28px;
      box-shadow: 0 30px 80px rgba(5,12,28,0.08);
      padding: 2rem;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 2rem;
      margin-bottom: 2.5rem;
      border: 1px solid rgba(15,23,42,0.05);
    }

    .icm-form-side h2 {
      font-weight: 700;
      margin-bottom: 0.4rem;
      color: var(--text-strong, var(--icm-blue-800));
    }

    .icm-form-side p {
      color: var(--muted, var(--icm-slate));
      margin-bottom: 1.3rem;
    }

    .icm-form-side .form-control {
      border-radius: 16px;
      border: 1px solid rgba(19,37,64,0.15);
      padding: 0.85rem;
      background: #f9fafc;
      color: var(--icm-blue-800);
    }

    .icm-form-side button {
      border-radius: 999px;
      border: none;
      background: var(--icm-red);
      color: #fff;
      padding: 0.85rem 1.2rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      box-shadow: 0 12px 30px rgba(187,22,38,0.35);
    }

    .icm-form-note {
      font-size: 0.9rem;
      background: rgba(244,184,68,0.12);
      border-radius: 16px;
      padding: 0.9rem 1rem;
      margin-top: 1.2rem;
      color: var(--icm-blue-700);
      border: 1px solid rgba(244,184,68,0.35);
    }

    .icm-metrics-side {
      background: linear-gradient(135deg, #132944, #09182b);
      border-radius: 24px;
      padding: 1.5rem;
      color: #fff;
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }

    .icm-metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .icm-metric-pill {
      background: rgba(255,255,255,0.05);
      border-radius: 20px;
      padding: 0.75rem;
      border: 1px solid rgba(255,255,255,0.1);
    }

    .icm-metric-pill span {
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.7);
    }

    .icm-metric-pill strong {
      display: block;
      font-size: 1.3rem;
      color: var(--hero-text, #fff);
    }

    .icm-chart-card {
      background: rgba(255,255,255,0.04);
      border-radius: 18px;
      padding: 1rem;
      border: 1px solid rgba(255,255,255,0.08);
    }

    .icm-chart-card p {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: rgba(255,255,255,0.6);
      margin-bottom: 0.4rem;
    }

    .icm-table-card {
      background: var(--card, #fff);
      border-radius: 28px;
      box-shadow: 0 35px 80px rgba(9,22,38,0.08);
      padding: 2rem;
      border: 1px solid rgba(15,23,42,0.06);
    }

    .icm-table-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 1rem;
      align-items: center;
      margin-bottom: 1.5rem;
    }

    .icm-table-header h3 {
      margin: 0;
      font-weight: 700;
      color: var(--icm-blue-800);
    }

    .icm-filters {
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
    }

    .icm-filters label {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--icm-slate);
      display: block;
      margin-bottom: 0.2rem;
    }

    .icm-filters .form-control,
    .icm-filters .form-select {
      border-radius: 16px;
      border: 1px solid rgba(19,37,64,0.12);
      min-width: 200px;
      background: #f9fafc;
    }

    table {
      color: var(--icm-blue-900);
      font-size: 0.92rem;
    }

    thead {
      background: var(--table-head, #0f172a);
      color: var(--table-head-text, #fff);
    }

    .icm-table-card table td,
    .icm-table-card table th {
      color: var(--text, #0b1527);
    }

    th {
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      border: none;
    }

    tbody tr {
      border-bottom: 1px solid rgba(19,37,64,0.1);
    }

    tbody tr:hover {
      background: rgba(19,37,64,0.04);
    }

    body.theme-dark tbody tr:hover {
      background: rgba(255, 255, 255, 0.06);
    }

    .icm-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid rgba(19,37,64,0.15);
      padding: 0.2rem 0.8rem;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text, var(--icm-blue-700));
      background: var(--badge-bg, #f6f9ff);
    }

    .icm-ext {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 52px;
      height: 34px;
      border-radius: 12px;
      font-weight: 600;
      text-transform: uppercase;
      background: var(--pill-bg, rgba(9,20,38,0.06));
      color: var(--text, var(--icm-blue-700));
    }

    .icm-empty {
      text-align: center;
      padding: 4rem 1rem;
      color: var(--icm-slate);
    }

    .btn-outline-info {
      border-radius: 999px;
      border-color: var(--icm-blue-700);
      color: var(--icm-blue-700);
    }

    .btn-outline-info:hover {
      background: var(--icm-blue-700);
      color: #fff;
    }

    @media (max-width: 992px) {
      .icm-form-card {
        grid-template-columns: 1fr;
        padding: 1.5rem;
      }
      .icm-filters .form-control,
      .icm-filters .form-select {
        min-width: 100%;
      }
      .icm-hero-meta {
        flex-direction: column;
      }
    }

    @media (max-width: 576px) {
      .icm-table-card {
        padding: 1.25rem;
      }
    }

    /* Paleta dinâmica */
    :root {
      --bg: #f4f6fb;
      --text: #0b1527;
      --text-strong: #132540;
      --muted: #5c6c85;
      --card: #ffffff;
      --table-head: #0f172a;
      --table-head-text: #ffffff;
      --pill-bg: rgba(15,23,42,0.05);
      --badge-bg: rgba(11,33,58,0.08);
      --upload-bg: #f6f8fc;
      --hero-text: #ffffff;
    }

    body.theme-dark {
      --bg: #070b14;
      --text: #f8fbff;
      --text-strong: #ffffff;
      --muted: #dbeafe;
      --card: #0c1424;
      --table-head: #0f172a;
      --table-head-text: #f8fbff;
      --pill-bg: #1f2937;
      --badge-bg: #1f2937;
      --upload-bg: #0c1424;
      --hero-text: #ffffff;
    }

    body.theme-dark .icm-table-card {
      background: #0c1424;
      border-color: rgba(255,255,255,0.24);
      color: #f8fbff;
      box-shadow: 0 28px 70px rgba(0,0,0,0.38);
    }

    body.theme-dark .icm-table-card table td,
    body.theme-dark .icm-table-card table th {
      color: #f8fbff;
    }

    body.theme-dark .icm-breadcrumb,
    body.theme-dark .icm-table-header h3,
    body.theme-dark .icm-form-side p {
      color: #e5eaf1;
    }

    /* Forçar títulos e headings claros no dark mode (inclui blocos de insights) */
    body.theme-dark h1,
    body.theme-dark h2,
    body.theme-dark h3,
    body.theme-dark h4,
    body.theme-dark h5,
    body.theme-dark h6 {
      color: #fdfdfd;
    }

    body.theme-dark .icm-pill,
    body.theme-dark .icm-ext {
      color: #f8fbff;
      background: rgba(255,255,255,0.18);
      border: 1px solid rgba(255,255,255,0.24);
    }

    body.theme-dark thead {
      background: var(--table-head);
      color: var(--table-head-text);
    }

    body.theme-dark tbody tr:hover {
      background: rgba(255,255,255,0.05);
    }
  </style>
</head>
<body class="{{ 'theme-dark' if request.cookies.get('theme')=='dark' else '' }}">
  <header class="icm-hero">
    <div class="container-icm icm-hero-inner">
      <div class="icm-breadcrumb"></div>
      <h1>Central documental Neural</h1>
      <div class="icm-hero-meta">
        <div class="icm-meta-card">
          <span>Documentos</span>
          <strong>{{ total_docs }}</strong>
        </div>
        <div class="icm-meta-card">
          <span>Tipos</span>
          <strong>{{ type_counts|length }}</strong>
        </div>
        <div class="icm-meta-card">
          <span>Extensões</span>
          <strong>{{ ext_counts|length }}</strong>
        </div>
        <div class="icm-meta-card">
          <span>Tam. médio</span>
          <strong>{{ avg_size_kb }} KB</strong>
        </div>
      </div>
      <div class="mt-3 d-flex flex-wrap gap-2">
        <a href="{{ url_for('treino') }}" class="btn btn-warning text-dark fw-semibold">Painel de Treinamento</a>
        <a href="{{ url_for('ia_status') }}" class="btn btn-outline-light">Status IA</a>
        <button id="themeToggle" class="btn btn-outline-light">Alternar tema</button>
      </div>
    </div>
  </header>

  <main class="icm-main">
    <div class="container-icm">
      <section class="icm-form-card">
        <div class="icm-form-side">
          <h2>Envio rápido</h2>
          <p>Arraste ou escolha seus arquivos e deixe o pipeline executar como no painel oficial.</p>
          <form action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data" class="mb-3">
            <label class="form-label fw-semibold text-uppercase small text-muted">Arquivos</label>
            <input type="file" name="file" class="form-control" multiple required>
            <div class="d-grid mt-3">
              <button type="submit">Enviar e processar</button>
            </div>
          </form>
          <div class="icm-form-note">
            O fluxo chama <code>scanner_docs.py</code> automaticamente. Rode <code>ia_local_analise.py</code> se quiser gerar tudo em lote ou <a href="{{ url_for('ia_status') }}">acompanhe o status das análises</a>.
          </div>
        </div>
        <div class="icm-metrics-side">
          <div class="icm-metrics-grid">
            <div class="icm-metric-pill">
              <span>Docs únicos</span>
              <strong>{{ total_docs }}</strong>
            </div>
            <div class="icm-metric-pill">
              <span>Tipos</span>
              <strong>{{ type_counts|length }}</strong>
            </div>
            <div class="icm-metric-pill">
              <span>Extensões</span>
              <strong>{{ ext_counts|length }}</strong>
            </div>
            <div class="icm-metric-pill">
              <span>Médio KB</span>
              <strong>{{ avg_size_kb }}</strong>
            </div>
          </div>
          <div class="row g-3">
            <div class="col-md-6">
              <div class="icm-chart-card">
                <p>Tipos</p>
                <canvas id="chartTypes" height="160"></canvas>
              </div>
            </div>
            <div class="col-md-6">
              <div class="icm-chart-card">
                <p>Extensões</p>
                <canvas id="chartExts" height="160"></canvas>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="icm-table-card mb-4">
        <div class="icm-table-header">
          <div>
            <div class="icm-breadcrumb" style="color: var(--icm-slate); margin-bottom: 0.2rem;">Insights das atividades</div>
            <h3>O que aplicar no projeto</h3>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-lg-6">
            <h6 class="text-uppercase text-muted small">Atenção (Atividade 5)</h6>
            <ul class="mb-3">
              <li>Prompts focados: chunk do texto + blocos de foco (entidades, datas, riscos) antes das instruções.</li>
              <li>Attention-guided extraction: pré-selecionar trechos via TF-IDF/BM25/embeddings e destacar no prompt.</li>
              <li>Soft alignment: limpar/alinhavar campos do JSON do dataset antes do LoRA.</li>
            </ul>
          </div>
          <div class="col-lg-6">
            <h6 class="text-uppercase text-muted small">CNN / Feature Maps (Atividade 4)</h6>
            <ul class="mb-3">
              <li>Pré-processamento robusto: normalizar texto, remover símbolos, segmentar em janelas menores.</li>
              <li>Text feature maps: extrair n-grams, entidades, tópicos, datas/valores e inserir como contexto extra.</li>
              <li>Camadas de reforço: texto bruto → entidades → JSON → validação → feedback humano.</li>
            </ul>
          </div>
        </div>
        <div class="row g-3">
          <div class="col-lg-12">
            <h6 class="text-uppercase text-muted small">Resumo executivo</h6>
            <div class="row row-cols-2 row-cols-lg-4 g-2">
              <div><span class="icm-badge">Encoder-Decoder</span> Prompts e pré-processamento melhores.</div>
              <div><span class="icm-badge">Atenção</span> Extração focada e dataset limpo.</div>
              <div><span class="icm-badge">Pooling</span> Resumo automático pré-IA.</div>
              <div><span class="icm-badge">Classificação</span> Roteamento automático de documentos.</div>
            </div>
            <div class="mt-3 text-muted small">
              Ganhos: autoaprendizado mais inteligente, JSON mais consistente, menos alucinação, LoRA mais eficiente, arquitetura alinhada a T5/Llama/Mistral.
            </div>
          </div>
        </div>
      </section>

      <section class="icm-table-card">
        <div class="icm-table-header">
          <div>
            <div class="icm-breadcrumb" style="color: var(--icm-slate); margin-bottom: 0.2rem;">Listagem</div>
            <h3>Documentos catalogados</h3>
          </div>
      <div class="icm-filters">
            <div>
              <label for="searchInput">Buscar</label>
              <input type="text" id="searchInput" class="form-control form-control-sm" placeholder="Nome, tipo ou extensão">
            </div>
            <div>
              <label for="filterType">Tipo</label>
              <select id="filterType" class="form-select form-select-sm">
                <option value="">Todos</option>
                {% for t in type_counts.keys() %}
                  <option value="{{ t }}">{{ t or "Desconhecido" }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label for="filterExt">Extensão</label>
              <select id="filterExt" class="form-select form-select-sm">
                <option value="">Todas</option>
                {% for e in ext_counts.keys() %}
                  <option value="{{ e }}">{{ e or "—" }}</option>
                {% endfor %}
              </select>
            </div>
          </div>
        </div>

        {% if docs %}
          <div class="table-responsive">
            <table class="table align-middle">
              <thead>
                <tr>
                  <th class="ps-2">Arquivo</th>
                  <th>Tipo</th>
                  <th>Extensão</th>
                  <th>Tamanho</th>
                  <th class="text-end pe-2">Ações</th>
                </tr>
              </thead>
              <tbody id="docsTable">
                {% for doc in docs %}
                  {% set t = (doc.doc_type or 'desconhecido') %}
                  {% set ext = (doc.file_type_ext or '—') %}
                  <tr data-name="{{ doc.file_name|lower }}" data-type="{{ t|lower }}" data-ext="{{ ext|lower }}">
                    <td class="ps-2">
                      <div class="fw-semibold">{{ doc.file_name }}</div>
                      <div class="text-muted small">{{ doc.file_path }}</div>
                    </td>
                    <td>
                      <span class="icm-badge">{{ t }}</span>
                    </td>
                    <td>
                      <span class="icm-ext">{{ ext }}</span>
                    </td>
                    <td class="text-muted small">
                      {% if doc.file_size_bytes %}
                        ~ {{ (doc.file_size_bytes/1024)|round(1) }} KB
                      {% else %}
                        —
                      {% endif %}
                    </td>
                    <td class="text-end pe-2 d-flex gap-2 justify-content-end">
                      <a class="btn btn-sm btn-outline-info" href="{{ url_for('ver', file_name=doc.file_name) }}">
                        Ver
                      </a>
                      <form method="post" action="{{ url_for('delete_file', file_name=doc.file_name) }}" onsubmit="return confirm('Remover {{ doc.file_name }}? Essa ação é permanente.');">
                        <button class="btn btn-sm btn-outline-danger" type="submit">Remover</button>
                      </form>
                    </td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <div class="icm-empty">
            Nenhum documento indexado. Suba arquivos para liberar os cards, como no site de referência.
          </div>
        {% endif %}
      </section>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script>
    (function() {
      const body = document.body;
      const btn = document.getElementById('themeToggle');
      const key = 'theme';
      function setTheme(value) {
        if (value === 'dark') body.classList.add('theme-dark');
        else body.classList.remove('theme-dark');
        localStorage.setItem(key, value);
      }
      const saved = localStorage.getItem(key);
      if (saved) setTheme(saved);
      if (btn) {
        btn.addEventListener('click', () => {
          const next = body.classList.contains('theme-dark') ? 'light' : 'dark';
          setTheme(next);
        });
      }
    })();

    const typeLabels = {{ type_counts.keys()|list|tojson }};
    const typeValues = {{ type_counts.values()|list|tojson }};
    const extLabels  = {{ ext_counts.keys()|list|tojson }};
    const extValues  = {{ ext_counts.values()|list|tojson }};

    const ctxTypes = document.getElementById('chartTypes');
    if (ctxTypes) {
      new Chart(ctxTypes, {
        type: 'doughnut',
        data: {
          labels: typeLabels,
          datasets: [{
            data: typeValues,
            borderWidth: 0,
            backgroundColor: [
              '#f4b844','#e86d3d','#bb1626','#6aa9ff','#31c48d','#f2695c','#9da4ff'
            ]
          }]
        },
        options: {
          plugins: {
            legend: { labels: { color: '#fff', font: { size: 11 } } }
          }
        }
      });
    }

    const ctxExts = document.getElementById('chartExts');
    if (ctxExts) {
      new Chart(ctxExts, {
        type: 'bar',
        data: {
          labels: extLabels,
          datasets: [{
            data: extValues,
            borderWidth: 0,
            backgroundColor: '#f4b844',
            borderRadius: 6
          }]
        },
        options: {
          scales: {
            x: { ticks: { color: '#fff', font: { size: 11 } }, grid: { display: false } },
            y: { ticks: { color: '#9fb3d1', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.1)' } }
          },
          plugins: { legend: { display: false } }
        }
      });
    }

    const searchInput = document.getElementById('searchInput');
    const filterType  = document.getElementById('filterType');
    const filterExt   = document.getElementById('filterExt');
    const table       = document.getElementById('docsTable');

    function applyFilters() {
      if (!table) return;
      const term = (searchInput?.value || '').toLowerCase();
      const t    = (filterType?.value || '').toLowerCase();
      const e    = (filterExt?.value || '').toLowerCase();

      const rows = table.querySelectorAll('tr');
      rows.forEach(row => {
        const name = row.dataset.name || '';
        const type = row.dataset.type || '';
        const ext  = row.dataset.ext || '';
        let visible = true;

        if (term && !(name.includes(term) || type.includes(term) || ext.includes(term))) {
          visible = false;
        }
        if (t && type !== t) visible = false;
        if (e && ext !== e) visible = false;

        row.style.display = visible ? '' : 'none';
      });
    }

    if (searchInput) searchInput.addEventListener('input', applyFilters);
    if (filterType)  filterType.addEventListener('change', applyFilters);
    if (filterExt)   filterExt.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""

DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>Análise · {{ file_name }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --bg: #e9eef5;
      --card: #ffffff;
      --card-dark: #0f1524;
      --text: #0c1422;
      --text-soft: #7a829e;
      --accent: #6f5bff;
      --accent-2: #f964a3;
      --border: rgba(12,20,34,0.08);
    }

    * { box-sizing: border-box; }

    body {
      background: linear-gradient(180deg, #f4f7fb, #e6ebf3);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 0;
      min-height: 100vh;
    }

    .shell {
      max-width: 1140px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 4rem;
    }

    .hero {
      background: var(--card);
      border-radius: 40px;
      padding: 2rem 2.4rem;
      box-shadow: 0 25px 60px rgba(15,21,36,0.15);
      margin-bottom: 2rem;
      border: 1px solid var(--border);
    }

    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.3rem;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .hero a {
      text-decoration: none;
      text-transform: uppercase;
      letter-spacing: .2em;
      font-size: .7rem;
      color: var(--text-soft);
      display: inline-flex;
      align-items: center;
      gap: .4rem;
    }

    .hero h1 {
      font-size: clamp(1.9rem, 3vw, 2.6rem);
      margin: 0;
      font-weight: 700;
    }

    .tag {
      display: inline-flex;
      padding: .5rem 1.4rem;
      border-radius: 999px;
      background: #f0f2ff;
      color: var(--accent);
      border: 1px solid rgba(111,91,255,0.2);
      letter-spacing: .15em;
      text-transform: uppercase;
      font-size: .75rem;
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 7fr) minmax(0, 5fr);
      gap: 1.5rem;
    }

    .card {
      background: var(--card);
      border-radius: 34px;
      padding: 2rem;
      border: 1px solid var(--border);
      box-shadow: 0 30px 70px rgba(15,21,36,0.12);
    }

    .card h2 {
      text-transform: uppercase;
      letter-spacing: .16em;
      font-size: 1rem;
      color: var(--text-soft);
      margin-bottom: 1.4rem;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .chip {
      border-radius: 16px;
      padding: 1rem 1.1rem;
      background: #f7f8ff;
      border: 1px solid rgba(15,21,36,0.05);
    }

    .chip span {
      display: block;
      text-transform: uppercase;
      letter-spacing: .18em;
      font-size: .7rem;
      color: var(--text-soft);
    }

    .chip strong {
      font-size: 1.4rem;
      font-weight: 600;
      display: block;
      margin-top: .2rem;
    }

    .section {
      border-top: 1px solid var(--border);
      padding: 1rem 0;
    }

    .section:first-of-type {
      border-top: none;
      padding-top: 0;
    }

    .section h3 {
      text-transform: uppercase;
      letter-spacing: .2em;
      font-size: .72rem;
      color: var(--text-soft);
      margin-bottom: .45rem;
    }

    .muted { color: var(--text-soft); font-size: .9rem; }

    pre {
      white-space: pre-wrap;
      background: #f9f5ff;
      border: 1px solid rgba(249,101,163,0.15);
      color: var(--text);
      border-radius: 18px;
      padding: 1rem;
      font-size: .9rem;
    }

    .card-alt {
      background: #fdf5f8;
      border-color: rgba(249,101,163,0.25);
    }

    .card-dark {
      background: var(--card-dark);
      color: #f1f3ff;
      border: none;
      box-shadow: 0 25px 60px rgba(6,8,18,0.5);
    }

    .card-dark h2 { color: rgba(255,255,255,0.6); }
    .card-dark pre {
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      color: #f1f3ff;
    }

    .empty {
      color: var(--text-soft);
      font-size: .95rem;
    }

    @media (max-width: 992px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .card, .hero {
        padding: 1.6rem;
      }
      .metrics {
        grid-template-columns: repeat(auto-fit, minmax(140px,1fr));
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="hero-top">
        <a href="{{ url_for('home') }}">← Voltar</a>
        {% if detailed.doc_type %}
          <span class="tag">{{ detailed.doc_type }}</span>
        {% endif %}
      </div>
      <h1>{{ file_name }}</h1>
    </div>

    <div class="grid">
      <div>
        <div class="card">
          <h2>Scanner estrutural</h2>
          {% if detailed %}
            <div class="metrics">
              <div class="chip">
                <span>Idioma</span>
                <strong>{{ detailed.language_guess or "—" }}</strong>
              </div>
              <div class="chip">
                <span>Palavras</span>
                <strong>{{ detailed.n_words or 0 }}</strong>
              </div>
              <div class="chip">
                <span>Parágrafos</span>
                <strong>{{ detailed.n_paragraphs or 0 }}</strong>
              </div>
              <div class="chip">
                <span>Bullets</span>
                <strong>{{ detailed.n_bullets or 0 }}</strong>
              </div>
            </div>

            <div class="section">
              <h3>Título detectado</h3>
              <div>{{ detailed.title or "—" }}</div>
            </div>

            <div class="section">
              <h3>Headings (amostra)</h3>
              {% if detailed.headings_sample %}
                <ul class="muted" style="padding-left:1.2rem;">
                  {% for heading in detailed.headings_sample.split('|')[:8] %}
                    <li style="margin-bottom:.25rem;">{{ heading.strip() }}</li>
                  {% endfor %}
                </ul>
              {% else %}
                <div class="muted">Nenhum heading detectado.</div>
              {% endif %}
            </div>

            <div class="section">
              <h3>Palavras mais frequentes</h3>
              <div class="muted">{{ detailed.top_words or "—" }}</div>
            </div>

            <div class="section">
              <h3>Resumo heurístico</h3>
              <pre>{{ detailed.resumo_rapido or "—" }}</pre>
            </div>
          {% else %}
            <p class="empty">Nenhuma análise estrutural encontrada. Execute <code>scanner_docs.py</code>.</p>
          {% endif %}
        </div>

        <div class="card mt-4">
          <h2>Métricas agrupadas</h2>
          {% if cat_metrics or num_metrics %}
            <div class="row g-3">
              <div class="col-md-6">
                <div class="section" style="border-top:none;">
                  <h3>Categóricas</h3>
                  {% if cat_metrics %}
                    <div class="d-flex flex-column gap-2">
                      {% for m in cat_metrics %}
                        <div>
                          <div class="small text-uppercase text-secondary fw-semibold mb-1">{{ m.label }}</div>
                          {% if m.is_list %}
                            <div class="d-flex flex-wrap gap-1">
                              {% for v in m.value %}
                                <span class="badge bg-secondary-subtle text-dark border border-secondary">{{ v }}</span>
                              {% endfor %}
                            </div>
                          {% else %}
                            <div>{{ m.value }}</div>
                          {% endif %}
                        </div>
                      {% endfor %}
                    </div>
                  {% else %}
                    <div class="muted">Nenhuma métrica categórica.</div>
                  {% endif %}
                </div>
              </div>
              <div class="col-md-6">
                <div class="section" style="border-top:none;">
                  <h3>Numéricas</h3>
                  {% if num_metrics %}
                    <div class="d-grid gap-2">
                      {% for m in num_metrics %}
                        <div class="chip">
                          <span>{{ m.label }}</span>
                          <strong>{{ m.value }}</strong>
                        </div>
                      {% endfor %}
                    </div>
                  {% else %}
                    <div class="muted">Nenhuma métrica numérica.</div>
                  {% endif %}
                </div>
              </div>
            </div>
          {% else %}
            <p class="empty">Nenhuma métrica agrupada disponível.</p>
          {% endif %}
        </div>
      </div>

      <div class="d-flex flex-column gap-4">
        <div class="card card-alt">
          <h2>IA Local</h2>
          {% if ia %}
            <div class="section">
              <h3>Tipo (IA)</h3>
              <div>{{ ia.tipo_documento or "—" }}</div>
            </div>
            <div class="section">
              <h3>Resumo</h3>
              <pre>{{ ia.resumo or "—" }}</pre>
            </div>
            <div class="section">
              <h3>Palavras-chave</h3>
              <div class="muted">{{ ia.palavras_chave or "—" }}</div>
            </div>
            <div class="section">
              <h3>Entidades</h3>
              <div class="muted">{{ ia.entidades or "—" }}</div>
            </div>
            <div class="section">
              <h3>Datas importantes</h3>
              <div class="muted">{{ ia.datas_importantes or "—" }}</div>
            </div>
            <div class="section">
              <h3>Valores monetários</h3>
              <div class="muted">{{ ia.valores_monetarios or "—" }}</div>
            </div>
            <div class="section">
              <h3>Riscos / observações</h3>
              <pre>{{ ia.riscos or ia.observacoes or "—" }}</pre>
            </div>
          {% else %}
            <p class="empty mb-2">Nenhum JSON de IA encontrado em <code>saida/ia_local/</code>.</p>
            <p class="muted mb-0">Rode <code>ia_local_analise.py</code> para habilitar este bloco.</p>
          {% endif %}
        </div>

        <div class="card card-dark flex-grow-1">
          <h2>JSON bruto</h2>
          {% if ia_raw %}
            <pre>{{ ia_raw }}</pre>
          {% else %}
            <p class="empty mb-0" style="color: rgba(255,255,255,0.7);">Nenhuma resposta crua encontrada.</p>
          {% endif %}
        </div>
      </div>
    </div>
  </div>
  <script>
    (function(){
      const key = 'theme';
      const body = document.body;
      const btn = document.getElementById('themeToggleStatus');
      function setTheme(v){
        if(v === 'dark') body.classList.add('theme-dark'); else body.classList.remove('theme-dark');
        localStorage.setItem(key, v);
      }
      const saved = localStorage.getItem(key);
      if(saved){ setTheme(saved); }
      if(btn){
        btn.addEventListener('click', ()=>{
          const next = body.classList.contains('theme-dark') ? 'light' : 'dark';
          setTheme(next);
        });
      }
    })();
  </script>
</body>
</html>
"""

IA_STATUS_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>Status IA</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {
      --bg: #f4f6fb;
      --text: #0b1527;
      --text-strong: #0b1527;
      --muted: #4b5563;
      --card: #ffffff;
      --hero: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(56,189,248,0.12));
      --hero-border: rgba(15,23,42,0.08);
      --table-bg: transparent;
      --table-striped: rgba(15,23,42,0.03);
      --table-hover: rgba(15,23,42,0.06);
      --table-head: #0f172a;
      --table-head-text: #ffffff;
      --pill: rgba(15,23,42,0.05);
      --pill-text: #0b1527;
      --border: rgba(15,23,42,0.08);
    }
    body.theme-dark {
      --bg: #0a0f1a;
      --text: #fdfdfd;
      --text-strong: #ffffff;
      --muted: #e5eaf1;
      --card: #0f172a;
      --hero: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(56,189,248,0.2));
      --hero-border: rgba(255,255,255,0.12);
      --table-bg: transparent;
      --table-striped: rgba(255,255,255,0.04);
      --table-hover: rgba(255,255,255,0.08);
      --table-head: #111827;
      --table-head-text: #fdfdfd;
      --pill: rgba(255,255,255,0.14);
      --pill-text: #fdfdfd;
      --border: rgba(255,255,255,0.16);
    }
    body { background: var(--bg); color: var(--text); }
    h1,h2,h3,h4,h5,h6 { color: var(--text-strong); }
    .shell { max-width: 1100px; margin: 0 auto; padding: 2.5rem 1.4rem 3rem; }
    .hero { background: var(--hero); border: 1px solid var(--hero-border); border-radius: 28px; padding: 1.4rem 1.6rem; box-shadow: 0 24px 60px rgba(0,0,0,0.18); color: var(--text-strong); }
    .hero h1 { margin: 0; font-size: 1.4rem; }
    .hero small { color: var(--muted); }
    .card-dark { background: var(--card); border: 1px solid var(--border); border-radius: 20px; color: var(--text); }
    .badge-soft { background: var(--pill); border: 1px solid var(--border); color: var(--pill-text); }
    .table-dark { --bs-table-bg: var(--table-bg); --bs-table-striped-bg: var(--table-striped); --bs-table-hover-bg: var(--table-hover); color: var(--text); }
    .table-dark td, .table-dark th { color: var(--text); }
    .filters input, .filters select { background: rgba(255,255,255,0.9); border: 1px solid var(--border); color: var(--text); }
    body.theme-dark .filters input, body.theme-dark .filters select { background: rgba(255,255,255,0.08); }
    .filters input::placeholder { color: var(--muted); }
    .filters label { color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.12em; }
    a { color: #3b82f6; }
    pre { background: rgba(15,23,42,0.05); border: 1px solid var(--border); color: var(--text); border-radius: 12px; padding: .75rem; }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero mb-3 d-flex justify-content-between align-items-center">
      <div>
        <h1>Status das análises IA</h1>
        <small>Monitoramento de jobs, modelos e histórico</small>
      </div>
      <div class="d-flex gap-2">
        <a class="btn btn-outline-secondary btn-sm" href="{{ url_for('home') }}">Voltar</a>
        <a class="btn btn-warning btn-sm text-dark" href="{{ url_for('treino') }}">Painel de Treino</a>
        <button id="themeToggleStatus" class="btn btn-outline-secondary btn-sm">Tema</button>
      </div>
    </div>

    <form method="get" class="row g-3 mb-3 filters">
      <div class="col-md-4">
        <label class="form-label">Buscar</label>
        <input type="text" name="q" class="form-control" value="{{ filters.q }}" placeholder="Arquivo ou tipo">
      </div>
      <div class="col-md-3">
        <label class="form-label">Modelo</label>
        <select name="model" class="form-select">
          <option value="">Todos</option>
          <option value="analysis" {% if filters.model=='analysis' %}selected{% endif %}>JSON only</option>
          <option value="general" {% if filters.model=='general' %}selected{% endif %}>Geral</option>
        </select>
      </div>
      <div class="col-md-3">
        <label class="form-label">Status</label>
        <select name="status" class="form-select">
          <option value="">Todos</option>
          <option value="pending" {% if filters.status=='pending' %}selected{% endif %}>Pendentes</option>
          <option value="done" {% if filters.status=='done' %}selected{% endif %}>Concluídos</option>
        </select>
      </div>
      <div class="col-md-2 d-flex align-items-end">
        <button class="btn btn-primary w-100">Filtrar</button>
      </div>
    </form>

    <div class="card-dark p-3 mb-3">
      <div class="table-responsive">
        <table class="table table-dark table-hover align-middle mb-0">
          <thead>
            <tr>
              <th>Arquivo</th>
              <th>Modelo</th>
              <th>Atualizado</th>
              <th>Tipo</th>
              <th>Status</th>
              <th class="text-end">Ações</th>
            </tr>
          </thead>
          <tbody>
            {% for e in entries %}
              <tr>
                <td class="fw-semibold">{{ e.file }}</td>
                <td><span class="badge badge-soft">{{ e.model }}</span></td>
                <td class="small text-muted">{{ e.updated }}</td>
                <td>{{ e.doc_type or "—" }}</td>
                <td>
                  {% if e.pending %}
                    <span class="badge bg-warning text-dark">Pendente</span>
                  {% else %}
                    <span class="badge bg-success">Concluído</span>
                  {% endif %}
                </td>
                <td class="text-end">
                  <form class="d-inline" method="post" action="{{ url_for('reprocess_ia', file_name=e.file) }}">
                    <input type="hidden" name="mode" value="analysis">
                    <button class="btn btn-outline-info btn-sm">Refazer (JSON)</button>
                  </form>
                  <form class="d-inline" method="post" action="{{ url_for('reprocess_ia', file_name=e.file) }}">
                    <input type="hidden" name="mode" value="general">
                    <button class="btn btn-outline-secondary btn-sm">Refazer (Geral)</button>
                  </form>
                  <a class="btn btn-outline-light btn-sm" href="{{ url_for('ver', file_name=e.file) }}">Ver</a>
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card-dark p-3">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <div>
          <div class="text-uppercase small text-secondary">Histórico de pipeline</div>
          <div class="fw-semibold">Últimas execuções</div>
        </div>
      </div>
      {% if history %}
        <div class="table-responsive">
          <table class="table table-dark table-sm align-middle mb-0">
            <thead>
              <tr>
                <th>Nome</th>
                <th>Comando</th>
                <th>Status</th>
                <th>Início</th>
                <th>Saída</th>
              </tr>
            </thead>
            <tbody>
              {% for h in history %}
                <tr>
                  <td>{{ h.name }}</td>
                  <td class="small text-muted"><code>{{ h.command }}</code></td>
                  <td>
                    <span class="badge bg-{{ 'success' if h.status=='ok' else 'danger' if h.status=='error' else 'warning text-dark' }}">{{ h.status }}</span>
                  </td>
                  <td class="small text-muted">{{ h.started }}</td>
                  <td class="small text-muted text-truncate" style="max-width:300px;">{{ h.output or "—" }}</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      {% else %}
        <p class="text-muted mb-0">Nenhuma execução registrada.</p>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""

TRAIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>Painel de Treinamento</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body class="bg-light">
  <div class="container py-4">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <h1 class="h4 mb-1">Painel de Treinamento e Métricas</h1>
        <div class="text-muted small">Dataset · LoRA · Conversão GGUF · Registro Ollama</div>
      </div>
      <div class="d-flex gap-2">
        <a class="btn btn-outline-secondary btn-sm" href="{{ url_for('home') }}">Voltar</a>
        <a class="btn btn-secondary btn-sm" href="{{ url_for('ia_status') }}">Status IA</a>
      </div>
    </div>

    <div class="row g-3 mb-3">
      <div class="col-md-3">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="text-muted text-uppercase small">Dataset</div>
            <div class="fw-bold fs-4">{{ metrics.dataset_count }}</div>
            <div class="small text-muted">exemplos ({{ metrics.dataset_size }})</div>
            <div class="small text-muted mt-1">Atualizado: {{ metrics.dataset_mtime }}</div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="text-muted text-uppercase small">IA JSON</div>
            <div class="fw-bold fs-4">{{ metrics.ia_files }}</div>
            <div class="small text-muted">arquivos processados</div>
            <div class="small text-muted mt-1">Pendentes: {{ metrics.pending }}</div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="text-muted text-uppercase small">Modelos Ollama</div>
            <div class="fw-bold fs-5">{{ metrics.ollama_models|join(", ") }}</div>
            <div class="small text-muted">em execução</div>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="text-muted text-uppercase small">LoRA / GGUF</div>
            <div class="fw-bold fs-6">{{ metrics.lora_dir }}</div>
            <div class="small text-muted">GGUF: {{ metrics.gguf_path }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="row g-3 mb-3">
      <div class="col-md-6">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <div>
                <div class="text-muted text-uppercase small">Distribuição IA</div>
                <div class="fw-semibold">Processados vs pendentes</div>
              </div>
            </div>
            <canvas id="chartIa" height="180"></canvas>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card shadow-sm h-100">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <div>
                <div class="text-muted text-uppercase small">Dataset</div>
                <div class="fw-semibold">Tamanho x exemplos</div>
              </div>
            </div>
            <canvas id="chartDataset" height="180"></canvas>
          </div>
        </div>
      </div>
    </div>

    <form class="card mb-3 shadow-sm" method="post" action="{{ url_for('ia_pipeline_run') }}">
      <div class="card-body">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <div>
            <div class="text-muted text-uppercase small">Ações rápidas</div>
            <div class="fw-semibold">Pipeline fim-a-fim</div>
          </div>
          <div class="d-flex flex-wrap gap-2">
            <button name="action" value="dataset" class="btn btn-outline-primary btn-sm">Gerar dataset</button>
            <button name="action" value="train" class="btn btn-primary btn-sm">Treinar LoRA</button>
            <button name="action" value="convert" class="btn btn-outline-secondary btn-sm">Converter GGUF</button>
            <button name="action" value="register" class="btn btn-outline-dark btn-sm">Registrar no Ollama</button>
          </div>
        </div>
        <p class="small text-muted mb-0">As ações rodam em background e aparecem no histórico abaixo.</p>
      </div>
    </form>

    <div class="card shadow-sm">
      <div class="card-body">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <div>
            <div class="text-muted text-uppercase small">Histórico</div>
            <div class="fw-semibold">Últimas execuções</div>
          </div>
          <span class="badge bg-secondary">máx 15</span>
        </div>
        {% if history %}
          <div class="table-responsive">
            <table class="table table-sm align-middle">
              <thead>
                <tr>
                  <th>Nome</th>
                  <th>Comando</th>
                  <th>Início</th>
                  <th>Status</th>
                  <th>Saída</th>
                </tr>
              </thead>
              <tbody>
                {% for h in history[:15] %}
                  <tr>
                    <td>{{ h.name }}</td>
                    <td><code>{{ h.command }}</code></td>
                    <td class="small text-muted">{{ h.started }}</td>
                    <td>{{ h.status }}</td>
                    <td class="small text-muted text-truncate" style="max-width:260px;">{{ h.output or "—" }}</td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        {% else %}
          <p class="text-muted mb-0">Nenhuma execução registrada ainda.</p>
        {% endif %}
      </div>
    </div>
  </div>
  <script>
    const metrics = {{ metrics|tojson }};
    // IA pie
    const ctxIa = document.getElementById('chartIa');
    if (ctxIa) {
      const processed = metrics.processed || 0;
      const pending = metrics.pending || 0;
      const total = processed + pending || 1;
      new Chart(ctxIa, {
        type: 'doughnut',
        data: {
          labels: ['Processados', 'Pendentes'],
          datasets: [{
            data: [processed, pending],
            backgroundColor: ['#0d6efd', '#ffc107'],
            borderWidth: 0
          }]
        },
        options: {
          plugins: {
            legend: { position: 'bottom' }
          }
        }
      });
    }
    // Dataset bar
    const ctxDs = document.getElementById('chartDataset');
    if (ctxDs) {
      const count = metrics.dataset_count || 0;
      const sizeLabel = metrics.dataset_size || '0 B';
      new Chart(ctxDs, {
        type: 'bar',
        data: {
          labels: ['Exemplos'],
          datasets: [{
            label: 'Qtd. exemplos',
            data: [count],
            backgroundColor: '#6610f2'
          }]
        },
        options: {
          plugins: { legend: { display: false }, tooltip: { callbacks: { footer: () => `Tamanho: ${sizeLabel}` } } },
          scales: {
            y: { beginAtZero: true, ticks: { stepSize: 1 } }
          }
        }
      });
    }
  </script>
</body>
</html>
"""

# ==========================
# Rotas
# ==========================

@app.route("/")
def home():
    docs = []
    total_docs = 0
    type_counts = Counter()
    ext_counts = Counter()
    total_size = 0

    if INDEX_JSON.exists():
        try:
            data = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
            for d in data:
                docs.append(type("DocObj", (), d))
                t = (d.get("doc_type") or "desconhecido")
                e = (d.get("file_type_ext") or "—")
                type_counts[t] += 1
                ext_counts[e] += 1
                total_size += d.get("file_size_bytes", 0) or 0
            total_docs = len(docs)
        except Exception:
            docs = []

    avg_size_kb = round((total_size / 1024 / total_docs), 1) if total_docs else 0.0

    return render_template_string(
        HOME_TEMPLATE,
        docs=docs,
        total_docs=total_docs,
        type_counts=type_counts,
        ext_counts=ext_counts,
        avg_size_kb=avg_size_kb,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    if not files:
        return redirect(url_for("home"))

    for f in files:
        save_path = UPLOAD_DIR / f.filename
        f.save(save_path)

    # Atualiza saídas do scanner e da IA local automaticamente
    subprocess.call(["python3", "scanner_docs.py", "--auto"])
    subprocess.call(["python3", "ia_local_analise.py", "--auto"])

    for f in files:
        submit_ia_task(f.filename, model=OLLAMA_MODEL_ANALYSIS)

    return redirect(url_for("home"))


@app.route("/delete/<file_name>", methods=["POST"])
def delete_file(file_name):
    target = UPLOAD_DIR / file_name
    removed = False
    if target.exists():
        try:
            target.unlink()
            removed = True
        except Exception:
            pass

    # Remove artefatos associados
    text_file = TEXT_DIR / f"{file_name}.txt"
    ia_json = IA_DIR / f"{file_name}.json"
    ia_raw = IA_DIR / f"{file_name}.json.raw.txt"
    for path in [text_file, ia_json, ia_raw]:
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    # Atualiza saídas e IA local apenas se removemos algo
    if removed:
        subprocess.call(["python3", "scanner_docs.py", "--auto"])
        subprocess.call(["python3", "ia_local_analise.py", "--auto"])

    return redirect(url_for("home"))


@app.route("/ver/<file_name>")
def ver(file_name):
    detailed = {}
    ia = {}
    ia_raw = ""
    cat_metrics = []
    num_metrics = []

    if ANALISE_JSON.exists():
        try:
            analises = json.loads(ANALISE_JSON.read_text(encoding="utf-8"))
            for a in analises:
                if a.get("file_name") == file_name:
                    detailed = a
                    break
        except Exception:
            detailed = {}

    candidates = [file_name, f"{file_name}.txt"]
    ia_file = None
    for name in candidates:
        candidate = IA_DIR / f"{name}.json"
        if candidate.exists():
            ia_file = candidate
            break

    if ia_file and ia_file.exists():
        try:
            ia = json.loads(ia_file.read_text(encoding="utf-8"))
            ia_raw = json.dumps(ia, ensure_ascii=False, indent=2)
        except Exception:
            ia = {}
            ia_raw = ia_file.read_text(encoding="utf-8", errors="ignore")
    if not ia:
        ia, ia_raw = ensure_ia_analysis(file_name, model=OLLAMA_MODEL_ANALYSIS)

    cat_metrics, num_metrics = split_metrics_for_view(detailed, ia)

    return render_template_string(
        DETAIL_TEMPLATE,
        file_name=file_name,
        detailed=detailed,
        ia=ia,
        ia_raw=ia_raw,
        cat_metrics=cat_metrics,
        num_metrics=num_metrics,
    )


@app.route("/ia/status")
def ia_status():
    entries = list_ia_status()
    q = request.args.get("q", "").lower()
    model_filter = request.args.get("model", "")
    status_filter = request.args.get("status", "")

    filtered = []
    for e in entries:
        if q and q not in e["file"].lower():
            continue
        if model_filter == "analysis" and e["model"] != OLLAMA_MODEL_ANALYSIS:
            continue
        if model_filter == "general" and e["model"] != OLLAMA_MODEL_GENERAL:
            continue
        if status_filter == "pending" and not e["pending"]:
            continue
        if status_filter == "done" and e["pending"]:
            continue
        filtered.append(e)

    with pipeline_lock:
        history = list(pipeline_history)

    filters = {"q": request.args.get("q", ""), "model": model_filter, "status": status_filter}
    return render_template_string(
        IA_STATUS_TEMPLATE,
        entries=filtered,
        filters=filters,
        history=history,
    )


@app.route("/ia/reprocess/<file_name>", methods=["POST"])
def reprocess_ia(file_name):
    mode = request.form.get("mode")
    model = OLLAMA_MODEL_ANALYSIS if mode != "general" else OLLAMA_MODEL_GENERAL
    ensure_ia_analysis(file_name, model=model)
    return redirect(url_for('ia_status'))


@app.route("/treino")
def treino():
    metrics = gather_training_metrics()
    with pipeline_lock:
        history = list(pipeline_history)
    return render_template_string(
        TRAIN_TEMPLATE,
        metrics=metrics,
        history=history,
    )


@app.route("/ia/pipeline/run", methods=["POST"])
def ia_pipeline_run():
    action = request.form.get("action")
    if action == "dataset":
        run_background_action("Dataset", ["python3", "auto_learning_pipeline.py"])
    elif action == "train":
        run_background_action(
            "Treino LoRA",
            [
                "python3",
                "scripts/treino_lora.py",
                "--dataset",
                str(DATASET_JSONL),
                "--base-model",
                DEFAULT_BASE_MODEL,
                "--output",
                "models/minha-lora",
                "--max-length",
                "1024",
            ],
        )
    elif action == "convert":
        run_background_action("Converter GGUF", ["python3", "convert_lora_to_gguf.py"])
    elif action == "register":
        run_background_action("Registrar Ollama", ["ollama", "create", "minha-lora-docs", "-f", "Modelfile"])
    return redirect(url_for('ia_status'))


@app.route("/health/ia")
def health_ia():
    """Ping simples no Ollama e nos arquivos críticos."""
    status = {"ollama": "down", "models": [], "dataset": "absent", "ia_dir": IA_DIR.exists()}
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        data = resp.json()
        status["ollama"] = "up"
        status["models"] = [m.get("name") for m in data.get("models", [])]
    except Exception:
        status["ollama"] = "down"
    status["dataset"] = "present" if DATASET_JSONL.exists() else "absent"
    return status


@app.route("/ia/edit/<file_name>", methods=["GET", "POST"])
def edit_ia(file_name):
    safe_name = Path(file_name).name
    ia_path = IA_DIR / f"{safe_name}.json"
    if request.method == "POST":
        content = request.form.get("ia_json", "")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            parsed.setdefault("_model", parsed.get("_model", OLLAMA_MODEL_ANALYSIS))
            parsed.setdefault("_timestamp", datetime.utcnow().isoformat())
            ia_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            return redirect(url_for('ia_status'))
    current = ""
    if ia_path.exists():
        current = ia_path.read_text(encoding="utf-8")
    template = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8">
      <title>Editar IA</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
      <div class="container py-4">
        <h1 class="h4 mb-3">Editar análise IA · {{ file_name }}</h1>
        <form method="post">
          <div class="mb-3">
            <textarea name="ia_json" class="form-control" rows="18">{{ current }}</textarea>
          </div>
          <div class="d-flex gap-2">
            <button class="btn btn-primary">Salvar</button>
            <a href="{{ url_for('ia_status') }}" class="btn btn-secondary">Voltar</a>
          </div>
        </form>
      </div>
    </body>
    </html>
    """
    return render_template_string(template, file_name=file_name, current=current)

# ==========================
# Main
# ==========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5050")))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  app.run(host=args.host, port=args.port, debug=args.debug)
