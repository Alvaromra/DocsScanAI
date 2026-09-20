"""
RAG local baseado em TF-IDF sobre os textos extraídos pelo scanner.

Lê os .txt de saida/texto_bruto (ou RAG_TEXT_DIR), divide cada documento
em trechos com sobreposição e devolve os trechos mais parecidos com a
pergunta. O índice é reconstruído sozinho quando a pasta muda.

API usada pelo app.py:
    get_context_for_query_sync(query, top_k=3) -> str
    get_context_for_query(query, top_k=3)      -> str  (async)
    rebuild_corpus_sync()                      -> int  (nº de trechos)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_unicode
from sklearn.metrics.pairwise import linear_kernel

try:
    from project.app import nlp as _nlp
except ImportError:  # executado de dentro de project/app
    try:
        import nlp as _nlp  # type: ignore
    except ImportError:
        _nlp = None

# Lematização com spaCy (desligue com RAG_USE_SPACY=0)
USE_SPACY = os.getenv("RAG_USE_SPACY", "1") == "1" and _nlp is not None

logger = logging.getLogger("rag_local")

_REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = Path(os.getenv("RAG_TEXT_DIR", _REPO_ROOT / "saida" / "texto_bruto"))

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MAX_FILES = int(os.getenv("RAG_MAX_FILES", "500"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.08"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3000"))
# Peso do índice de pedaços de palavras na nota final (0 desliga)
CHAR_WEIGHT = float(os.getenv("RAG_CHAR_WEIGHT", "0.5"))

# Stopwords básicas em português (o sklearn só traz a lista em inglês).
# Sem acento, porque o vetorizador remove acentos antes de comparar.
_PT_STOPWORDS_RAW = {
    "a", "ao", "aos", "as", "até", "com", "como", "da", "das", "de", "dela",
    "dele", "do", "dos", "e", "ela", "ele", "em", "entre", "era", "essa",
    "esse", "esta", "este", "eu", "foi", "há", "isso", "isto", "já", "la",
    "lhe", "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na", "nas",
    "nem", "no", "nos", "não", "o", "os", "ou", "para", "pela", "pelas",
    "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se", "sem",
    "ser", "seu", "sua", "são", "também", "te", "tem", "um", "uma", "você",
    "à", "às", "é",
}
PT_STOPWORDS = sorted({strip_accents_unicode(w) for w in _PT_STOPWORDS_RAW})


@dataclass
class _Index:
    signature: Tuple
    vectorizer: Optional[TfidfVectorizer]
    matrix: object
    sources: List[str]
    chunks: List[str]
    lang: str = "pt"          # idioma predominante do corpus
    lemmatized: bool = False  # se o índice foi montado com spaCy
    char_vectorizer: Optional[TfidfVectorizer] = None
    char_matrix: object = None


_lock = threading.Lock()
_index: Optional[_Index] = None


def _list_files() -> List[Path]:
    if not TEXT_DIR.is_dir():
        return []
    return sorted(TEXT_DIR.glob("*.txt"))[:MAX_FILES]


def _signature(files: List[Path]) -> Tuple:
    """Muda sempre que um arquivo é criado, apagado ou alterado."""
    sig = []
    for f in files:
        try:
            st = f.stat()
            sig.append((f.name, st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    return tuple(sig)


def _split(text: str) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + CHUNK_SIZE]
        if len(piece.strip()) > 50:
            chunks.append(piece)
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def _build(files: List[Path], signature: Tuple) -> _Index:
    sources, chunks, chunk_langs = [], [], []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("RAG: falha ao ler %s: %s", f.name, exc)
            continue
        name = f.name[:-4] if f.name.endswith(".txt") else f.name
        # Idioma detectado no documento inteiro: trechos curtos erram mais
        lang = _nlp.detect_language(text) if USE_SPACY else "pt"
        for piece in _split(text):
            sources.append(name)
            chunks.append(piece)
            chunk_langs.append(lang)

    if not chunks:
        logger.info("RAG: corpus vazio em %s", TEXT_DIR)
        return _Index(signature, None, None, [], [])

    known = [l for l in chunk_langs if l != "desconhecido"]
    corpus_lang = Counter(known).most_common(1)[0][0] if known else "pt"

    lemmatized = False
    to_index = chunks
    if USE_SPACY:
        try:
            langs = [l if l != "desconhecido" else corpus_lang for l in chunk_langs]
            to_index = _nlp.normalize_for_search(chunks, langs=langs)
            lemmatized = True
        except Exception as exc:
            logger.warning("RAG: lematização falhou, usando texto puro: %s", exc)
            to_index = chunks

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=PT_STOPWORDS,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    try:
        matrix = vectorizer.fit_transform(to_index)
    except ValueError:  # vocabulário vazio (só stopwords/números)
        return _Index(signature, None, None, [], [])

    # Segundo índice com pedaços de palavras (3 a 5 letras). Cobre o que o
    # lematizador erra ("barras" -> "barro") e flexões em qualquer idioma.
    char_vectorizer, char_matrix = None, None
    if CHAR_WEIGHT > 0:
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True,
            strip_accents="unicode", sublinear_tf=True, min_df=1,
        )
        char_matrix = char_vectorizer.fit_transform(to_index)

    logger.info(
        "RAG: %d trechos de %d arquivos indexados (idioma: %s, lemas: %s)",
        len(chunks), len(files), corpus_lang, "sim" if lemmatized else "não",
    )
    return _Index(signature, vectorizer, matrix, sources, chunks, corpus_lang,
                  lemmatized, char_vectorizer, char_matrix)


def _get_index(force: bool = False) -> _Index:
    global _index
    files = _list_files()
    sig = _signature(files)
    with _lock:
        if force or _index is None or _index.signature != sig:
            _index = _build(files, sig)
        return _index


def rebuild_corpus_sync() -> int:
    """Reconstrói o índice imediatamente. Retorna o número de trechos."""
    return len(_get_index(force=True).chunks)


def search(query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """Retorna [(arquivo, trecho, score)] ordenado por relevância."""
    query = (query or "").strip()
    if not query:
        return []
    idx = _get_index()
    if idx.vectorizer is None:
        return []
    if idx.lemmatized:
        # Perguntas curtas quase nunca têm idioma detectável: usa o do corpus
        query = _nlp.normalize_query(query, fallback_lang=idx.lang) or query
    scores = linear_kernel(idx.vectorizer.transform([query]), idx.matrix).ravel()
    if idx.char_vectorizer is not None:
        char_scores = linear_kernel(idx.char_vectorizer.transform([query]), idx.char_matrix).ravel()
        scores = (1 - CHAR_WEIGHT) * scores + CHAR_WEIGHT * char_scores
    order = scores.argsort()[::-1]

    results, seen = [], set()
    for i in order:
        score = float(scores[i])
        if score < MIN_SCORE or len(results) >= top_k:
            break
        key = (idx.sources[i], idx.chunks[i][:80])
        if key in seen:
            continue
        seen.add(key)
        results.append((idx.sources[i], idx.chunks[i], score))
    return results


def get_context_for_query_sync(query: str, top_k: int = 3) -> str:
    """Contexto pronto para o prompt; string vazia se nada relevante."""
    parts, total = [], 0
    for source, chunk, _ in search(query, top_k=top_k):
        block = f"[{source}]\n{chunk}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


async def get_context_for_query(query: str, top_k: int = 3) -> str:
    """Versão async: roda a busca numa thread para não travar o loop."""
    return await asyncio.to_thread(get_context_for_query_sync, query, top_k)


def _warmup() -> None:
    try:
        _get_index()
    except Exception as exc:  # não derruba o app por causa do RAG
        logger.warning("RAG: aquecimento falhou: %s", exc)


# Monta o índice em segundo plano no import, assim a primeira pergunta
# do chat não estoura o timeout de 1.5 s do rag_search_async.
if os.getenv("RAG_WARMUP", "1") == "1":
    threading.Thread(target=_warmup, name="rag-warmup", daemon=True).start()
