"""
Camada de linguagem natural baseada em spaCy.

Oferece três coisas para o resto do projeto:
    detect_language(text)            -> "pt", "en", "es" ou "desconhecido"
    normalize_for_search(texts)      -> textos lematizados, sem stopwords
    extract_entities(text)           -> pessoas, organizações e locais

Tudo é opcional: se o spaCy ou um modelo não estiver instalado, as
funções caem num modo simples (regex) e o app continua funcionando.

Modelos usados (instale os idiomas que precisar):
    python -m spacy download pt_core_news_sm
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("nlp")

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as _SW_EN
    from spacy.lang.es.stop_words import STOP_WORDS as _SW_ES
    from spacy.lang.pt.stop_words import STOP_WORDS as _SW_PT

    SPACY_AVAILABLE = True
except ImportError:  # spaCy não instalado: modo simples
    spacy = None
    SPACY_AVAILABLE = False
    _SW_PT = {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "das"}
    _SW_EN = {"the", "of", "and", "to", "in", "is", "that", "for", "it", "with", "as", "was", "on", "be", "by", "this", "are", "or", "an", "from", "at", "which", "not"}
    _SW_ES = {"de", "la", "que", "el", "en", "y", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus"}

# Idioma -> modelo spaCy. Dá para trocar por modelos maiores via env,
# por exemplo NLP_MODEL_PT=pt_core_news_md.
MODELS = {
    "pt": os.getenv("NLP_MODEL_PT", "pt_core_news_sm"),
    "en": os.getenv("NLP_MODEL_EN", "en_core_web_sm"),
    "es": os.getenv("NLP_MODEL_ES", "es_core_news_sm"),
}
DEFAULT_LANG = os.getenv("NLP_DEFAULT_LANG", "pt")

_STOPWORDS = {"pt": set(_SW_PT), "en": set(_SW_EN), "es": set(_SW_ES)}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_load_lock = threading.Lock()

# Rótulos de entidade variam por modelo (PER no pt/es, PERSON no en)
_ENTITY_GROUPS = {
    "PER": "pessoas", "PERSON": "pessoas",
    "ORG": "organizacoes",
    "LOC": "locais", "GPE": "locais",
}


# --------------------------------------------------------------------
# Idioma
# --------------------------------------------------------------------

def detect_language(text: str, sample_chars: int = 5000) -> str:
    """
    Detecta o idioma pela proporção de stopwords de cada língua.
    Rápido, sem dependência extra e bom o bastante para documentos;
    textos muito curtos (menos de ~5 palavras) voltam "desconhecido".
    """
    words = [w.lower() for w in _WORD_RE.findall((text or "")[:sample_chars])]
    if len(words) < 5:
        return "desconhecido"
    scores = {lang: sum(w in sw for w in words) / len(words) for lang, sw in _STOPWORDS.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    # pt e es compartilham muitas stopwords; exige margem mínima
    if ranked[0] < 0.08 or ranked[0] - ranked[1] < 0.02:
        return "desconhecido"
    return best


# --------------------------------------------------------------------
# Carregamento dos modelos
# --------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_nlp(lang: str):
    """
    Retorna o pipeline do idioma, carregado uma única vez.
    Ordem de tentativa: modelo treinado > spacy.blank(lang) > None.
    """
    if not SPACY_AVAILABLE:
        return None
    lang = lang if lang in MODELS else DEFAULT_LANG
    with _load_lock:
        try:
            nlp = spacy.load(MODELS[lang])
            logger.info("NLP: modelo %s carregado", MODELS[lang])
            return nlp
        except OSError:
            logger.warning(
                "NLP: modelo %s não instalado (python -m spacy download %s). "
                "Usando tokenizador básico, sem lemas nem entidades.",
                MODELS[lang], MODELS[lang],
            )
            try:
                return spacy.blank(lang)
            except Exception:
                return None


def has_model(lang: str) -> bool:
    """True se o idioma tem modelo treinado (lemas e entidades)."""
    nlp = get_nlp(lang)
    return bool(nlp is not None and nlp.pipe_names)


def _resolve_lang(text: str, lang: Optional[str], fallback: str) -> str:
    if lang and lang in MODELS:
        return lang
    found = detect_language(text)
    return found if found in MODELS else fallback


# --------------------------------------------------------------------
# Normalização para busca (RAG)
# --------------------------------------------------------------------

def _simple_normalize(text: str, lang: str) -> str:
    sw = _STOPWORDS.get(lang, set())
    return " ".join(w for w in (t.lower() for t in _WORD_RE.findall(text)) if w not in sw and len(w) > 1)


def normalize_for_search(
    texts: Iterable[str],
    langs: Optional[List[Optional[str]]] = None,
    fallback_lang: str = DEFAULT_LANG,
    batch_size: int = 64,
) -> List[str]:
    """
    Converte textos em sequências de lemas minúsculos, sem stopwords,
    pontuação e números. "As multas dos contratos" vira "multa contrato",
    o que faz a busca achar o mesmo termo em flexões diferentes.

    langs: idioma de cada texto (mesmo tamanho de texts). Se ausente,
    o idioma é detectado por texto.
    """
    texts = list(texts)
    langs = list(langs) if langs is not None else [None] * len(texts)
    resolved = [_resolve_lang(t, l, fallback_lang) for t, l in zip(texts, langs)]

    out: List[str] = [""] * len(texts)
    by_lang: Dict[str, List[int]] = defaultdict(list)
    for i, lang in enumerate(resolved):
        by_lang[lang].append(i)

    for lang, idxs in by_lang.items():
        nlp = get_nlp(lang)
        if nlp is None or not nlp.pipe_names:
            for i in idxs:
                out[i] = _simple_normalize(texts[i], lang)
            continue

        # Parser e NER não são necessários para lemas: desligar acelera bastante
        disable = [p for p in ("parser", "ner") if p in nlp.pipe_names]
        docs = nlp.pipe((texts[i] for i in idxs), batch_size=batch_size, disable=disable)
        for i, doc in zip(idxs, docs):
            out[i] = " ".join(
                (tok.lemma_ or tok.text).lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok.text) > 1
            )
    return out


def normalize_query(query: str, fallback_lang: str = DEFAULT_LANG) -> str:
    """Normaliza a pergunta do usuário do mesmo jeito que o corpus."""
    return normalize_for_search([query], fallback_lang=fallback_lang)[0]


# --------------------------------------------------------------------
# Entidades
# --------------------------------------------------------------------

def extract_entities(
    text: str,
    lang: Optional[str] = None,
    max_chars: int = 100_000,
    top_n: int = 15,
) -> Dict[str, List[str]]:
    """
    Extrai pessoas, organizações e locais, ordenados por frequência.
    Retorna listas vazias se não houver modelo treinado para o idioma.
    """
    result: Dict[str, List[str]] = {"pessoas": [], "organizacoes": [], "locais": []}
    if not text:
        return result
    lang = _resolve_lang(text, lang, DEFAULT_LANG)
    nlp = get_nlp(lang)
    if nlp is None or "ner" not in nlp.pipe_names:
        return result

    counters: Dict[str, Counter] = {k: Counter() for k in result}
    display: Dict[str, str] = {}
    # Lemmatizer e parser não são usados aqui
    disable = [p for p in ("parser", "lemmatizer") if p in nlp.pipe_names]
    doc = nlp(text[:max_chars], disable=disable)
    for ent in doc.ents:
        group = _ENTITY_GROUPS.get(ent.label_)
        name = " ".join(ent.text.split())
        if not group or len(name) < 3 or not any(c.isalpha() for c in name):
            continue
        key = name.lower()
        counters[group][key] += 1
        display.setdefault(key, name)

    for group, counter in counters.items():
        result[group] = [display[k] for k, _ in counter.most_common(top_n)]
    return result


def status() -> Dict[str, object]:
    """Resumo do que está disponível, útil para health checks."""
    return {
        "spacy": SPACY_AVAILABLE,
        "version": getattr(spacy, "__version__", None),
        "models": {lang: has_model(lang) for lang in MODELS} if SPACY_AVAILABLE else {},
    }
