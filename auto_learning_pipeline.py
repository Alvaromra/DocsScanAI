#!/usr/bin/env python3
"""
auto_learning_pipeline.py

Arquitetura básica de autoaprendizado/local fine-tuning.

Fluxo:
1. Reúne textos brutos em ./saida/texto_bruto/
2. Procura análises corrigidas (feedback) em ./saida/feedback/
   - Se não houver feedback, usa as análises da IA atual (./saida/ia_local/)
3. Gera um dataset JSONL em ./saida/treinamento/dataset.jsonl
   com pares (prompt, resposta desejada) pronto para fine-tuning
   (LoRA, PEFT, etc.).

O dataset pode ser usado com ferramentas como:
  - `ollama create` (após converter para o formato suportado)
  - scripts de fine-tuning HuggingFace/PEFT
  - qualquer rotina personalizada (basta ler o JSONL).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).parent
SAIDA_DIR = BASE_DIR / "saida"
TEXT_DIR = SAIDA_DIR / "texto_bruto"
IA_DIR = SAIDA_DIR / "ia_local"
FEEDBACK_DIR = SAIDA_DIR / "feedback"
TRAIN_DIR = SAIDA_DIR / "treinamento"

def ensure_dirs():
    for d in [SAIDA_DIR, TEXT_DIR, IA_DIR, FEEDBACK_DIR, TRAIN_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_text(file_path: Path) -> Optional[str]:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[ERRO] Não consegui ler {file_path}: {exc}")
        return None


def load_analysis(
    stem: str,
    prefer_feedback: bool = True,
    fallback_to_ia: bool = True,
) -> Optional[Dict]:
    """
    stem é o nome do .txt (por ex. 'documento.pdf.txt').
    Preferimos feedback se existir, senão IA local.
    """
    candidates: List[Path] = []
    stem_json = stem + ".json"
    if prefer_feedback:
        candidates.append(FEEDBACK_DIR / stem_json)
    if fallback_to_ia:
        candidates.append(IA_DIR / stem_json)
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[ERRO] Falha ao ler JSON {candidate}: {exc}")
    return None


def build_prompt(doc_text: str, file_name: str) -> str:
    """
    Prompt padrão usado no dataset (pode ser adaptado conforme estratégia de treino).
    """
    return (
        "Você é um analista especializado em documentos. "
        "Leia o conteúdo abaixo e produza um JSON com campos padronizados.\n\n"
        f"Arquivo: {file_name}\n\n"
        f"Conteúdo:\n{doc_text}\n"
    )


def create_dataset(prefer_feedback: bool = True, fallback_to_ia: bool = True) -> Path:
    ensure_dirs()
    out_path = TRAIN_DIR / "dataset.jsonl"
    entries = []

    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print("[AVISO] Nenhum texto bruto encontrado em saida/texto_bruto/.")

    for txt in txt_files:
        text = load_text(txt)
        if not text:
            continue

        analysis = load_analysis(
            txt.name,
            prefer_feedback=prefer_feedback,
            fallback_to_ia=fallback_to_ia,
        )
        if not analysis:
            print(f"[AVISO] Sem análise (feedback ou IA) para {txt.name}. Pulando.")
            continue

        prompt = build_prompt(text, txt.name)
        response = json.dumps(analysis, ensure_ascii=False)
        entries.append({"prompt": prompt, "response": response})

    if not entries:
        print("[AVISO] Nenhuma entrada válida para gerar dataset.")
        out_path.write_text("", encoding="utf-8")
        return out_path

    with out_path.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Dataset com {len(entries)} entradas salvo em: {out_path}")
    print("     Use esse arquivo para treinar/fazer fine-tuning do modelo local.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pipeline básico de autoaprendizado.")
    parser.add_argument(
        "--only-feedback",
        action="store_true",
        help="Usa apenas as análises presentes em saida/feedback (sem fallback para ia_local).",
    )
    args = parser.parse_args()
    prefer_feedback = True
    fallback_to_ia = not args.only_feedback
    create_dataset(prefer_feedback=prefer_feedback, fallback_to_ia=fallback_to_ia)


if __name__ == "__main__":
    main()
