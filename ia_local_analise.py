#!/usr/bin/env python3
"""
Camada 2 - Análise com IA local sobre os textos em ./saida/texto_bruto/

- Lê todos os arquivos .txt gerados pelo scanner (texto bruto dos documentos)
- Envia para um modelo local (ex.: Ollama em http://localhost:11434)
- Pede um JSON estruturado com análise semântica:
    - tipo_documento
    - resumo
    - palavras_chave
    - entidades
    - datas_importantes
    - valores_monetarios
    - riscos
    - observacoes
- Salva:
    - saida/ia_local/<nome_arquivo>.json  (resposta estruturada)
    - saida/ia_local/<nome_arquivo>.txt   (resposta em texto bruto, se quiser)
"""

import os
import sys
import json
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# =====================================
# Dependência: requests (para chamar a IA local via HTTP)
# =====================================

def ensure_requests():
    try:
        import requests  # noqa
        return
    except ImportError:
        print("⚠ Biblioteca 'requests' não encontrada. Tentando instalar automaticamente...")
        cmd = [sys.executable, "-m", "pip", "install", "requests"]
        print("➡", " ".join(cmd))
        try:
            subprocess.check_call(cmd)
            print("✅ 'requests' instalada com sucesso.\n")
        except Exception:
            print("❌ Não foi possível instalar 'requests' automaticamente.")
            print("   Instale manualmente com:\n")
            print("   pip install requests\n")
            sys.exit(1)

ensure_requests()
import requests  # agora deve existir


# =====================================
# Configurações
# =====================================

# URL do Ollama (aceita a base ou a URL completa /api/generate)
_OLLAMA_BASE = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_URL = (
    _OLLAMA_BASE
    if "/api/generate" in _OLLAMA_BASE
    else f"{_OLLAMA_BASE.rstrip('/')}/api/generate"
)

# Modelo padrão: mesma variável usada pelo app.py
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL_ANALYSIS", "llama3.1:8b")

# Limite de caracteres enviados (para não explodir contexto)
MAX_CHARS = 20000

SAIDA_DIR = Path("./saida")
TEXT_DIR = SAIDA_DIR / "texto_bruto"
IA_DIR = SAIDA_DIR / "ia_local"
TRAIN_DIR = SAIDA_DIR / "treinamento"
MODEL_CONFIG = TRAIN_DIR / "model_config.json"


# =====================================
# Função para chamar a IA local
# =====================================

def call_local_llm(model: str, prompt: str, *, silent: bool = False) -> Optional[str]:
    """
    Chama o modelo local via Ollama (ou servidor compatível com /api/generate).
    Retorna o texto da resposta. Se ocorrer falha, retorna None.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        if not silent:
            print("\n❌ Não foi possível conectar em", OLLAMA_URL)
            print("   Verifique se o Ollama (ou outro servidor) está rodando.")
        return None
    except Exception as e:
        if not silent:
            print("\n❌ Erro ao chamar a IA local:", e)
        return None

    if resp.status_code != 200:
        if not silent:
            print("\n❌ Resposta HTTP inesperada da IA local:", resp.status_code, resp.text[:500])
        return None

    data = resp.json()
    # API do Ollama retorna o texto em "response"
    return data.get("response", "")


# =====================================
# Prompt que será enviado para a IA
# =====================================

def build_prompt(doc_text: str, file_name: str) -> str:
    """
    Monta um prompt bem claro para a IA local, pedindo SAÍDA ESTRUTURADA em JSON.
    """
    return f"""
Você é um assistente especializado em análise de documentos.

Analise COMPLETAMENTE o documento abaixo e responda EXCLUSIVAMENTE em JSON válido,
sem nenhum comentário antes ou depois.

Estruture a resposta com os seguintes campos:

{{
  "arquivo": "...",
  "tipo_documento": "...",
  "idioma": "...",
  "resumo": "...",
  "palavras_chave": ["...", "..."],
  "entidades": ["...", "..."],
  "datas_importantes": ["...", "..."],
  "valores_monetarios": ["...", "..."],
  "riscos": ["...", "..."],
  "observacoes": "..."
}}

Regras importantes:
- "tipo_documento": por exemplo "curriculo", "contrato", "nota_fiscal", "relatorio_tecnico",
  "artigo_academico", "email", "apresentacao", "documento_juridico", "desconhecido", etc.
- "idioma": "pt", "en" ou outro código simples.
- "resumo": um resumo conciso com até 10-15 linhas.
- "palavras_chave": termos relevantes do documento (entre 5 e 20).
- "entidades": nomes de pessoas, empresas, instituições, produtos, projetos etc.
- "datas_importantes": datas relevantes que aparecem no texto, em formato string.
- "valores_monetarios": valores em dinheiro mencionados (R$, $, etc.), como strings.
- "riscos": riscos, pontos de atenção ou problemas identificados no conteúdo (se houver).
- "observacoes": qualquer comentário adicional útil sobre o documento.

Documento (arquivo: {file_name}):

\"\"\"text
{doc_text}
\"\"\" end of text
"""


# =====================================
# Loop principal sobre texto_bruto
# =====================================

def check_llm_available(model: str) -> bool:
    """Realiza uma requisição rápida para validar se a IA local está acessível."""
    test = call_local_llm(model, "ping", silent=True)
    return test is not None


def load_model_override(default_model: str) -> str:
    if not MODEL_CONFIG.exists():
        return default_model
    try:
        data = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
        return data.get("ollama_model", default_model) or default_model
    except Exception:
        return default_model


def parse_args():
    parser = argparse.ArgumentParser(description="Análise com IA local sobre texto_bruto.")
    parser.add_argument("--path", help="Caminho para a pasta texto_bruto.")
    parser.add_argument("--model", help="Modelo IA a utilizar. Padrão: %(default)s", default=DEFAULT_MODEL)
    parser.add_argument("--auto", action="store_true", help="Executa sem prompts interativos.")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Processa apenas estes arquivos (nomes ou nomes sem extensão).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    interactive = not (args.auto or args.path or args.model != DEFAULT_MODEL)

    if interactive:
        print("=" * 60)
        print("   CAMADA 2 - Análise com IA Local (Ollama) ")
        print("=" * 60)
        print("\nEste programa lê os arquivos .txt em ./saida/texto_bruto/")
        print("e envia o conteúdo para um modelo local (por exemplo, Ollama)")
        print("para produzir uma ANÁLISE SEMÂNTICA estruturada em JSON.\n")

    if args.path:
        base_text_dir = Path(args.path)
    else:
        if interactive:
            base_text_dir_input = input(
                "Digite o caminho da pasta 'texto_bruto' (ENTER para usar './saida/texto_bruto'): "
            ).strip()
        else:
            base_text_dir_input = ""
        base_text_dir = Path(base_text_dir_input) if base_text_dir_input else TEXT_DIR

    if not base_text_dir.exists():
        print(f"\n❌ Pasta '{base_text_dir}' não encontrada.")
        print("   Rode primeiro o scanner (scanner_docs.py) para gerar os textos brutos.")
        return

    resolved_default_model = load_model_override(DEFAULT_MODEL)
    if args.model and args.model != DEFAULT_MODEL:
        model = args.model
    else:
        if interactive:
            model = input(f"Modelo local a usar (ENTER para '{resolved_default_model}'): ").strip() or resolved_default_model
        else:
            model = resolved_default_model

    print(f"\n📌 Usando modelo local: {model}")
    print(f"📂 Lendo textos de: {base_text_dir.resolve()}")

    out_dir = IA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not check_llm_available(model):
        print("\n⚠ IA local indisponível em", OLLAMA_URL)
        print("   Execução da camada IA foi ignorada automaticamente.")
        return

    txt_files = sorted(glob.glob(str(base_text_dir / "*.txt")))
    if not txt_files:
        print("\n⚠ Nenhum arquivo .txt encontrado em", base_text_dir)
        print("   Rode primeiro o scanner (scanner_docs.py) para gerar texto_bruto.")
        return

    if args.files:
        wanted = {Path(f).name for f in args.files}
        wanted.update({Path(f).stem for f in args.files})
        filtered = []
        for path_str in txt_files:
            p = Path(path_str)
            if p.name in wanted or p.stem in wanted:
                filtered.append(path_str)
        if not filtered:
            print("\n⚠ Nenhum arquivo corresponde aos nomes informados em --files.")
            return
        txt_files = filtered

    print(f"\nEncontrados {len(txt_files)} arquivos para analisar.\n")

    for i, path_str in enumerate(txt_files, start=1):
        path = Path(path_str)
        file_name = path.name
        print(f"[{i}/{len(txt_files)}] Analisando com IA local: {file_name} ...")

        json_out = out_dir / f"{file_name}.json"
        try:
            if json_out.exists() and json_out.stat().st_mtime >= path.stat().st_mtime:
                print("   ℹ Análise já existente e atualizada. Pulando.")
                continue
        except OSError:
            pass

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"   ❌ Erro ao ler {file_name}: {e}")
            continue

        if not text.strip():
            print("   ⚠ Arquivo vazio, pulando.")
            continue

        # limitar tamanho se o documento for muito grande
        if len(text) > MAX_CHARS:
            text_to_send = text[:MAX_CHARS]
            print(f"   ℹ Documento grande, enviando apenas os primeiros {MAX_CHARS} caracteres.")
        else:
            text_to_send = text

        prompt = build_prompt(text_to_send, file_name)
        response_text = call_local_llm(model, prompt)
        if not response_text:
            print("   ⚠ IA local indisponível ou sem resposta válida. Pulando arquivo.")
            continue
        response_text = response_text.strip()

        # Tentamos interpretar como JSON
        json_data = None
        try:
            json_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Tentar corrigir casos em que o modelo adiciona texto extra
            # Vamos tentar extrair o trecho entre a primeira { e a última }
            first = response_text.find("{")
            last = response_text.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    json_data = json.loads(response_text[first:last+1])
                except json.JSONDecodeError:
                    json_data = None

        # Salvar sempre a resposta bruta (para debug)
        raw_out = out_dir / f"{file_name}.raw.txt"
        raw_out.write_text(response_text, encoding="utf-8")

        if json_data is None:
            print("   ⚠ Não foi possível interpretar a resposta como JSON válido.")
            print("     A resposta bruta foi salva em:", raw_out)
            continue

        with json_out.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print("   ✅ Análise JSON salva em:", json_out)

    print("\n🎉 Fim da análise com IA local!")
    print("Veja os arquivos gerados em:", out_dir.resolve())
    print(" - *.json  → análises estruturadas por documento")
    print(" - *.raw.txt → resposta bruta da IA (para debug, se precisar)\n")


if __name__ == "__main__":
    main()
