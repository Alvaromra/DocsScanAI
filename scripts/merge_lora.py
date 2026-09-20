#!/usr/bin/env python3
"""
merge_lora.py

Carrega um modelo base HuggingFace + adaptador LoRA/PEFT e gera um modelo
fundido, pronto para ser exportado para GGUF/Ollama.

Exemplo:
  python3 scripts/merge_lora.py \
     --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --lora-path models/minha-lora \
     --output-dir models/minha-lora-merged
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Nome/path do modelo base HuggingFace.")
    parser.add_argument("--lora-path", required=True, help="Diretório com os arquivos LoRA/PEFT (adapter_model.safetensors etc.).")
    parser.add_argument("--output-dir", required=True, help="Diretório de saída para o modelo fundido.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Carregando modelo base {args.base_model} em {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"[INFO] Aplicando LoRA de {args.lora_path} ...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("[INFO] Fundindo pesos LoRA com o modelo base...")
    peft_model = peft_model.merge_and_unload()

    print(f"[INFO] Salvando modelo fundido em {output_dir} ...")
    peft_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("[OK] Modelo fundido salvo. Agora converta para GGUF ou use em outra etapa.")


if __name__ == "__main__":
    main()
