import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess

# Caminhos no seu Mac
BASE_MODEL_DIR = "/Users/alvaromra/Desktop/Script_para_docs/models/base-tinyllama"
LORA_DIR = "/Users/alvaromra/Desktop/Script_para_docs/models/minha-lora"
MERGED_DIR = "/Users/alvaromra/Desktop/Script_para_docs/models/merged-hf"
OUTPUT_GGUF = "/Users/alvaromra/Desktop/Script_para_docs/models/minha-lora-merged.gguf"

# Caminho para o conversor do llama.cpp
CONVERT_SCRIPT = "/Users/alvaromra/Desktop/llama.cpp/convert_hf_to_gguf.py"

def main():
    print("🔍 Verificando diretórios...")
    if not os.path.isdir(BASE_MODEL_DIR):
        raise SystemExit(f"❌ Base model não encontrado: {BASE_MODEL_DIR}")
    if not os.path.isdir(LORA_DIR):
        raise SystemExit(f"❌ LoRA não encontrado: {LORA_DIR}")

    print("📥 Carregando modelo base...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, device_map="cpu")

    print("📥 Carregando pesos LoRA...")
    lora = PeftModel.from_pretrained(base, LORA_DIR)

    print("🔗 Mesclando LoRA → Modelo base...")
    merged = lora.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    print(f"💾 Salvando modelo mesclado em {MERGED_DIR}...")
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print("⚙️ Convertendo para GGUF...")
    cmd = [
        "python3",
        CONVERT_SCRIPT,
        MERGED_DIR,
        "--outfile", OUTPUT_GGUF,
        "--outtype", "q8_0",
    ]

    print("🔧 Rodando comando:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print(f"🎉 GGUF gerado com sucesso em {OUTPUT_GGUF} !")

if __name__ == "__main__":
    main()
