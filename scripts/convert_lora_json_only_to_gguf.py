import os
import subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================================================
# CONFIGURAÇÕES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_MODEL_DIR = BASE_DIR / "models" / "base-tinyllama"
LORA_DIR = BASE_DIR / "models" / "minha-lora-json-only"
MERGED_HF_DIR = BASE_DIR / "models" / "merged-hf-json-only"
OUT_GGUF = BASE_DIR / "models" / "minha-lora-json-only-merged.gguf"

LLAMACPP_CONVERTER = BASE_DIR / "llama.cpp" / "convert_hf_to_gguf.py"

# =========================================================
print("🔍 Verificando diretórios...")

if not BASE_MODEL_DIR.exists():
    raise FileNotFoundError(f"Modelo base não encontrado em: {BASE_MODEL_DIR}")

if not LORA_DIR.exists():
    raise FileNotFoundError(f"Pesos LoRA não encontrados em: {LORA_DIR}")

# Criar pasta do modelo mesclado
MERGED_HF_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CARREGAR MODELO BASE + LORA
# =========================================================
print("📥 Carregando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    device_map="cpu",
)

print("📥 Carregando pesos LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("🔗 Mesclando LoRA → Modelo base...")
model = model.merge_and_unload()

print(f"💾 Salvando modelo mesclado em {MERGED_HF_DIR}...")
model.save_pretrained(MERGED_HF_DIR)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
tokenizer.save_pretrained(MERGED_HF_DIR)


# =========================================================
# CONVERTER PARA GGUF
# =========================================================
print("⚙️ Convertendo para GGUF...")
cmd = [
    "python3",
    str(LLAMACPP_CONVERTER),
    str(MERGED_HF_DIR),
    "--outfile",
    str(OUT_GGUF),
    "--outtype",
    "q8_0"
]

print("🔧 Rodando comando:")
print(" ".join(cmd))

subprocess.run(cmd, check=True)

print(f"🎉 GGUF gerado com sucesso em {OUT_GGUF}!")

