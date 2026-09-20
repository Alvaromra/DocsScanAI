#!/usr/bin/env python3
"""
treino_lora.py

Exemplo simplificado de pipeline de fine-tuning usando LoRA/PEFT para um modelo
carregado via HuggingFace Transformers + PyTorch. Execute somente se tiver
os pacotes instalados e espaço/gpu disponíveis.

Este script lê o dataset gerado por auto_learning_pipeline.py (JSONL com
"prompt"/"response"), aplica LoRA no modelo base especificado e salva os pesos.

Uso:
  python3 scripts/treino_lora.py \
      --dataset saida/treinamento/dataset.jsonl \
      --base-model meta-llama/Llama-2-7b-chat-hf \
      --output models/minha-lora

Dependências sugeridas:
  pip install torch transformers datasets peft
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Caminho para dataset.jsonl gerado pelo pipeline.")
    parser.add_argument("--base-model", required=True, help="Nome do modelo base (HuggingFace).")
    parser.add_argument("--output", required=True, help="Diretório para salvar os pesos LoRA.")
    parser.add_argument("--epochs", type=int, default=1, help="Número de épocas (padrão: 1).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size de treino.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max-length", type=int, default=1024, help="Comprimento máximo de tokens (padrão: 1024).")
    return parser.parse_args()


def format_example(example):
    prompt = example["prompt"]
    response = example["response"]
    text = f"{prompt.strip()}\n### Resposta:\n{response.strip()}"
    return {"text": text}


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset {dataset_path} não encontrado.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(dataset_path))
    ds = ds.map(format_example)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(mod, input_, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    fp16_flag = device == "cuda"
    bf16_flag = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        fp16=fp16_flag,
        bf16=bf16_flag,
        gradient_checkpointing=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[OK] Treino LoRA concluído. Pesos salvos em: {output_dir}")


if __name__ == "__main__":
    main()
