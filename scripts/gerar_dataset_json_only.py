#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera um dataset de fine-tuning para o modelo local (TinyLlama + LoRA)
focado em SEMPRE responder em JSON válido e padronizado.

Saída:
  - Cria/atualiza: saida/treinamento/dataset_json_only.jsonl
  - Contém ~50 exemplos de (prompt, response)

Formato:
  {
    "prompt": "<instruções + texto>",
    "response": "<string JSON, exata>"
  }
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SAIDA_TREINAMENTO = BASE_DIR / "saida" / "treinamento"
SAIDA_TREINAMENTO.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = SAIDA_TREINAMENTO / "dataset_json_only.jsonl"


def make_system_prompt(texto: str, arquivo: str) -> str:
    """
    Monta o prompt com as regras rígidas de saída em JSON.
    """
    return f"""Você é um analista especializado em documentos.

Sua tarefa é SEMPRE responder em JSON válido, com a seguinte estrutura exata:

{{
  "arquivo": "",
  "tipo_documento": "",
  "idioma": "",
  "resumo": "",
  "palavras_chave": [],
  "entidades": [],
  "datas_importantes": [],
  "valores_monetarios": [],
  "riscos": [],
  "observacoes": ""
}}

REGRAS IMPORTANTES:
- Responda APENAS o JSON, sem nenhum texto antes ou depois.
- Não explique o que está fazendo.
- Não use comentários.
- Não use Markdown.
- Se não souber algum campo, deixe-o vazio ("" ou []).
- Nunca mude o nome dos campos do JSON.
- Nunca adicione campos extras.

Agora analise o texto abaixo e responda usando APENAS o JSON acima.

Arquivo: {arquivo}

Texto:
{texto}
"""


def build_examples_base():
    """
    Cria uma lista de exemplos base (diversos tipos de documento).
    Cada item: dict com campos:
      - arquivo
      - texto
      - tipo_documento
      - idioma
      - resumo
      - palavras_chave
      - entidades
      - datas_importantes
      - valores_monetarios
      - riscos
      - observacoes
    """
    return [
        # 1) Declaração de matrícula (pt)
        dict(
            arquivo="declaracao_matricula.txt",
            texto=(
                "DECLARAÇÃO\n\n"
                "Declaramos, para os devidos fins, que ALVARO MARCAL DE ARAUJO é aluno "
                "regularmente matriculado na Universidade de Brasília (UnB), no curso de "
                "Engenharia de Redes de Comunicação, bacharelado diurno.\n\n"
                "Brasília, 25 de outubro de 2025."
            ),
            tipo_documento="declaração de matrícula",
            idioma="pt",
            resumo=(
                "Declaração de que Alvaro Marcal de Araujo é aluno regularmente matriculado "
                "no curso de Engenharia de Redes de Comunicação na UnB."
            ),
            palavras_chave=[
                "declaração",
                "matrícula",
                "Universidade de Brasília",
                "Engenharia de Redes de Comunicação",
            ],
            entidades=[
                "ALVARO MARCAL DE ARAUJO",
                "Universidade de Brasília (UnB)",
            ],
            datas_importantes=["25/10/2025"],
            valores_monetarios=[],
            riscos=[],
            observacoes="Documento acadêmico simples de comprovação de vínculo."
        ),
        # 2) Currículo (pt)
        dict(
            arquivo="curriculo_alvaro.txt",
            texto=(
                "Álvaro Marçal de Araújo\n"
                "Estudante de Engenharia de Redes, Infraestrutura e DevOps\n"
                "Experiência com Linux, Docker, Ansible, Jenkins, OpenShift e CI/CD.\n"
                "Atuação prática com servidores Nginx, Apache e banco de dados MySQL em Docker.\n"
                "Desenvolvedor full-stack (Python, Node.js, React, HTML, CSS, SQL).\n"
            ),
            tipo_documento="curriculo",
            idioma="pt",
            resumo=(
                "Currículo de Álvaro Marçal de Araújo, estudante de Engenharia de Redes com foco "
                "em infraestrutura, DevOps e desenvolvimento full-stack."
            ),
            palavras_chave=[
                "curriculo",
                "Engenharia de Redes",
                "DevOps",
                "Docker",
                "Ansible",
                "CI/CD",
            ],
            entidades=[
                "Álvaro Marçal de Araújo",
                "Universidade de Brasília (UnB)",
            ],
            datas_importantes=[],
            valores_monetarios=[],
            riscos=[],
            observacoes="Resumo profissional voltado para vagas de infraestrutura e automação."
        ),
        # 3) Guia técnico em inglês (VoIP / GoTo)
        dict(
            arquivo="goto_guide_voip.txt",
            texto=(
                "Technical Preparation Guide - Managed Services Engineer\n\n"
                "This document covers SIP signaling, RTP/SRTP media transport, QoS metrics "
                "(latency, jitter, packet loss), and troubleshooting methodologies for VoIP.\n"
                "It also explains how GoTo Connect and Contact Center integrate with customer "
                "networks using firewalls, NAT and REST APIs."
            ),
            tipo_documento="technical_guide",
            idioma="en",
            resumo=(
                "Technical guide for a Managed Services Engineer role, covering SIP, RTP/SRTP, "
                "QoS and VoIP troubleshooting, as well as GoTo platforms and integrations."
            ),
            palavras_chave=[
                "VoIP",
                "SIP",
                "RTP",
                "SRTP",
                "QoS",
                "GoTo Connect",
            ],
            entidades=[
                "GoTo",
                "GoTo Connect",
                "GoTo Contact Center",
            ],
            datas_importantes=[],
            valores_monetarios=[],
            riscos=[
                "VoIP call quality issues due to latency, jitter or packet loss."
            ],
            observacoes="Documento técnico para preparação de entrevista."
        ),
        # 4) Fatura / Boleto simples
        dict(
            arquivo="fatura_internet.txt",
            texto=(
                "FATURA DE SERVIÇO DE INTERNET\n\n"
                "Cliente: João da Silva\n"
                "Plano: Fibra 500 Mbps\n"
                "Período de referência: 01/11/2025 a 30/11/2025\n"
                "Valor do serviço: R$ 129,90\n"
                "Data de vencimento: 10/12/2025\n"
            ),
            tipo_documento="fatura",
            idioma="pt",
            resumo=(
                "Fatura de serviço de internet fibra 500 Mbps para o cliente João da Silva, "
                "com vencimento em 10/12/2025."
            ),
            palavras_chave=[
                "fatura",
                "internet",
                "fibra",
                "R$ 129,90",
            ],
            entidades=[
                "João da Silva",
            ],
            datas_importantes=[
                "01/11/2025",
                "30/11/2025",
                "10/12/2025",
            ],
            valores_monetarios=[
                "R$ 129,90",
            ],
            riscos=[
                "Suspensão do serviço em caso de não pagamento até o vencimento."
            ],
            observacoes=""
        ),
        # 5) E-mail de oferta de emprego (en)
        dict(
            arquivo="job_offer_email.txt",
            texto=(
                "Subject: Job Offer - Junior DevOps Engineer\n\n"
                "Dear Alvaro,\n\n"
                "We are pleased to offer you the position of Junior DevOps Engineer at "
                "TechBridge Solutions, starting on January 15, 2026, with an annual salary "
                "of USD 45,000.\n\n"
                "Please confirm your acceptance by December 20, 2025.\n\n"
                "Best regards,\n"
                "HR Team"
            ),
            tipo_documento="job_offer_email",
            idioma="en",
            resumo=(
                "Email offering Alvaro a Junior DevOps Engineer position at TechBridge Solutions, "
                "with an annual salary of USD 45,000 and start date on January 15, 2026."
            ),
            palavras_chave=[
                "job offer",
                "DevOps Engineer",
                "salary",
            ],
            entidades=[
                "Alvaro",
                "TechBridge Solutions",
            ],
            datas_importantes=[
                "15 January 2026",
                "20 December 2025",
            ],
            valores_monetarios=[
                "USD 45,000",
            ],
            riscos=[
                "Loss of job opportunity if acceptance is not confirmed by the deadline."
            ],
            observacoes=""
        ),
        # 6) Laudo médico genérico (pt, sem dados sensíveis reais)
        dict(
            arquivo="laudo_neurologia.txt",
            texto=(
                "Relatório Médico - Serviço de Neurologia\n\n"
                "Paciente de 23 anos, em acompanhamento desde 2019, por queixa de tiques motores "
                "e movimentos involuntários. Exames de imagem sem alterações estruturais relevantes. "
                "Foi iniciado tratamento com medicação em baixa dose, com boa resposta clínica.\n"
                "Recomenda-se seguimento ambulatorial e reavaliação em 6 meses."
            ),
            tipo_documento="relatorio_medico",
            idioma="pt",
            resumo=(
                "Relatório médico neurológico descrevendo paciente com tiques motores e boa resposta "
                "ao tratamento, com recomendação de seguimento ambulatorial."
            ),
            palavras_chave=[
                "relatório médico",
                "neurologia",
                "tiques motores",
                "tratamento",
            ],
            entidades=[],
            datas_importantes=[
                "2019",
            ],
            valores_monetarios=[],
            riscos=[
                "Risco de piora dos tiques em caso de interrupção do tratamento ou falta de seguimento."
            ],
            observacoes="Texto fictício apenas para treinamento do modelo."
        ),
        # 7) Trecho de contrato (pt)
        dict(
            arquivo="contrato_prestacao_servicos.txt",
            texto=(
                "CONTRATO DE PRESTAÇÃO DE SERVIÇOS\n\n"
                "Cláusula 1ª - O CONTRATADO se obriga a prestar serviços de consultoria em redes "
                "e infraestrutura ao CONTRATANTE pelo prazo de 12 (doze) meses.\n"
                "Cláusula 2ª - O valor mensal pelos serviços prestados será de R$ 4.000,00, "
                "pagos até o quinto dia útil de cada mês.\n"
            ),
            tipo_documento="contrato",
            idioma="pt",
            resumo=(
                "Trecho de contrato de prestação de serviços de consultoria em redes e infraestrutura, "
                "com prazo de 12 meses e valor mensal de R$ 4.000,00."
            ),
            palavras_chave=[
                "contrato",
                "prestação de serviços",
                "consultoria",
                "redes",
                "infraestrutura",
            ],
            entidades=[
                "CONTRATADO",
                "CONTRATANTE",
            ],
            datas_importantes=[],
            valores_monetarios=[
                "R$ 4.000,00",
            ],
            riscos=[
                "Possível inadimplência do contratante.",
                "Possíveis penalidades contratuais em caso de descumprimento."
            ],
            observacoes=""
        ),
        # 8) Notícia resumida (pt)
        dict(
            arquivo="noticia_ia_universidade.txt",
            texto=(
                "Universidade anuncia novo laboratório de Inteligência Artificial aplicado a redes "
                "de comunicação. O projeto será iniciado em março de 2026 e contará com parcerias "
                "de empresas do setor de telecomunicações.\n"
            ),
            tipo_documento="noticia",
            idioma="pt",
            resumo=(
                "Notícia sobre a criação de um laboratório de IA aplicada a redes de comunicação, "
                "com início previsto para março de 2026 e parcerias com empresas de telecom."
            ),
            palavras_chave=[
                "notícia",
                "inteligência artificial",
                "redes de comunicação",
                "laboratório",
            ],
            entidades=[
                "Universidade",
                "empresas de telecomunicações",
            ],
            datas_importantes=[
                "03/2026",
            ],
            valores_monetarios=[],
            riscos=[],
            observacoes=""
        ),
    ]


def build_training_examples(num_examples: int = 50):
    """
    A partir da base de exemplos, gera ~num_examples pares (prompt, response).
    Caso haja menos exemplos base, replica com pequenas variações no nome do arquivo.
    """
    base_docs = build_examples_base()
    examples = []

    # Gera um exemplo por documento base
    for doc in base_docs:
        prompt = make_system_prompt(texto=doc["texto"], arquivo=doc["arquivo"])
        response_obj = {
            "arquivo": doc["arquivo"],
            "tipo_documento": doc["tipo_documento"],
            "idioma": doc["idioma"],
            "resumo": doc["resumo"],
            "palavras_chave": doc["palavras_chave"],
            "entidades": doc["entidades"],
            "datas_importantes": doc["datas_importantes"],
            "valores_monetarios": doc["valores_monetarios"],
            "riscos": doc["riscos"],
            "observacoes": doc["observacoes"],
        }
        examples.append(
            {
                "prompt": prompt,
                # response é uma string JSON (como nos seus outros datasets)
                "response": json.dumps(response_obj, ensure_ascii=False),
            }
        )

    # Se quiser mais exemplos (ex: 50), replica com pequenas variações nos nomes de arquivo
    i = 1
    while len(examples) < num_examples:
        for doc in base_docs:
            if len(examples) >= num_examples:
                break
            arquivo_novo = doc["arquivo"].replace(".txt", f"_{i:02d}.txt")
            prompt = make_system_prompt(texto=doc["texto"], arquivo=arquivo_novo)

            response_obj = {
                "arquivo": arquivo_novo,
                "tipo_documento": doc["tipo_documento"],
                "idioma": doc["idioma"],
                "resumo": doc["resumo"],
                "palavras_chave": doc["palavras_chave"],
                "entidades": doc["entidades"],
                "datas_importantes": doc["datas_importantes"],
                "valores_monetarios": doc["valores_monetarios"],
                "riscos": doc["riscos"],
                "observacoes": doc["observacoes"],
            }

            examples.append(
                {
                    "prompt": prompt,
                    "response": json.dumps(response_obj, ensure_ascii=False),
                }
            )
        i += 1

    return examples


def main():
    examples = build_training_examples(num_examples=50)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[OK] Dataset JSON-only gerado com {len(examples)} exemplos em:")
    print(f"     {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

