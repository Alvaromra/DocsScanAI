# ScanDocs

Ferramentas para varrer documentos localmente e expor uma API simples de análise.

## Estrutura
- `scanner_docs.py` — CLI que lê PDFs/DOCX/CSV/XLSX/TXT/JSON/HTML/logs, extrai texto, detecta padrões (email, CPF/CNPJ, URLs, telefones, datas, valores), analisa estrutura e salva relatórios em `./saida/`.
- `project/` — API FastAPI com Docker Compose (API + Postgres) e guia de deploy em `project/README_DEPLOY.md`.
- `app.py` — mini app Flask que chama modelos locais via Ollama (opcional).
- `scripts/` — utilitários para treino/merge/conversão de LoRA.
- `documentos/` — **IGNORADA DO GIT**. Coloque aqui os arquivos que quiser processar.

## Pré-requisitos
- Python 3.10+ e pip
- Docker (para usar a API com Compose)
- Para análises com LLM local: Ollama rodando e modelos configurados nas variáveis do `app.py`.

## Como usar o scanner (CLI)
```bash
python3 scanner_docs.py --auto        # usa ./documentos
# ou
python3 scanner_docs.py --path /caminho/para/pasta
```
Saídas geradas em `./saida/`:
- `_docs_index.*` visão geral dos arquivos
- `analise_detalhada.*` estrutura e estatísticas
- `geral/`, `por_padrao/`, `por_arquivo/` com matches de padrões
- `texto_bruto/` texto completo por documento
- `cv_extracao/` quando currículos são detectados

O script instala dependências básicas automaticamente (pandas, PyPDF2, openpyxl, python-docx, reportlab). Se a instalação falhar, instale manualmente com pip e rode de novo.

## Como rodar a API + WebApp (Docker Compose)
```bash
cd project
docker compose up --build
# API em http://localhost:8000 (FastAPI)
# WebApp Flask em http://localhost:5001
# Ollama (LLM) exposto em http://localhost:11434
```
Endpoints principais:
- `GET /health`
- `POST /analisar` (texto)
- `POST /upload` (pdf/txt)
- `GET /` → Web UI simples para enviar arquivo e listar análises salvas no Postgres
- Modos: `mode=baseline` (default) ou `mode=rag` para usar contexto do corpus (pasta `saida/texto_bruto` montada no container).

Mais detalhes e dicas de deploy em `project/README_DEPLOY.md`.

## Observações
- A pasta `documentos/` permanece local/privada e está no `.gitignore`.
- Modelos pesados e pastas geradas (models/, uploads/, saida/, etc.) também são ignorados no Git.
- O serviço Flask monta o repositório local via volume (`..:/workspace`), reutilizando seus arquivos e pipelines já existentes.
- O serviço Ollama roda no Compose (porta 11434) para a camada de IA. Substitua o modelo em `app.py` se quiser outro (default `llama3.1:8b`).
- Para RAG: coloque textos em `saida/texto_bruto/` (gerado pelo scanner) e chame `POST /analisar?mode=rag` ou `POST /upload?mode=rag` para recuperar contexto antes do LLM. Se Ollama não estiver disponível, o endpoint cai no modo simplificado.
