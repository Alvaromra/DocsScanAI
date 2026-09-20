# DocsScanAI

Varredura e análise de documentos **100% local**: o texto sai do arquivo, é indexado na sua máquina e respondido por um LLM que roda offline via [Ollama](https://ollama.com). Nada é enviado para serviços externos, o que permite trabalhar com contratos, currículos e outros documentos sensíveis.

## O que ele faz

- **Extrai texto** de PDF, DOCX, CSV, XLSX, TXT, JSON, HTML e logs.
- **Encontra padrões** no conteúdo: e-mail, CPF, CNPJ, telefone, URL, datas e valores monetários.
- **Analisa a estrutura** de cada documento: tipo provável (contrato, currículo, nota fiscal, código), títulos, seções, estatísticas de texto.
- **Reconhece idioma e entidades** com spaCy, em português e inglês: pessoas, organizações e locais.
- **Responde perguntas sobre os documentos** (RAG): a busca combina lemas do spaCy com n-gramas de caracteres, então "multas" encontra "multa" e "penalty" encontra "penalties". O trecho relevante vai como contexto para o modelo local.
- **Exporta tudo** em CSV, XLSX, JSON, TXT, PDF e DOCX.
- **Treina um modelo próprio** a partir dos seus documentos, com um pipeline de LoRA e conversão para GGUF.

## Como está organizado

| Caminho | O que é |
| --- | --- |
| `scanner_docs.py` | CLI de varredura. Gera tudo o que está em `saida/`. |
| `app.py` | Web app Flask: upload, listagem, página do documento e chat. |
| `ia_local_analise.py` | Análise de cada documento com o LLM local. |
| `project/app/nlp.py` | Camada de spaCy: idioma, lematização e entidades. |
| `project/app/rag_local.py` | Índice TF-IDF e busca de contexto para o chat. |
| `project/app/main.py` | API FastAPI com persistência em Postgres. |
| `project/` | Dockerfile, Compose e guia de deploy. |
| `scripts/` | Treino, merge e conversão de LoRA. |
| `auto_learning_pipeline.py` | Pipeline que gera dataset e dispara o treino. |
| `documentos/`, `saida/` | Seus arquivos e as saídas geradas. Ambas ignoradas pelo Git. |

## Começando

Requisitos: Python 3.10 ou mais novo, e o [Ollama](https://ollama.com) instalado para as partes de IA.

```bash
git clone https://github.com/Alvaromra/DocsScanAI.git
cd DocsScanAI
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

O `requirements.txt` já traz o spaCy e os modelos de português e inglês. O modelo de espanhol está comentado no arquivo, é só descomentar se precisar.

### Só o scanner, pela linha de comando

```bash
python3 scanner_docs.py --auto              # processa ./documentos
python3 scanner_docs.py --path ~/Documents  # ou outra pasta
```

Resultados em `saida/`:

- `_docs_index.*`: visão geral dos arquivos encontrados
- `analise_detalhada.*`: estrutura, estatísticas, idioma e entidades
- `geral/`, `por_padrao/`, `por_arquivo/`: padrões encontrados
- `texto_bruto/`: texto completo de cada documento, que alimenta o RAG
- `cv_extracao/`: dados estruturados quando um currículo é detectado

### Web app com chat

Baixe um modelo e suba o app:

```bash
ollama pull phi3:mini
FLASK_HOST=127.0.0.1 FLASK_PORT=5050 \
OLLAMA_MODEL_GENERAL=phi3:mini OLLAMA_MODEL_ANALYSIS=phi3:mini \
sh project/scripts/run_flask.sh
```

Abra http://localhost:5050, envie um documento e pergunte sobre ele no chat.

O `phi3:mini` roda bem em máquinas modestas. Para respostas melhores, com mais memória disponível, troque por `llama3.1:8b`.

### API e Postgres com Docker

```bash
cd project
docker compose up --build
```

- API FastAPI em http://localhost:8000 (`GET /health`, `POST /analisar`, `POST /upload`)
- Web app Flask em http://localhost:5001

Detalhes de deploy em [`project/README_DEPLOY.md`](project/README_DEPLOY.md).

## Configuração

Todas as variáveis são opcionais e têm padrão.

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Endereço do Ollama |
| `OLLAMA_MODEL_GENERAL` | `llama3.1:8b` | Modelo do chat |
| `OLLAMA_MODEL_ANALYSIS` | `llama3.1:8b` | Modelo da análise de documentos |
| `OLLAMA_PRELOAD` | `1` | Carrega o modelo ao subir, em segundo plano |
| `FLASK_HOST` / `FLASK_PORT` | `0.0.0.0` / `5000` | Onde o web app escuta |
| `RAG_USE_SPACY` | `1` | Liga a lematização na busca |
| `RAG_MIN_SCORE` | `0.08` | Nota mínima para um trecho virar contexto |
| `RAG_CHAR_WEIGHT` | `0.5` | Peso dos n-gramas de caracteres na busca |
| `SCANNER_NLP` | `1` | Liga idioma e entidades no scanner |
| `NLP_MODEL_PT` | `pt_core_news_sm` | Modelo de português (`_md` é mais preciso) |

## Estado atual e limitações

Este é um projeto de estudo, em evolução, e vale ser honesto sobre onde ele está:

- **Não há autenticação.** Qualquer pessoa que alcance a porta consegue enviar e apagar documentos. Rode em `127.0.0.1` enquanto isso não muda.
- O `app.py` ainda é um arquivo grande com HTML embutido, e a separação entre Flask e FastAPI vai ser resolvida.
- O modelo pequeno do spaCy erra lemas e classificações de entidade de vez em quando. Os modelos `_md` reduzem isso.
- Ainda não há testes automatizados nem CI.

## Licença

Ainda não definida.
