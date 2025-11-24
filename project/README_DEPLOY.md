# Document Analyzer – Deploy Guide

## Rodar local com Docker
```bash
docker compose up --build
```
API em http://localhost:8000

## Windows / macOS / Linux
- Instale Docker Desktop (ou Docker Engine no Linux).
- Rode `docker compose up --build`.
- Acesse pelo IP local (mesma rede, porta 8000).

## Serviços
- api (FastAPI + Uvicorn)
- db (Postgres)
- opcional: ollama (comentado)

## Variáveis (.env)
```
POSTGRES_DB=docsia
POSTGRES_USER=docsia_user
POSTGRES_PASSWORD=senha_super_secreta
POSTGRES_HOST=db
POSTGRES_PORT=5432
```

## Deploy online

### Railway
1. Conecte o repo do GitHub.
2. Build com Dockerfile.
3. Defina a PORT=8000.
4. Deploy automático.

### Render
1. Novo Web Service.
2. Selecione Docker.
3. Porta 8000.
4. Deploy.

### Fly.io
1. `fly launch`
2. `fly deploy`
3. URL pública gerada.

### Docker Hub + VM
1. `docker build -t seuuser/docsia .`
2. `docker push seuuser/docsia`
3. Na VM: `docker pull seuuser/docsia && docker run -p 8000:8000 seuuser/docsia`

## Arquitetura
- FastAPI stateless → fácil de escalar.
- Docker garante o mesmo ambiente em macOS/Windows/Linux.
- Compose coordena API + DB (+ IA opcional).
- Ollama permanece externo (host.docker.internal) ou como serviço no compose.

## Endpoints
- GET /health
- POST /analisar (texto)
- POST /upload (pdf/txt)

## Modelos locais
Coloque modelos em /app/models (GGUF, safetensors, HF). Integre no código chamando o pipeline/ollama conforme necessário.
