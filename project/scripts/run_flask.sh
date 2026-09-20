#!/bin/sh
# Sobe o web app Flask (app.py na raiz do repo).
# Usado pelo serviço "flask" do docker-compose; também funciona local:
#   sh project/scripts/run_flask.sh
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Garante que "from project.app..." funciona
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p saida/texto_bruto uploads

exec python3 app.py --host "${FLASK_HOST:-0.0.0.0}" --port "${FLASK_PORT:-5000}"
