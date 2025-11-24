#!/bin/bash
set -e
gunicorn app:app --bind 0.0.0.0:${PORT:-5050}
