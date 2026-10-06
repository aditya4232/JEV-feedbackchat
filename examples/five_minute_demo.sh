#!/usr/bin/env bash
# Five-minute offline demo of jev-feedbackchat. No API key needed.
# Run from anywhere:  bash examples/five_minute_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/4  Install pinned dependencies (including the optional Jev SDK)"
uv sync --frozen --extra live

echo "==> 2/4  Run the four-turn demo conversation (explicit offline/mock modes)"
uv run jevloop --mode offline --generator-mode mock --transcript transcripts/run.jsonl

echo "==> 3/4  Test suite (live-marked tests are excluded unless explicitly selected)"
uv run pytest -q

echo "==> 4/4  Evaluation harness check (offline mock) and separate keyword baseline"
uv run jevloop-eval --mode offline
uv run jevloop-eval --mode baseline

echo
echo "Done. To use real Jev instead of the mock:"
echo "  cp .env.example .env && edit TYPESAFE_API_KEY"
echo "  set -a; source .env; set +a"
echo "  uv run jevloop --mode live"
