#!/usr/bin/env bash
# Five-minute offline demo of jev-feedbackchat. No API key needed.
# Run from anywhere:  bash examples/five_minute_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/4  Install (dev tools sync by default; the live extra adds the Jev SDK)"
uv sync --extra live

echo "==> 2/4  Run the four-turn demo conversation (offline mock classifier)"
uv run jevloop --transcript transcripts/demo.jsonl

echo "==> 3/4  Test suite (live tests are skipped without TYPESAFE_API_KEY)"
uv run pytest -q

echo "==> 4/4  Evaluation harness check (offline) and labelled regex baseline"
uv run jevloop-eval
uv run jevloop-eval --mode baseline

echo
echo "Done. To use real Jev instead of the mock:"
echo "  cp .env.example .env && edit TYPESAFE_API_KEY"
echo "  set -a; source .env; set +a"
echo "  uv run jevloop --mode live"
