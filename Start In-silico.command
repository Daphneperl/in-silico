#!/bin/bash
cd "$(dirname "$0")" 2>/dev/null || true

if [ -f "./index.html" ] && [ -f "./package.json" ]; then
  APP_DIR="$(pwd)"
else
  APP_DIR=""
  for candidate in \
    "$HOME/in-silico" \
    "$HOME/Documents/in-silico" \
    "$HOME/Documents/GitHub/In-silico-synth" \
    "$HOME/Documents/GitHub/in-silico"
  do
    if [ -f "$candidate/index.html" ] && [ -f "$candidate/package.json" ]; then
      APP_DIR="$candidate"
      break
    fi
  done
fi

if [ -z "$APP_DIR" ]; then
  osascript -e 'display dialog "Could not find the in-silico folder." buttons {"OK"} default button 1 with icon stop'
  exit 1
fi

cd "$APP_DIR" || exit 1

if lsof -iTCP:3001 -sTCP:LISTEN >/dev/null 2>&1; then
  open "http://127.0.0.1:3001"
  exit 0
fi

npm run dev
