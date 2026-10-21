#!/usr/bin/env bash
# Launch the PHC-X demo: the API, the extension sandbox page, and a
# pre-flight check that the model loaded.  Ctrl-C stops everything.
set -euo pipefail

cd "$(dirname "$0")/.."

API_PORT="${PHC_API_PORT:-8000}"
DEMO_PORT="${PHC_DEMO_PORT:-8022}"

cleanup() {
  echo
  echo "stopping..."
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "${DEMO_PID:-}" ]] && kill "$DEMO_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting the PHC-X service on :$API_PORT ..."
python3 -m uvicorn api.main:app --host 127.0.0.1 --port "$API_PORT" \
  >/tmp/phcx-demo-api.log 2>&1 &
API_PID=$!

echo "serving the extension sandbox on :$DEMO_PORT ..."
(cd extension/demo && python3 -m http.server "$DEMO_PORT" \
  >/tmp/phcx-demo-page.log 2>&1) &
DEMO_PID=$!

# wait for the service and confirm the model is actually loaded
for _ in $(seq 1 40); do
  if curl -sf "http://127.0.0.1:$API_PORT/health" >/tmp/phcx-health.json 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if [[ ! -s /tmp/phcx-health.json ]]; then
  echo "the service did not come up; see /tmp/phcx-demo-api.log" >&2
  exit 1
fi

python3 - <<'PY'
import json
with open("/tmp/phcx-health.json") as fh:
    h = json.load(fh)
print(f"  status   {h['status']}")
print(f"  model    {h['model']}")
print(f"  policy   {h.get('model_path')}")
if h["model"] == "heuristic":
    print("\n  WARNING: no trained checkpoint found -- the demo will fall back to the")
    print("  transparent heuristic.  Run:  phc train --config configs/train.yaml")
PY

cat <<EOF

  ready.

    API           http://127.0.0.1:$API_PORT/health
    dashboard     http://127.0.0.1:$API_PORT/dashboard/
    sandbox page  http://127.0.0.1:$DEMO_PORT/chat.html

  To see the guard:
    1. open chrome://extensions, turn on Developer mode
    2. "Load unpacked" -> the extension/ folder of this repo
    3. open the sandbox page above and press Enter on a message

  (Chrome 137+ ignores --load-extension, so load it from the UI.)

  Ctrl-C to stop.
EOF

wait $API_PID
