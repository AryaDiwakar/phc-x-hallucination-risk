# PHC-X Product Track

PHC-X is also a local-first plugin product, not only an experiment:

```text
browser chat composer -> MV3 extension -> local PHC-X API -> policy decision
                                       -> generate / retrieve / clarify / abstain
```

The supported ChatGPT integration is a **Chrome/Chromium browser extension**
that runs on `chatgpt.com` (and the legacy `chat.openai.com` hostname). It is
not a custom GPT, a ChatGPT API plugin, or a model fine-tune: ChatGPT remains
the model, while PHC-X checks the prompt and the full transcript currently
loaded in the page before the prompt is sent. The API runs on the user's
machine; conversation text is sent only to that local service.

## Product Behaviour

- The extension intercepts Enter/send events without sending conversation data
  to a hosted service.
- It reads ChatGPT's role-tagged message nodes in page order and sends all
  loaded user/assistant text turns for each check (not just the most recent
  few). PHC-X's feature window uses recent turns, while its consistency and
  contamination checks can inspect the full supplied history.
- The API runs locally at `127.0.0.1`, loads the trained checkpoint, and logs
  decisions append-only for review.
- Low-risk messages proceed normally.
- Exact greetings and acknowledgements (for example, “hii” or “thanks”) bypass
  empty-retrieval risk; they do not ask for a factual answer.
- Blocking is off by default: the risk model is experimental, so scores are
  advisory unless the user opts into **pause on clarify / abstain**.
- Retrieval recommendations show the user the risk factors and evidence path.
- High-risk checks offer a local **Improve prompt** rewrite. It preserves the
  request, cites relevant user-provided context when found, otherwise adds
  explicit evidence/uncertainty constraints, and reranks candidate phrasings
  with the local PHC-X pipeline. When Ollama and `qwen2.5:3b` are available,
  and explicitly enabled with `PHCX_ENHANCER_MODEL=qwen2.5:3b`, that local model
  can contribute a short answer-safety instruction; no chat transcript is sent
  to it. It is off by default because it sometimes proposes irrelevant wording.
  PHC-X deterministically preserves the exact request and quotes source context.
  The user reviews/inserts the draft; it never auto-sends, and a lower model
  score is not proof of correctness. The local LLM is never the risk judge.
- Clarify/abstain decisions require an explicit `send anyway` override.
- If the local service is unavailable, the extension fails open after a short
  timeout rather than trapping the user.
- Dashboard controls expose policy thresholds, action distribution, benchmark
  gates, and the experiment archive. Labelled feedback is collected through the
  API's `POST /v1/feedback` endpoint (appended to the same decision log); the
  dashboard itself is read-only.

## Product Acceptance

```bash
phc serve
# Load extension/ as an unpacked extension in Chrome, then visit ChatGPT.
# Optional local sandbox (in another terminal):
python3 -m http.server 8001
# open http://127.0.0.1:8001/extension/demo/chat.html
```

### Install in Chrome

1. Start the local service from the repository: `phc serve`.
2. Open `chrome://extensions`, enable **Developer mode**, and choose **Load
   unpacked**. Select this repository's `extension/` directory.
3. Open `chatgpt.com`, then use the PHC-X toolbar popup to confirm the local
   service is online. Submit with Enter or ChatGPT's send button to run a check.
4. Optional: open the demo at `http://127.0.0.1:8001/extension/demo/chat.html`
   while the repository's static server is running on port 8001.

The extension is currently distributed as source for local unpacked
installation; it has not been published to the Chrome Web Store. Transcript
capture is limited to text present in the loaded page DOM (for example, it
cannot inspect messages that have not been loaded or image contents).

The risk model was trained on synthetic conversations and has known transfer
limitations. A narrow source-aware override prevents high-confidence numeric
and date questions from being blocked solely because the assistant repeated a
matching prior user-provided claim. User assertions remain attributed as such,
not treated as independently verified facts. This is a targeted false-positive
safeguard, not a claim of state-of-the-art generalization; external and human
validation are still required.

For that reason, the API defaults to an 85/15 heuristic/RF blend. Use
`PHCX_MODEL=heuristic` for the transparent baseline, `PHCX_MODEL=auto` for
RF-only, or an explicit model path for a checkpoint. In the generated 10k
comparison, base AUROC was 0.693 for the heuristic, 0.501 for RF, and 0.743
for the blend. This is only a generator-specific diagnostic—not sufficient
deployment validation or a real-world .9 claim.

The acceptance path checks health, version, dashboard static assets, single
analysis, batch analysis, policy GET/PUT, results, decision logging, extension
JavaScript syntax, and a benign first-turn greeting (`hi -> LOW/GENERATE`).

## Research/Product Boundary

The product can ship as a local research preview now. Production deployment
still needs privacy review, browser-specific integration tests, rate limiting,
secure policy storage, telemetry consent, and human validation of false
blocks. The external evaluation work is intentionally separate from product
telemetry so user data is not silently used as research labels.
