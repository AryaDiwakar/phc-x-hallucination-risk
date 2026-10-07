# PHC-X Literature Review

Reference list for the PHC-X project. **Every citation below was individually
verified against arXiv / ACL Anthology on 2026-10-06**; items we could not
verify are flagged explicitly. No citation in this project may appear in the
paper unless it appears here.

---

## 1. Pre-generation (pre-decoding) hallucination detection

The core line of work PHC-X builds on: predict hallucination *before* tokens
are produced, using internal signals.

| Work | Citation | Method | Access |
|---|---|---|---|
| FactCheckmate | Alnuhait, D., Kirtane, N., Khalifa, M., Peng, H. (2024/2025). *FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs.* Findings of EMNLP 2025. arXiv:2410.02899 | Classifier over hidden states predicts pre-decoding hallucination, then steers hidden states to mitigate it | white-box |
| KEEN | Gottesman, D., Geva, M. (2024). *Estimating Knowledge in Large Language Models Without Generating a Single Token.* EMNLP 2024. arXiv:2406.12673, DOI:10.18653/v1/2024.emnlp-main.232 | Probes entity representations to estimate per-entity knowledge before generation | white-box |
| HALP | Kogilathota, S. A., Vallabha E G, S., Sun, L., Zhou, J. (2026). *HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token.* EACL 2026, pp. 6067–6085. arXiv:2603.05465 | MLP probes on pre-generative VLM states; up to 0.93 AUROC, single forward pass | white-box |
| Soft-target attention probing | Miftakhova, A., Zaytsev, A. (2026). *Pre-Generation Hallucination Detection in Large Language Models via Soft-Target Attention Probing.* ACL ARR May 2026. arXiv:2606.21917 | Risk estimation trained on soft targets from the output distribution; attention probes over prompt representations | white-box |
| HalluTracer | Guo, Z., et al. (2026). *HalluTracer: Pre-Decoding Truthfulness Prediction via Depth-Averaged Probe-Logit.* arXiv:2608.16353 | Depth-averaged probe logits predict truthfulness before decoding | white-box |
| Confidence-aware routing | Nandakishor, M. (2025). *Confidence-Aware Routing …: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation.* arXiv:2510.01237 | P(hallucination \| query) estimated before generation, then queries are routed | black/gray-box |
| HaluNet | Tong, C., et al. (2025). *HaluNet: Learning Hallucination Risk from Internal Signals in LLM Question Answering.* arXiv:2512.24562 | Fuses token likelihood, entropy, hidden states from a single pass | white-box |

**Positioning:** all of these estimate risk from the *model's internals on a
single prompt*. None of them looks at the *conversation* (history, provenance,
turn-level consistency), and none is evaluated as a pre-generation gate over a
multi-turn benchmark. PHC-X is black-box by default and conversation-aware.

## 2. Uncertainty and post-generation detection

| Work | Citation | Method |
|---|---|---|
| Semantic entropy | Kuhn, L., Gal, Y., Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.* ICLR 2023. arXiv:2302.09664 | Entropy over clusters of semantically equivalent sampled generations |
| Semantic entropy probes | Kossen, J., Han, J., Razzak, M., Schut, L., Malik, S., Gal, Y. (2024). *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs.* ICML 2024 FMDL workshop. arXiv:2406.15927 | Linear probes predict semantic entropy from a single generation's hidden states |
| Kernel language entropy | Nikitin, A. V., Kossen, J., Gal, Y., Marttinen, P. (2024). *Kernel Language Entropy.* NeurIPS 2024. arXiv:2405.20003 | von Neumann entropy over a semantic-similarity kernel of representations |
| Factuality probes (long-form) | Han, J., Band, N., Razzak, M., Kossen, J., Rudner, T. G. J., Gal, Y. (2025). *Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation.* Findings of EMNLP 2025, pp. 16209–16226. DOI:10.18653/v1/2025.findings-emnlp.880 (no arXiv version exists) | Claim-level factuality probabilities from hidden states in one generation call |
| SelfCheckGPT | Manakul, P., Liusie, A., Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative LLMs.* EMNLP 2023, pp. 9004–9017. arXiv:2303.08896 | Sample consistency across multiple black-box responses |
| Attention sinks | Binkowski, J., Adamczewski, K., Kajdanowicz, T. (2026). *Attention Sinks as Internal Signals for Hallucination Detection in LLMs.* ICML 2026. arXiv:2604.10697 | Training-free internal signal from attention-sink dynamics |

> **Correction flag:** an earlier draft of this project cited *"Kossen et al.,
> semantic entropy, NeurIPS 2024"* — **that paper does not exist**. Cite Kuhn
> et al. (ICLR 2023) for semantic entropy and Kossen et al. (arXiv:2406.15927)
> for the probing variant.

## 3. Intervention, verification, and attribution

| Work | Citation | Method |
|---|---|---|
| Chain-of-Verification | Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., Weston, J. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models.* Findings of ACL 2024, pp. 3563–3578. arXiv:2309.11495 | Draft → plan verification questions → answer independently → revise |
| RARR | Gao, L., Dai, Z., Pasupat, P., et al. (2023). *RARR: Researching and Revising What Language Models Say, Using Language Models.* ACL 2023, pp. 16477–16508. arXiv:2210.08726 | Retrieves evidence *after* generation and revises the output for attribution |
| False presuppositions | Sieker, J., Lachenmaier, C., Zarrieß, S. (2025). *LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High.* CogSci 2025. arXiv:2505.22354 | Linguistic analysis of adopting vs. rejecting false presuppositions in prompts |

> **Note:** RARR is post-generation; PHC-X uses pre-generation evidence
> retrieval instead (the risk decision happens *before* decoding).

## 4. Dialogue, memory, and multi-turn risk

| Work | Citation | Method |
|---|---|---|
| DiaHalu | Chen, K., Chen, Q., Zhou, J., He, Y., He, L. (2024). *DiaHalu: A Dialogue-level Hallucination Evaluation Benchmark for LLMs.* Findings of EMNLP 2024, pp. 9057–9079. arXiv:2403.00896 | 1,103 multi-turn dialogues, expert-annotated for 5 hallucination subtypes |
| HalluHard | Fan, D., Delsad, S., Flammarion, N., Andriushchenko, M. (2026). *HalluHard: A Hard Multi-Turn Hallucination Benchmark.* arXiv:2602.01031 | 950 seed questions with inline citations; hallucinations worsen with turn position |
| Stateful Guardrails | (2026). *Stateful Guardrails for Multi-Turn LLM Systems: A Conversational Risk Accumulation Framework.* arXiv:2607.19361 | Session-level accumulation of risk scores across turns — **closest existing work to PHC-X** |
| HalluGuard | Zeng, X., et al. (2026). *HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs.* ICLR 2026. arXiv:2601.18753 | Separates data- vs. reasoning-driven hallucinations; lists multi-turn *"prognostic indicators … predict and preempt"* as **future work** |
| ConsistencyGate | Zhang, Y., Li, S. (2026). *ConsistencyGate: Preventing Memory Contamination in LLM Agents via Self-Consistency Admission Control.* arXiv:2607.22962 | Write-time admission gate on agent memory using K-query self-consistency |
| STALE | Chao, H., Bai, Y., Sheng, R., Li, T., Sun, Y. (2026). *STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?* arXiv:2605.06527 | Benchmark of 400 implicit stale-memory conflict scenarios |

> **Ambiguity flag:** *"MemGuard"* names at least four unrelated 2026 papers
> (arXiv:2605.28009, 2608.21867, 2603.26797, 2510.02373). We do not cite it
> unless a specific ID is used.

## 5. Confirmed research gap

As of 2026-10-06 we found **no paper that predicts hallucination risk before
generation using conversation-level signals** (history consistency,
provenance of stored claims, entity/temporal/presupposition tracking) and then
chooses an intervention (retrieve / clarify / abstain) under a calibrated
policy:

- Pre-generation works (§1) are single-prompt and white-box.
- Dialogue works (§4) evaluate or accumulate risk *after* turns exist, or
  gate memory writes (ConsistencyGate), not the next assistant turn.
- HalluGuard (ICLR 2026) explicitly names conversation-level prognostic
  prediction as future work.

PHC-X targets exactly this gap: black-box, conversation-aware, provenance-aware
pre-generation risk prediction with calibrated thresholds and adaptive
intervention, benchmarked on a purpose-built conversation-level dataset
(PHC-Conversation) and delivered end-to-end (CLI, API, dashboard, extension).
