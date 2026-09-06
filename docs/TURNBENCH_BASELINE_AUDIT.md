# TurnBench Baseline Coverage Audit

Audited: 2026-09-05 UTC

## Verdict

No: this repository has validated the TurnBench harness, but it has not run all
TurnBench baselines locally. The exact distinction matters:

- **Local model inference complete:** the exact official RMS VAD reference,
  SmartTurn v3.1/v3.2, exact TurnBench VapGPT/OTO, ESPnet TurnTaking mixed-mono,
  Mimi Endpointer, Kyutai Semantic VAD, and MaAI trilingual Realtime-VAP on all
  38 dev conversations. DualTurn also has a full local run, although the
  official roster still treats it as a TODO.
  The MaAI run is an additional VAP checkpoint/pipeline and remains separate
  from the exact TurnBench `vap` baseline.
- **Sanity parity complete:** the dev-only oracle/gold replay. It validates the
  scorer and is not a deployable baseline.
- **All other official model baselines:** not run locally on TurnBench.

These statements are backed by this repository's [implementation
status](STATUS.md) and the complete local TurnBench inference artifacts under
[`results/en/turnbench/realtime_vap_tri/20260904T142500Z/final`](../results/en/turnbench/realtime_vap_tri/20260904T142500Z/final)
and
[`results/en/turnbench/vap_gpt_oto/20260904T192700Z`](../results/en/turnbench/vap_gpt_oto/20260904T192700Z),
the complete ESPnet artifact under
[`results/en/turnbench/espnet_turntaking/20260904T223000Z-final`](../results/en/turnbench/espnet_turntaking/20260904T223000Z-final),
and the complete Mimi artifact under
[`results/en/turnbench/mimi_endpointer/20260905T061000Z`](../results/en/turnbench/mimi_endpointer/20260905T061000Z),
plus the complete Kyutai artifact under
[`results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final`](../results/en/turnbench/kyutai_semantic_vad/20260905T073700Z-final).

## Current official scope

At the pinned and current upstream commit
[`e02d547`](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e),
TurnBench lists 15 roster entries: 14 scored paper baselines plus the dev-only
oracle. `dualturn` is explicitly a TODO/stub and has no official submission
([official roster](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/README.md#roster),
[official test results](https://github.com/SesameAILabs/turnbench/blob/e02d547803fd0e45221afcaeb1cd24b61ae7568e/results/leaderboard-test.json)).

The live EOT leaderboard additionally contains three external configurations:
Tavus Sparrow-2 and Neosapience Nunchi v1 in dual- and single-channel modes.
They are marked external and do not have matching adapters in the official
TurnBench repository, so their published scores can be used as reference rows
but cannot currently be treated as locally reproducible baselines
([live leaderboard](https://turnbench.sesame.com/)).

## Coverage matrix

| Official entry | Local status | Public reproduction path / blocker | Priority |
| --- | --- | --- | --- |
| RMS energy VAD | Exact local inference complete | All 38 dev conversations; generated JSON equals the official committed artifact | Done |
| VAP / VapGPT | Exact local inference complete | All 38 dev conversations; 5.26M task probabilities agree with the quantized official artifacts within `5.01e-5`, and official scores reproduce exactly ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/vap)) | Done |
| Smart Turn v3 | v3.1 and v3.2 local inference complete | Full 38-conversation GPU runs; v3.1 is not exact artifact parity because the official wall-clock VAD reset is scheduling-dependent ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/smart_turn_v3)) | Done |
| ESPnet Turntaking, mixed mono | Full local inference complete | All 38 dev conversations; fixed-official-threshold recall, FPR, and latency parity pass the planned tolerance. The TF32 probability trace is not bit-exact ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/espnet_turntaking)) | Done |
| ESPnet Turntaking, per-channel | Missing | Reuses the same public model but runs two passes; useful ablation after the mixed-mono run ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/espnet_turntaking_perchannel)) | P1 |
| Mimi Endpointer | Full local inference complete | All 38 dev conversations; fixed advertised-threshold metrics pass the harness tolerance, with exact interruption metrics and a one-TP EOT boundary difference ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/mimi_endpointer)) | Done |
| Kyutai Semantic VAD | Full local inference complete | All 38 dev conversations through the native streaming runner; EOT and derived INT fixed-threshold metrics pass the planned harness tolerance after matching upstream CUDA resampling ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/kyutai_semantic_vad)) | Done |
| WavLM Base Causal | Missing | Public checkpoint; needs the documented patched S3PRL path and non-stock frontend ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/wavlm_base_causal)) | P2 |
| WavLM Large Causal | Missing | Same integration burden as Base, with a larger encoder ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/wavlm_large_causal)) | P2 |
| WavLM Large Anchor | Missing | Public checkpoint, but depends on an open ESPnet Universa branch, author overlays, and compatibility stubs ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/wavlm_large_anchor)) | P3 |
| OpenAI server VAD | Missing | Paid network API; official recipe says server behavior can drift and is not bit-reproducible ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/openai_server_vad)) | P3 |
| OpenAI semantic VAD | Missing | Same paid/non-bit-reproducible API constraint ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/openai_semantic_vad)) | P3 |
| Gemini VAD | Missing | Cloud full-duplex recording pipeline plus gated downstream VAD model; not a component-only local checkpoint ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/gemini_vad)) | P3 |
| Moshi VAD | Missing | Public Moshi path, but requires expensive real-time full-duplex recording and a gated downstream VAD stage ([recipe](https://github.com/SesameAILabs/turnbench/tree/e02d547803fd0e45221afcaeb1cd24b61ae7568e/baselines/moshi_vad)) | P3 |
| DualTurn (upstream TODO) | Full local inference complete | Pinned Apache-2.0 80 ms streaming graph over all 38 dev conversations; native dual-channel EOT and derived FVAD interruption are kept separate. TurnBench has no official score to reproduce | Done locally |

The external Sparrow-2 and Nunchi rows are **published-score-only** until their
owners expose a runnable model/API suitable for the TurnBench input contract.
The upstream `dualturn` entry is still marked TODO and has no official scored
submission, although a public downloadable DualTurn checkpoint and predictor
now exist.

## Open-weight and license audit

“Public weights” and “open source” are not interchangeable. The table records
the model-card license for the downloadable checkpoint separately from the
implementation license. Creative Commons licenses permit redistribution and
commercial use under their conditions, but are not OSI software licenses.
An absent weight license is reported as unknown rather than inferred from the
code or backbone license.

| Model | Public downloadable weights? | Declared weight license | Code / production note | Local-run recommendation |
| --- | --- | --- | --- | --- |
| RMS VAD | N/A | N/A | Rule-based; exact local 38-conversation run complete | **Done** |
| VAP / VapGPT | Yes | CC-BY-4.0 for [`viks66/VAP_checkpoints`](https://huggingface.co/viks66/VAP_checkpoints) | [VAP code](https://github.com/ErikEkstedt/VoiceActivityProjection) is MIT; attribution is required for the checkpoint | **Done**: exact TurnBench OTO run is the main native dual-channel VAP reference |
| Smart Turn v3.1 and v3.2 | Yes; both versions are files in the same [HF repository](https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main) | BSD-2-Clause | [Code](https://github.com/pipecat-ai/smart-turn) is also BSD-2-Clause; permissive and commercially usable with notices | **Done**: both versions have separate full TurnBench runs |
| ESPnet TurnTaking | Yes | CC-BY-4.0 on the [SWBD model card](https://huggingface.co/espnet/Turn_taking_prediction_SWBD) | [ESPnet code](https://github.com/espnet/espnet) is Apache-2.0; checkpoint attribution required | **Done** in mixed-mono mode; per-channel remains an optional ablation using the same weights |
| Mimi Endpointer | Yes | CC-BY-4.0 on [`viks66/mimi-endpointer`](https://huggingface.co/viks66/mimi-endpointer) | Public TurnBench recipe; attribution required for weights | **Done**: full TurnBench run validates the lightweight codec-state architecture reference |
| Kyutai Semantic VAD (STT-1B) | Yes | CC-BY-4.0 on [`kyutai/stt-1b-en_fr-candle`](https://huggingface.co/kyutai/stt-1b-en_fr-candle) | [Inference code](https://github.com/kyutai-labs/delayed-streams-modeling) is dual Apache-2.0/MIT; the model is EN/FR, not Chinese | **Done locally** as an English semantic-streaming control; not a Chinese candidate |
| WavLM Base/Large Causal | Yes | **Not declared** on [`ZhuoyanTao/causal-wavlm-turn-taking`](https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking) | The custom TurnBench checkpoints/code have no model-card license. Their Microsoft WavLM backbone points to CC-BY-SA-3.0, but that does not establish a license for the custom checkpoints | Research evaluation only until clarified; run **Base Causal only** if another encoder architecture is needed |
| WavLM Large Anchor | Yes, in the same ZhuoyanTao repository | **Not declared** | Same unresolved checkpoint-license issue plus a substantially more fragile ESPnet/Universa stack | Skip for the first report |
| Moshi | Yes | CC-BY-4.0 on [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16) | [Code](https://github.com/kyutai-labs/moshi) is dual Apache-2.0/MIT; TurnBench evaluation is a costly full-duplex generative pipeline, not a component detector | Skip for core detector benchmarking; keep as whole-agent/reference work |
| DualTurn Endpointing | Yes | Apache-2.0 on [`anyreach-ai/dualturn-endpointing`](https://huggingface.co/anyreach-ai/dualturn-endpointing) | Ungated 1.45M-parameter readout with self-contained ONNX graphs; TurnBench still labels its baseline row TODO/no official score | Full local run **done**, with derived interruption separate from native EOT |
| OpenAI server/semantic VAD | No | Closed hosted API | Paid, server behavior can drift; no downloadable checkpoint | Do not block the release on these |
| Gemini | No | Closed hosted API | Paid full-duplex generative pipeline; no downloadable checkpoint | Skip for component baseline report |
| Sparrow-2 / Nunchi | Not established by the official TurnBench materials | Unknown | External leaderboard submissions with no official TurnBench adapter; do not infer availability or license | Published-score reference rows only |

For commercial prototyping, the clearest low-friction choices in this set are
Smart Turn (BSD-2-Clause) and DualTurn (Apache-2.0). VAP, ESPnet TurnTaking,
Mimi Endpointer, Kyutai STT, and Moshi expose public weights under CC-BY-4.0,
which permits commercial use subject to attribution and the license terms.
The three ZhuoyanTao WavLM turn-taking checkpoints should not be presented as
commercially cleared while their repository omits a weight license.

## Recommended remaining set

Do not block the Chinese benchmark on reproducing every expensive/cloud row.
The smallest defensible English architecture-validation set is complete:

1. ESPnet TurnTaking mixed-mono: complete; per-channel is only an ablation.
2. Mimi Endpointer: complete codec-state reference.

Kyutai Semantic VAD is also complete as the useful English-only semantic
streaming control. WavLM Base Causal can follow if broader architecture coverage
is needed, but its undeclared custom-checkpoint license should be recorded in
every result manifest.

After those, WavLM Base Causal is the most useful next architectural reference.
WavLM Anchor, cloud APIs, Gemini, and Moshi add completeness but are not required
before the first Chinese baseline report. Every model can technically submit to
TurnBench if it emits causal event timestamps; “all TurnBench-testable models”
is therefore unbounded. The practical completeness target should be the public
official roster plus clearly marked external reference rows, not every possible
turn detector.
