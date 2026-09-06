# X2-Turn Adapter Protocol Contract

Status: implemented contract, captured 2026-09-04 UTC. This document pins the
public X2-Turn release and defines the safe boundary for the `turn-eval`
adapter. It uses only first-party source code, model metadata, and
the authors' paper. Statements marked **repository policy** or **inference** are
not claims made by the X2-Turn authors.

The canonical code repository is [`X-Square-Robot/X2-Turn` at commit
`3609a0fd11b4a1dc2b2629e95538bbe0860da96d`](https://github.com/X-Square-Robot/X2-Turn/tree/3609a0fd11b4a1dc2b2629e95538bbe0860da96d).
The canonical checkpoint is [`x-square-robot/X2-Turn-4B-0812` at revision
`d188a8b86058ece5e293124a278a1a6256af81fa`](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/tree/d188a8b86058ece5e293124a278a1a6256af81fa).
Neither repository publishes a release tag, so runs must save these full
revisions rather than `main`. The corresponding paper is [Fu et al.,
*X2-Turn: Frame-Synchronous Dual-Head Modeling for Joint Streaming ASR and Turn
State Prediction*, arXiv:2608.10878v2](https://arxiv.org/html/2608.10878v2).

## 1. What the released model is

X2-Turn is a single-channel, Chinese/English streaming ASR model with a
parallel turn-state head. It fine-tunes the Voxtral Mini 4B Realtime backbone;
the ASR and turn heads consume the same causal decoder hidden state, while the
turn prediction is not fed back into subsequent ASR decoding. The paper's
architecture consumes a 16 kHz waveform, downsamples audio representations to
12.5 Hz, and emits one ASR token and one turn prediction at each 80 ms step
([paper Sections 2.1 and 3.2](https://arxiv.org/html/2608.10878v2#S2.SS1)).

The public checkpoint is approximately 5.235 billion stored BF16 parameters,
not literally four billion parameters. Hugging Face reports exactly
5,234,985,728 BF16 parameters. Header inspection of the pinned safetensors file
shows 713 tensors and two separate `[131072, 3072]` BF16 output matrices:
`base_model.lm_head.weight` and `vad_lm_head.weight`. The extra full-vocabulary
turn head alone contains 402,653,184 parameters. `4B` is therefore the released
model's product/family name, not an exact artifact parameter count
([model configuration](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/config.json),
[official model architecture note](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/README.md#architecture)).

The model is audio-native in the sense that it performs its own ASR and does
not accept a benchmark transcript. It is not dual-channel: neither the public
Transformers API nor the realtime protocol accepts a separate agent channel.
Agent-speaking state is consumed by the downstream controller policy, not the
model head itself
([local API](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/inference.py),
[realtime bridge](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/realtime.py)).

## 2. Pinned model artifact

The pinned Hub revision contains exactly the following files. Sizes and LFS
SHA-256 values come from the [official revision tree
API](https://huggingface.co/api/models/x-square-robot/X2-Turn-4B-0812/tree/d188a8b86058ece5e293124a278a1a6256af81fa?recursive=true&expand=true).

| File | Bytes | SHA-256 when published through LFS | Role |
| --- | ---: | --- | --- |
| `.gitattributes` | 1,567 | Git blob, not LFS | LFS rules |
| `LICENSE` | 11,351 | Git blob, not LFS | Apache-2.0 text |
| `NOTICE` | 549 | Git blob, not LFS | Derivation/legal notice |
| `README.md` | 8,650 | Git blob, not LFS | Model card |
| `config.json` | 1,572 | Git blob, not LFS | Transformers architecture |
| `generation_config.json` | 191 | Git blob, not LFS | Generation defaults |
| `model.safetensors` | 10,470,070,544 | `75c7fd12babb6d7ffebf6d0cb86388ff5e20b9a1e9ed079c45eaed62ef6a9117` | Canonical BF16 weights |
| `params.json` | 1,343 | Git blob, not LFS | Mistral/vLLM model parameters |
| `processor_config.json` | 384 | Git blob, not LFS | 16 kHz feature extractor |
| `tekken.json` | 14,910,348 | `8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44` | Tokenizer |
| `training_args.bin` | 7,249 | `7235dd0674494aba661d768ebaac8e0b62445afa9f840da951965013810ddd11` | Opaque training metadata |

The listed files total 10,485,013,748 bytes. This is distinct from the Hub's
deduplicated `usedStorage` accounting.

`training_args.bin` is a PyTorch/pickle-family artifact and is not required by
inference. **Repository policy:** do not deserialize it in the preparation or
inference path. The official loader intentionally downloads only
`config.json` and `model.safetensors`; `AutoProcessor` separately resolves the
processor/tokenizer files
([checkpoint loader](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/modeling.py#L196-L264)).

Acquisition must use the configured cache and a full revision:

```bash
export HF_HOME=/mnt/md0/user_wayne/.hf_cache
hf download x-square-robot/X2-Turn-4B-0812 \
  --revision d188a8b86058ece5e293124a278a1a6256af81fa
```

## 3. License and release caveat

The source repository, its `voxtral-realtime` component, and the model card all
declare Apache-2.0. The base model
[`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
also declares Apache-2.0. However, the checkpoint's own pinned `NOTICE` says
that the repository owner must verify base-model obligations and training-data
rights before public distribution
([model NOTICE](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/NOTICE)).

Consequently, `turn-eval` may record the declared license as `Apache-2.0`, but
must also preserve the notice and must not turn that declaration into an
independent assurance about training-data rights. This is a provenance caveat,
not a conclusion that the license is invalid.

## 4. Environment and supported execution paths

The official local environment uses Python 3.11, FFmpeg, libsndfile, and the
unpublished checkout package `voxtral-realtime[transformers]`. Its declared
Python dependencies are `huggingface-hub`, `mistral-common`, `torch`,
`safetensors`, and `transformers>=5.10.0`, plus the bridge dependencies. The
package is not published to PyPI
([environment file](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/environments/environment-transformers.yml),
[`pyproject.toml`](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/pyproject.toml)).

The pinned Hugging Face model card incorrectly suggests installing
`./X2-Turn[transformers]`; the repository root has no `pyproject.toml`. The
canonical GitHub quick start correctly installs
`./voxtral-realtime[transformers]`. The isolated environment must follow the
code repository, not the broken model-card command.

The official quick start requires Linux, an NVIDIA GPU, and at least 24 GB
VRAM; the documented model load uses BF16 on CUDA. The example CLI exposes
`float32`, `float16`, and `bfloat16`, defaults to BF16 on CUDA, and falls back
to float32 on CPU. That means the source has a CPU code path, but upstream does
not claim useful CPU latency or provide a CPU memory benchmark. There are no
released quantized or ONNX weights
([quick-start requirements](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/README.md#quick-start),
[offline CLI](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/transformers/examples/offline_inference.py)).

There are two materially different inference modes:

1. `infer_asr_turn()` is a convenient **offline, two-pass Transformers** path.
   It greedily generates the aligned ASR sequence from the complete clip, then
   performs a second aligned forward pass to obtain turn logits. It must not be
   reported as a native incremental streaming run
   ([Transformers integration guide](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/transformers/README.md)).
2. Production incremental output requires the project's custom overlay on
   vLLM 0.19.1 commit
   `b1388b1fbf5aaef47937fabe98931211684666a6`. Stock vLLM does not emit the
   custom `turn.delta` event. The overlay must be applied to exactly that clean
   vLLM checkout, and the checkpoint must be exported to the Mistral-keyed
   `consolidated.safetensors` layout
   ([vLLM integration guide](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/vllm/README.md)).

**Repository policy:** isolate X2-Turn in its own `uv` project, pin the code
commit plus every resolved dependency in `uv.lock`, and distinguish
`offline-two-pass` from `native-streaming-vllm` in result manifests. The
upstream Conda files are source evidence, not permission to bypass this
repository's `uv` requirement.

## 5. Audio preprocessing and the 80 ms timeline

The released processor specifies mono-compatible 16 kHz audio, a 400-sample
(25 ms) analysis window, a 160-sample (10 ms) hop, and 128 mel bins. The model
configuration consumes eight mel hops per output token, so the native interval
is exactly:

```text
8 hops × 160 samples / 16,000 samples/s = 0.080 s
```

This verifies the 80 ms claim from configuration rather than relying only on
the README
([processor configuration](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/processor_config.json),
[model configuration](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/config.json)).

The pinned tokenizer metadata additionally declares
`transcription_delay_ms=480`, `streaming_look_ahead_ms=2.5`,
`streaming_look_back_ms=52.5`, and 32 left-padding tokens
([`tekken.json`](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/tekken.json)).
These values describe the released streaming tokenizer/frontend contract; the
2.5 ms lookahead must be retained in causality metadata rather than rounded to
zero.

The local file API loads through `mistral_common.Audio`, resamples to the
processor rate, and invokes the feature extractor with `center=True`. The Turn
Demo's convenience loader explicitly averages multichannel files to mono and
uses linear interpolation for resampling. The realtime smoke client is stricter:
it requires mono, signed 16-bit PCM and sends chunks of 1,280 samples, exactly
80 ms at 16 kHz
([local inference](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/inference.py#L38-L71),
[demo loader](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/turn-demo/demo_turn/predictions.py#L61-L74),
[realtime smoke client](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/vllm/examples/voxtral_mtp/test_stream_mtp.py)).

**Repository policy:** the adapter receives mono float32 audio from the core
runner, records the original sample rate, resamples internally to 16 kHz, and
records the exact resampler. It must not pass transcript, source sample rate,
filename, or other dataset metadata to the model.

### 5.1 Streaming delay is not the frame interval

The output grid is always 80 ms, but each predicted semantic target is delayed
by configurable target delay `tau`. The checkpoint default is six frames, or
480 ms. The local API accepts positive multiples of 80 ms. Training sampled
delays from 1 through 30 frames (80–2,400 ms), while the published turn
evaluation reports only 320, 400, and 480 ms
([paper Sections 2.1, 3.2, and 4.2](https://arxiv.org/html/2608.10878v2#S3.SS2),
[inference implementation](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/inference.py#L106-L126)).

The paper anchors a word's state at word onset plus `tau`, and defines its
latency relative to the word end as `tau - word_duration`. This is not directly
the endpointing latency definition used by `turn-eval`; it is a word-level
paper metric
([paper Section 3.3](https://arxiv.org/html/2608.10878v2#S3.SS3)).

**Repository policy:** each raw frame prediction is timestamped at the causal
time when the frame has been emitted, and the 320/400/480 ms setting is recorded
as target delay. The emitted frame time already reflects delayed-stream
availability: do not add `tau` a second time, and never subtract it to make the
decision appear earlier. Preserve the separately declared 2.5 ms frontend
lookahead. The offline two-pass wrapper receives the complete clip and uses centered
feature extraction, so it is suitable for state-classification reproduction
but not evidence of causal streaming latency. Latency claims require the
incremental vLLM path or an independently verified causal implementation.

## 6. Exact output schema

The public wrapper defines a full-vocabulary turn head. `forward()` returns
`vad_logits` with conceptual shape `[batch, timeline, 131072]`. Both official
inference paths select reserved IDs 35–40, cast those six logits to float32,
apply a six-way softmax, and take argmax. Their order is fixed
([Transformers implementation](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/modeling.py#L15-L25),
[vLLM implementation](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/vllm/vllm/model_executor/models/voxtral_mtp_utils.py#L13-L49)):

| Token ID | Inference name | Trained paper token | Verified paper semantics |
| ---: | --- | --- | --- |
| 35 | `idle` | `<|idle|>` | User silence/non-speech from forced alignment. |
| 36 | `noidle` | `<|noidle|>` | Early active speech without enough semantic content; explicitly not silence/background noise. |
| 37 | `speaking` | `<|incomplete|>` | Active speech with partial semantic content. |
| 38 | `turn_end` | `<|complete|>` | Active speech with complete semantic content. |
| 39 | `backchannel` | `<|backchannel|>` | User backchannel or filler, such as “um” or “ah”. |
| 40 | `uncertain` | none | Not defined or supervised by the paper. |

The five trained meanings are defined in [paper Section
2.2](https://arxiv.org/html/2608.10878v2#S2.SS2). Silence frames, including
pre-speech, inter-word, mid-utterance, and trailing silence, are trained as
`idle`; turn states are assigned only to the positions occupied by a word's
word-boundary and subword tokens
([paper Section 2.3](https://arxiv.org/html/2608.10878v2#S2.SS3)).

The local `ASRTurnResult` contains:

- `transcript`: decoded ASR text;
- `generated_token_ids`: ASR-head generation after the prompt prefix;
- `frame_ms`: derived native interval, normally 80;
- `turn_frames`: one row per available frame, with zero-based index,
  `[start_ms, end_ms)`, argmax label, winning probability, and all six
  probabilities.

The realtime overlay emits one JSON event per generated frame:

```json
{
  "type": "turn.delta",
  "turn_class": "speaking",
  "turn_id": 37,
  "probs": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "frame_index": 12
}
```

`probs` is ordered exactly as IDs 35–40 and is a length-six softmax
([vLLM protocol patch](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/patches/vllm-voxtral-mtp.patch)).

### 6.1 `uncertain` contract inconsistency

The model card says `uncertain` is an inference-only low-confidence fallback
that was not trained. The released Transformers and vLLM implementations do
not implement a confidence threshold or fallback rule: they include ID 40 in
the same softmax and argmax as the five trained slots. Therefore its exact
activation semantics are not reproducibly specified by the release
([model-card mapping](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/README.md#paper--inference-label-mapping),
[Transformers `_predict_turn`](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/transformers/inference.py#L74-L91)).

**Repository policy:** retain the raw `uncertain` probability and argmax, treat
it as abstain/unknown for any derived classifier, and record the handling rule.
Do not silently map it to `incomplete`, `wait`, or `idle`.

## 7. Streaming, cache, and reset behavior

The model configuration enables the text decoder KV cache and uses an 8,192
token sliding window. The audio encoder is causal with a 750-frame sliding
window and declares no encoder cache in `params.json`
([`config.json`](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/config.json),
[`params.json`](https://huggingface.co/x-square-robot/X2-Turn-4B-0812/blob/d188a8b86058ece5e293124a278a1a6256af81fa/params.json)).
The patched vLLM server owns the actual incremental request/cache lifecycle;
the public client does not expose the cache as an adapter object.

For each independent clip, the safe reset is:

1. create a new realtime WebSocket/request session;
2. send `session.update` with the pinned served model;
3. append mono PCM and consume paired transcription/turn deltas;
4. send a final `input_audio_buffer.commit`;
5. wait for `transcription.done`, then close the WebSocket;
6. discard all local ASR parts, turn frames, controller state, acoustic-gate
   state, counters, and delay padding before the next clip.

The official bridge follows that pattern. Its higher-level `reset()` closes the
inner realtime session and resets controller and acoustic-gate state. It also
automatically resets after accepting a user turn
([bridge reset](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/server.py#L79-L102),
[session lifecycle](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/realtime.py#L64-L72)).
The offline Transformers helper is call-stateless: it regenerates from a fresh
encoded clip and does not retain an exposed cache between calls.

## 8. Official utterance-level reduction

The paper's EasyTurn evaluation compares the **last non-idle predicted state**
with the utterance ground truth. It reports per-category utterance accuracy for
complete, incomplete, and Chinese backchannel; it does not publish a six-way
frame confusion matrix or a Wait score
([paper Table 1 and Section 4.1](https://arxiv.org/html/2608.10878v2#S4.SS1)).

For reproduction, the upstream-reported main operating point is `tau=480 ms`.
The public model card maps paper labels to inference labels as follows:

| Evaluation category | Native X2 inference label |
| --- | --- |
| `complete` | `turn_end` |
| `incomplete` | `speaking` |
| `backchannel` | `backchannel` |
| `wait` | **no native label** |

The states `idle` and `noidle` are timeline-support states, not utterance-level
EasyTurn target categories. `uncertain` has the contract issue described above.

## 9. FastTurn compatibility

X2-Turn publishes no FastTurn adapter, scorer, evaluation result, or official
FastTurn label mapping. The pinned source tree contains no reference to
FastTurn. The native overlap is limited to the three mappings above, which are
supported by the model card's paper-to-inference table. In particular, X2-Turn
has no Wait state: a spoken instruction to wait may be semantically complete
and therefore produce `turn_end`, but that is not a Wait classifier.

Required modes:

- `fastturn-eot`: supported as a **derived utterance reduction**, restricted to
  gold `complete`/`incomplete`, using last non-idle state and an explicit rule
  for other argmax states.
- `fastturn-real`: usable only as a **derived three-state mapping** for
  `complete`, `incomplete`, and `backchannel`; mapping and abstention policy
  must be frozen before scoring.
- `fastturn-full`: **unsupported natively** because `wait` has no X2-Turn
  class. It may only appear on a separate `derived-state` leaderboard after a
  calibration-only mapping/policy is specified. It must never be presented as
  native four-state performance.

The safest first mapping is exact-only:

```text
turn_end    -> complete
speaking    -> incomplete
backchannel -> backchannel
idle/noidle/uncertain -> abstain (count as incorrect in closed-set metrics)
```

This is a **repository inference**, not an upstream FastTurn recipe. An
alternative temporal policy may use probabilities across frames, but its
parameters must be calibrated away from the public FastTurn test collection
and persisted in the result manifest.

X2-Turn also has no native binary interruption output. The official controller
implements barge-in, while the bot is speaking, when frame argmax is either
`speaking` or `turn_end`; `noidle` alone does not interrupt and backchannels are
handled separately
([controller policy](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/src/voxtral_realtime/turn/controller.py#L17-L20)).
Using this on SID-Bench is therefore `derived-interruption`, with the exact
mapping and temporal confirmation policy saved in the manifest; it is not a
native interruption probability.

## 10. Training/evaluation overlap and leakage audit

The paper discloses Stage 2 training on approximately 126 hours from a subset
of the **EasyTurn training set** for Chinese and approximately 249 hours from a
subset of Fisher for English. Evaluation uses EasyTurn Chinese and English
test sets
([paper Section 3.1](https://arxiv.org/html/2608.10878v2#S3.SS1)). It does not
claim that FastTurn Test Set was used for X2-Turn training, and the code/model
release contains no FastTurn training manifest or example identifiers.

There is nevertheless no auditable proof that X2-Turn and the public FastTurn
Test Set are disjoint. X2-Turn does not release the selected EasyTurn training
IDs or audio hashes, and the public FastTurn release does not expose original
conversation IDs sufficient to compare source recordings. The correct status
is therefore:

```text
declared FastTurn use: none
verified exact overlap: not found
provable disjointness: unavailable
contamination risk: unknown, must be disclosed
```

This uncertainty is stronger for publication-grade claims than for an
engineering baseline. It does not justify declaring known contamination.

The pinned FastTurn transcripts contain terminal gold-label tags for real
examples, as documented in [`FASTTURN_PROTOCOL.md`](FASTTURN_PROTOCOL.md).
X2-Turn does not require them and generates ASR internally. Passing FastTurn's
`txt`, state, path-derived category, source sample rate, or synthetic/real flag
to this adapter would be label leakage. Only waveform samples and their source
sample rate for resampling may cross the model seam.

## 11. Smallest safe smoke test

The lowest-risk functional check uses the official synthetic fixture
`turn-demo/assets/sample_en.wav` from the pinned code commit. It is 53,868
samples of mono signed 16-bit PCM at 16 kHz (3.36675 s), generated by the
project with FFmpeg `flite`, and is not benchmark data
([fixture provenance](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/turn-demo/assets/README.md)).

Preparation requirements:

1. use a dedicated `uv` environment built from the pinned source checkout;
2. export `HF_HOME=/mnt/md0/user_wayne/.hf_cache`;
3. download/verify the pinned Hub revision and the
   `model.safetensors` SHA-256 above;
4. use one NVIDIA GPU with at least 24 GB free VRAM and BF16;
5. load `AutoProcessor` and `load_mtp_checkpoint` with the full Hub revision,
   then call `.eval()` under inference mode.

Functional assertions, based on the official quick-start expectation:

- `frame_ms == 80`;
- every frame interval is monotonic and exactly 80 ms;
- every probability vector has six finite non-negative values summing to one
  within numerical tolerance;
- argmax ID/name agrees with the fixed table above;
- transcript is close to “hello can you tell me what the weather is like
  today”;
- output contains roughly 53 frames, normally including `speaking`, followed
  by `turn_end`, then `idle`. Upstream explicitly allows counts to differ by a
  frame or two across GPUs/library versions
  ([official expected result](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/README.md#expected-result)).

This check validates model loading and output semantics only. It does not
validate causal streaming. A second streaming smoke must start the pinned vLLM
overlay, stream the same fixture in 1,280-sample chunks, and assert that both
`transcription.delta` and `turn.delta` arrive before final completion
([official vLLM smoke procedure](https://github.com/X-Square-Robot/X2-Turn/blob/3609a0fd11b4a1dc2b2629e95538bbe0860da96d/voxtral-realtime/integrations/vllm/examples/voxtral_mtp/README.md)).

Do not start with FastTurn examples: that would mix installation validation,
dataset mapping, and benchmark scoring and could encourage tuning against the
public test set.

## 12. Required result-manifest fields

At minimum, an X2-Turn run must add these fields to the generic manifest:

```json
{
  "model": "x-square-robot/X2-Turn-4B-0812",
  "model_revision": "d188a8b86058ece5e293124a278a1a6256af81fa",
  "source_revision": "3609a0fd11b4a1dc2b2629e95538bbe0860da96d",
  "base_model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
  "native_sample_rate": 16000,
  "frame_interval_ms": 80,
  "target_delay_ms": 480,
  "frontend_lookahead_ms": 2.5,
  "frontend_lookback_ms": 52.5,
  "inference_backend": "transformers-two-pass | patched-vllm-streaming",
  "evaluation_mode": "offline-two-pass | native-streaming",
  "signal_type": "native-turn-head | derived-state",
  "utterance_reduction": "last-non-idle",
  "state_mapping_revision": "...",
  "uncertain_policy": "abstain",
  "transcript_input": "model-generated-only",
  "fastturn_train_overlap": "unknown-not-auditable",
  "declared_license": "Apache-2.0",
  "weight_sha256": "75c7fd12babb6d7ffebf6d0cb86388ff5e20b9a1e9ed079c45eaed62ef6a9117"
}
```

The adapter must also preserve raw six-way frame probabilities and ASR output,
even when the primary artifact contains one derived clip label. This makes a
future mapping correction possible without rerunning a 10.47 GB model.

## 13. Open issues before leaderboard inclusion

- The release does not provide a FastTurn scorer or FastTurn predictions.
- FastTurn train/test disjointness cannot be audited from published IDs.
- `uncertain` documentation and executable inference disagree.
- The local reference path is two-pass and must not be used for native
  streaming latency claims.
- Only the patched vLLM path provides incremental `turn.delta`; maintaining a
  fork pinned to one vLLM commit increases reproduction cost.
- The paper's word-relative latency and `turn-eval` conversational latency are
  different quantities.
- Native four-state FastTurn evaluation is impossible because Wait is absent.
- The checkpoint `NOTICE` preserves a training-rights verification caveat.

Until these issues are resolved, X2-Turn is best reported as a high-capacity
Chinese/English streaming reference with native frame signals and explicitly
derived FastTurn clip decisions, not as a drop-in native FastTurn four-class
model.
