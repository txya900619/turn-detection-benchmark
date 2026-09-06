from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from time import monotonic_ns

import numpy as np

from turn_eval.core.types import (
    AudioData,
    ClipRequest,
    ExecutionMode,
    ModelDescriptor,
    StatePrediction,
    TurnState,
)

EASY_TURN_REPO_ID = "ASLP-lab/Easy-Turn"
EASY_TURN_REVISION = "7d1f8b06776b575ac22e2fd4aa7913da2f4b8b98"
EASY_TURN_CHECKPOINT_FILE = "checkpoint.pt"
EASY_TURN_CHECKPOINT_SHA256 = "1cb535bb0ffe3ffa0ac941bae4898aeb87adef7b88bb58f665af08eded6f6070"
EASY_TURN_CHECKPOINT_BYTES = 3_451_801_684
EASY_TURN_SOURCE_URL = "https://github.com/ASLP-lab/Easy-Turn.git"
EASY_TURN_SOURCE_REVISION = "fe175eaa8442c9a0834acf29f8640ee74c2aceca"
EASY_TURN_QWEN_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EASY_TURN_QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EASY_TURN_SAMPLE_RATE = 16_000
EASY_TURN_MAX_AUDIO_S = 30.0
EASY_TURN_MAX_AUDIO_SAMPLES = int(EASY_TURN_SAMPLE_RATE * EASY_TURN_MAX_AUDIO_S)
EASY_TURN_PROMPT_KEY = "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>"
EASY_TURN_NATIVE_LABELS = tuple(state.value for state in TurnState)

_TAG_PATTERN = re.compile(r"<\s*(complete|incomplete|backchannel|wait)\s*>", re.IGNORECASE)
_TERMINAL_SUFFIX_PATTERN = re.compile(
    r"^(?:[\s。！？!?，,；;：:、]|<\|(?:endoftext|im_end)\|>)*$",
    re.IGNORECASE,
)


def parse_easy_turn_output(output: str) -> TurnState | None:
    """Parse exactly one terminal native Easy Turn tag, otherwise abstain."""
    matches = list(_TAG_PATTERN.finditer(output))
    if len(matches) != 1:
        return None
    match = matches[0]
    if not _TERMINAL_SUFFIX_PATTERN.fullmatch(output[match.end() :]):
        return None
    return TurnState(match.group(1).lower())


def _default_source_cache() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "turn-eval-sources" / "easy-turn"
    return Path("~/.cache/huggingface/turn-eval-sources/easy-turn").expanduser()


def _validate_source_root(root: Path) -> Path:
    resolved = root.resolve()
    required = (
        resolved / "wenet/llm_asr/init_llmasr.py",
        resolved / "examples/wenetspeech/whisper/conf/train.yaml",
        resolved / "examples/wenetspeech/whisper/conf/prompt.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"invalid Easy Turn source root; missing {missing}")
    return resolved


def prepare_easy_turn_source(
    source_root: Path | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    """Materialize the pinned upstream source and return its Easy_Turn package root."""
    if source_root is not None:
        return _validate_source_root(source_root)
    checkout = _default_source_cache() / EASY_TURN_SOURCE_REVISION
    package_root = checkout / "Easy_Turn"
    if package_root.is_dir():
        return _validate_source_root(package_root)
    if local_files_only:
        raise FileNotFoundError(f"pinned Easy Turn source is not cached at {checkout}")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="easy-turn-source-", dir=checkout.parent))
    try:
        subprocess.run(
            ["git", "init", "--quiet", str(staging)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "remote", "add", "origin", EASY_TURN_SOURCE_URL],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                EASY_TURN_SOURCE_REVISION,
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(staging), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            check=True,
        )
        staging.replace(checkout)
    except Exception:
        # The staging directory contains only this attempted checkout.
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate_source_root(package_root)


class EasyTurnFastTurnAdapter:
    """Pinned upstream Easy Turn audio-to-native-four-state adapter."""

    def __init__(
        self,
        output_labels: tuple[TurnState, ...],
        *,
        revision: str = EASY_TURN_REVISION,
        device: str = "cuda:0",
        source_root: Path | None = None,
        local_files_only: bool = False,
        infer: Callable[[np.ndarray], str] | None = None,
        infer_batch: Callable[[tuple[np.ndarray, ...]], tuple[str, ...]] | None = None,
    ) -> None:
        if not output_labels or len(output_labels) != len(set(output_labels)):
            raise ValueError("Easy Turn output labels must be non-empty and unique")
        if any(label not in TurnState for label in output_labels):
            raise ValueError("Easy Turn received an unsupported output label")
        self._output_labels = output_labels
        self._device = device
        self._prompt: str | None = None
        if infer is not None and infer_batch is not None:
            raise ValueError("provide only one Easy Turn inference override")
        if infer_batch is not None:
            self._infer_batch = infer_batch
        elif infer is not None:
            self._infer_batch = lambda samples: tuple(infer(sample) for sample in samples)
        else:
            self._infer_batch = self._load_runtime(
                revision=revision,
                device=device,
                source_root=source_root,
                local_files_only=local_files_only,
            )
        self._descriptor = ModelDescriptor(
            name="easy-turn",
            revision=revision,
            native_sample_rate=EASY_TURN_SAMPLE_RATE,
            execution_modes=frozenset({ExecutionMode.OFFLINE}),
            output_signals=frozenset({"turn_state"}),
            supported_languages=frozenset({"zh"}),
            artifact_checksum=f"sha256:{EASY_TURN_CHECKPOINT_SHA256}",
            artifact_size_bytes=EASY_TURN_CHECKPOINT_BYTES,
            model_license="Apache-2.0",
            code_license="Apache-2.0",
            known_training_datasets=("Easy-Turn-Trainset",),
        )

    def _load_runtime(
        self,
        *,
        revision: str,
        device: str,
        source_root: Path | None,
        local_files_only: bool,
    ) -> Callable[[tuple[np.ndarray, ...]], tuple[str, ...]]:
        try:
            import librosa
            import torch
            import yaml
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as error:  # pragma: no cover - isolated environment
            raise RuntimeError(
                "Easy Turn dependencies are missing; use environments/easy_turn"
            ) from error

        package_root = prepare_easy_turn_source(
            source_root,
            local_files_only=local_files_only,
        )
        checkpoint = Path(
            hf_hub_download(
                EASY_TURN_REPO_ID,
                EASY_TURN_CHECKPOINT_FILE,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
        qwen_root = Path(
            snapshot_download(
                EASY_TURN_QWEN_REPO_ID,
                revision=EASY_TURN_QWEN_REVISION,
                local_files_only=local_files_only,
            )
        )
        config_path = package_root / "examples/wenetspeech/whisper/conf/train.yaml"
        prompt_path = package_root / "examples/wenetspeech/whisper/conf/prompt.yaml"
        with config_path.open(encoding="utf-8") as stream:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
        configs["llm_path"] = str(qwen_root)
        configs["tokenizer_conf"]["llm_path"] = str(qwen_root)

        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        if "wenet" not in sys.modules:
            # Upstream wenet/__init__.py imports the entire ASR CLI and optional
            # Whisper stack. Easy Turn inference needs only the pinned package
            # modules below, so register the package path without that eager import.
            wenet_package = types.ModuleType("wenet")
            wenet_package.__path__ = [str(package_root / "wenet")]
            wenet_package.__package__ = "wenet"
            sys.modules["wenet"] = wenet_package
        from wenet.llm_asr.init_llmasr import init_llmasr

        class Args:
            pass

        model, _ = init_llmasr(Args(), configs, is_inference=True)
        state = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        del state
        model.to(torch.device(device)).eval()

        with prompt_path.open(encoding="utf-8") as stream:
            prompts = yaml.load(stream, Loader=yaml.FullLoader)
        self._prompt = prompts[EASY_TURN_PROMPT_KEY][0]

        def extract_features(samples: np.ndarray):
            waveform = torch.from_numpy(np.array(samples, dtype=np.float32, copy=True))
            window = torch.hann_window(400)
            stft = torch.stft(
                waveform,
                400,
                160,
                window=window,
                return_complex=True,
            )
            magnitudes = stft[..., :-1].abs() ** 2
            filters = torch.from_numpy(
                librosa.filters.mel(sr=EASY_TURN_SAMPLE_RATE, n_fft=400, n_mels=80)
            )
            mel_spec = filters @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            return ((log_spec + 4.0) / 4.0).transpose(0, 1)

        def run(samples_batch: tuple[np.ndarray, ...]) -> tuple[str, ...]:
            if not samples_batch:
                raise ValueError("Easy Turn cannot infer an empty batch")
            features = tuple(extract_features(samples) for samples in samples_batch)
            if len(features) == 1:
                feature = features[0].unsqueeze(0).to(device)
                length = torch.tensor(
                    [features[0].shape[0]],
                    dtype=torch.int32,
                    device=device,
                )
                with torch.inference_mode():
                    output = model.generate(feature, length, self._prompt)
                if device.startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))
                return tuple(output)

            lengths = torch.tensor(
                [item.shape[0] for item in features],
                dtype=torch.int32,
                device=device,
            )
            padded_features = torch.nn.utils.rnn.pad_sequence(
                features,
                batch_first=True,
            ).to(device)
            with torch.inference_mode():
                speech_embeds, speech_masks = model.get_embedding_from_wav(
                    padded_features,
                    lengths,
                )
                batch_count = len(features)
                bos = model.speech_token_emded(
                    torch.zeros((batch_count, 1), dtype=torch.long, device=device)
                )
                eos = model.speech_token_emded(
                    torch.ones((batch_count, 1), dtype=torch.long, device=device)
                )
                prompt_ids = model.tokenizer(
                    [self._prompt],
                    return_tensors="pt",
                )["input_ids"].to(device)
                prompt_embeds = model.embed_tokens(prompt_ids)[0]
                sequences = []
                for index in range(batch_count):
                    valid_speech = speech_embeds[index][speech_masks[index].bool()]
                    sequences.append(
                        torch.cat((prompt_embeds, bos[index], valid_speech, eos[index]), dim=0)
                    )
                maximum_length = max(sequence.shape[0] for sequence in sequences)
                hidden_size = sequences[0].shape[-1]
                embeds = torch.zeros(
                    (batch_count, maximum_length, hidden_size),
                    dtype=sequences[0].dtype,
                    device=device,
                )
                attention_mask = torch.zeros(
                    (batch_count, maximum_length),
                    dtype=torch.long,
                    device=device,
                )
                for index, sequence in enumerate(sequences):
                    embeds[index, -sequence.shape[0] :] = sequence
                    attention_mask[index, -sequence.shape[0] :] = 1
                if model.embed_tokens.weight.dtype in {torch.float16, torch.bfloat16}:
                    embeds = embeds.to(torch.bfloat16)
                    attention_mask = attention_mask.to(torch.bfloat16)
                outputs = model.llama_model.generate(
                    inputs_embeds=embeds,
                    max_new_tokens=model.max_length,
                    num_beams=model.num_beams,
                    do_sample=model.do_sample,
                    min_length=model.min_length,
                    top_p=model.top_p,
                    top_k=model.top_k,
                    repetition_penalty=model.repetition_penalty,
                    length_penalty=model.length_penalty,
                    temperature=model.temperature,
                    attention_mask=attention_mask,
                    eos_token_id=151643,
                    pad_token_id=151643,
                    use_cache=True,
                )
                trimmed_outputs = []
                for tokens in outputs:
                    eos_positions = (tokens == 151643).nonzero(as_tuple=False)
                    end = int(eos_positions[0].item()) + 1 if len(eos_positions) else len(tokens)
                    trimmed_outputs.append(tokens[:end])
                output = tuple(
                    model.tokenizer.decode(
                        tokens,
                        add_special_tokens=False,
                        skip_special_tokens=True,
                    )
                    for tokens in trimmed_outputs
                )
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            return tuple(output)

        return run

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def resample_audio(self, audio: AudioData) -> AudioData:
        """Match upstream torchaudio Resample preprocessing."""
        if audio.sample_rate == EASY_TURN_SAMPLE_RATE:
            return audio
        try:
            import torch
            import torchaudio
        except ImportError as error:  # pragma: no cover - isolated environment
            raise RuntimeError("Easy Turn requires torch and torchaudio") from error
        writable_samples = np.array(audio.samples, dtype=np.float32, copy=True)
        waveform = torch.from_numpy(writable_samples).unsqueeze(0)
        samples = torchaudio.transforms.Resample(
            orig_freq=audio.sample_rate,
            new_freq=EASY_TURN_SAMPLE_RATE,
        )(waveform)
        return AudioData(samples.squeeze(0).numpy(), EASY_TURN_SAMPLE_RATE)

    def predict_clip(self, request: ClipRequest) -> StatePrediction:
        if request.user_audio.sample_rate != EASY_TURN_SAMPLE_RATE:
            raise ValueError("Easy Turn requires 16 kHz audio")
        return self.predict_clips((request,))[0]

    def predict_clips(self, requests: tuple[ClipRequest, ...]) -> tuple[StatePrediction, ...]:
        if not requests:
            raise ValueError("Easy Turn cannot predict an empty request batch")
        if any(request.user_audio.sample_rate != EASY_TURN_SAMPLE_RATE for request in requests):
            raise ValueError("Easy Turn requires 16 kHz audio")
        supported_requests = tuple(
            request
            for request in requests
            if request.user_audio.samples.size <= EASY_TURN_MAX_AUDIO_SAMPLES
        )
        raw_outputs = (
            self._infer_batch(tuple(request.user_audio.samples for request in supported_requests))
            if supported_requests
            else ()
        )
        if len(raw_outputs) != len(supported_requests):
            raise ValueError("Easy Turn backend returned the wrong number of outputs")
        output_iter = iter(raw_outputs)
        predictions = []
        for request in requests:
            if request.user_audio.samples.size > EASY_TURN_MAX_AUDIO_SAMPLES:
                predictions.append(self._build_too_long_prediction(request))
            else:
                predictions.append(self._build_prediction(request, next(output_iter)))
        return tuple(predictions)

    def _build_too_long_prediction(self, request: ClipRequest) -> StatePrediction:
        duration_s = request.user_audio.samples.size / EASY_TURN_SAMPLE_RATE
        return StatePrediction(
            example_id=request.example_id,
            predicted_label=None,
            scores={label.value: 0.0 for label in self._output_labels},
            model_output={
                "raw_text": None,
                "native_label": None,
                "score_semantics": "hard-decoded-one-hot-not-probabilities",
                "invalid_handling": "abstain-closed-set-incorrect",
                "abstention_reason": "input-too-long",
                "input_duration_s": duration_s,
                "maximum_input_duration_s": EASY_TURN_MAX_AUDIO_S,
            },
            emitted_at_ns=monotonic_ns(),
        )

    def _build_prediction(self, request: ClipRequest, raw_output: str) -> StatePrediction:
        native_label = parse_easy_turn_output(raw_output)
        predicted = native_label if native_label in self._output_labels else None
        scores = {
            label.value: float(label is predicted) if predicted is not None else 0.0
            for label in self._output_labels
        }
        return StatePrediction(
            example_id=request.example_id,
            predicted_label=predicted,
            scores=scores,
            model_output={
                "raw_text": raw_output,
                "native_label": None if native_label is None else native_label.value,
                "score_semantics": "hard-decoded-one-hot-not-probabilities",
                "invalid_handling": "abstain-closed-set-incorrect",
            },
            emitted_at_ns=monotonic_ns(),
        )
