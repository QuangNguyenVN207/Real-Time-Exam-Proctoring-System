# Whisper audio module — reproducible P1 package

## Scope

The module has two learned components with different roles:

1. `vinai/PhoWhisper-small` is used as a pretrained Vietnamese speech-to-text model. This project does not fine-tune PhoWhisper.
2. `vinai/phobert-base-v2` is fine-tuned by this project to classify each transcript as `Normal` (0) or `Cheating` (1).

The final alert is produced by `DecisionFusionService`: a `medium`/`high` keyword risk triggers an alert, otherwise PhoBERT triggers when the probability of `Cheating` reaches the configured threshold (default `0.60`).

## Source-of-truth artifacts

- Fine-tuned model: `phobert/weights/model.safetensors`
- Model and tokenizer configuration: `phobert/weights/config.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, `added_tokens.json`
- Best training checkpoint log: `phobert/weights/checkpoint-235/trainer_state.json`
- Last training checkpoint log: `phobert/weights/checkpoint-423/trainer_state.json`
- Training dataset: `phobert/dataset/cheating.csv`
- Training code: `phobert/train_phobert.py`
- Reproducible validation and figures: `phobert/evaluate_p1.py`
- End-to-end labeled-audio evaluation: `tests/test_pipeline.py`
- Optional ASR WER/CER evaluation when VIVOS is provided: `evaluate_vivos.py`

The run was configured for at most 15 epochs and stopped early after epoch 9. The final log is therefore checkpoint step 423, not 432. The log marks step 235 at epoch 5 as best. The selected root `model.safetensors` is the recovered step-235 weight; direct evaluation reproduces its logged weighted validation loss, accuracy, precision, recall and F1.

`optimizer.pt` is not required for inference, evaluation or this P1 release. It is only required together with the scheduler and RNG state to resume training with the exact optimizer state.

## Install

Use Python 3.12 in a clean environment from the repository root:

```powershell
python -m venv .venv-audio-p1
.venv-audio-p1\Scripts\python.exe -m pip install -r backend\ai_services\whisper\requirements-p1.txt
```

PyTorch is pinned to a CPU wheel in `requirements-p1.txt` for a portable baseline. A CUDA PyTorch build may be substituted on a compatible NVIDIA machine without changing the model files.

The release ZIP contains the complete pretrained `vinai/PhoWhisper-small`
snapshot at revision `a86b604c346caf7148c37512eafe783a16420adb` under
`runtime_models/phowhisper-small`. Runtime defaults to that local directory and
does not download a model. If the directory is missing or incomplete, startup
fails instead of falling back to a remote or newer model.

Verify all package files, checksums, runtime settings, PhoBERT, and PhoWhisper
with no network access:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv-audio-p1\Scripts\python.exe -m backend.ai_services.whisper.verify_release
```

Use `--integrity-only` when only file/config verification is required and a CPU
forward pass would be too expensive.

## Reproduce PhoBERT validation and figures

This command does not train the model:

```powershell
.venv-audio-p1\Scripts\python.exe -m backend.ai_services.whisper.phobert.evaluate_p1
```

Outputs are written to `backend/ai_services/whisper/outputs/p1_phobert/`:

- `summary.json`: metrics, hashes, versions and protocol
- `validation_predictions.csv`: row-level probabilities and errors
- `threshold_sweep.csv`: accuracy, precision, recall and F1 from threshold 0.00 to 1.00
- `epoch_metrics.csv`: values read directly from `trainer_state.json`
- `training_curves.png`: train/validation loss and validation metrics
- `threshold_sweep.png`: threshold sensitivity
- `roc_pr_curves.png`: ROC and precision–recall curves
- `confusion_matrix.png`: confusion matrix at the selected validation threshold
- `report_snippet_vi.md`: concise, report-ready wording with limitations

The evaluator exactly reconstructs the original split in `train_phobert.py`: `test_size=0.15`, `random_state=42`, stratified by label. The threshold with maximum validation F1 is reported as a descriptive tuning result, not a locked-test estimate.

## Reproduce the end-to-end audio sample evaluation

```powershell
.venv-audio-p1\Scripts\python.exe -m backend.ai_services.whisper.tests.test_pipeline
```

Outputs are written to `backend/ai_services/whisper/outputs/p1_audio_pipeline/`. The JSON report contains both all-file metrics and primary metrics after removing byte-identical audio duplicates by SHA-256.

The measured sample result in this package is 40 unique audio hashes (21 cheating, 19 normal): TP=21, TN=19, FP=0, FN=0, with 100% successful processing coverage. The two duplicate paths are retained in the prediction table but excluded from the primary metric. At the selected runtime threshold 0.60, the component ablation is:

| Component | Precision | Recall | F1 |
|---|---:|---:|---:|
| Keyword rules | 1.000 | 0.714 | 0.833 |
| PhoBERT | 1.000 | 1.000 | 1.000 |
| Fusion | 1.000 | 1.000 | 1.000 |

The audio output folder includes:

- `pipeline_results.json` and `pipeline_predictions.csv`: configuration, predictions, transcripts and metrics
- `pipeline_metrics.png`: raw versus SHA-256-deduplicated metrics
- `component_ablation.csv` and `component_ablation.png`: keyword-only, PhoBERT-only and fused results
- `score_distribution.png`: PhoBERT probability separation by true label
- `pipeline_threshold_sweep.csv` and `pipeline_threshold_sweep.png`: sensitivity across thresholds
- `report_snippet_vi.md`: concise report-ready Vietnamese wording and limitations

The perfect score is a functional result on a small, easy sample collection, not an independent generalization estimate. Its broad F1 plateau (0.02–0.99) is evidence that these samples are strongly separated, not evidence that every threshold in this interval will generalize. The configured threshold therefore remains 0.60.

For a separate ASR-only benchmark, provide a VIVOS test folder containing `prompts.txt` and `waves/<speaker>/<audio_id>.wav`:

```powershell
.venv-audio-p1\Scripts\python.exe -m backend.ai_services.whisper.evaluate_vivos --vivos-test-dir D:\path\to\vivos\test
```

This evaluator reports corpus WER and CER directly. It does not present `1 - WER` as “accuracy”.

## Integrity and reporting limits

- Do not report the old hard-coded `75.25%` VIVOS accuracy figure as a measured result. The pushed `figure.py` contains constants rather than reading an evaluation output.
- A WER result requires VIVOS (or another ASR dataset) audio plus reference transcripts. The bundled labeled samples only provide behavior labels, so they support end-to-end alert metrics but not WER.
- The text dataset is small and contains exact duplicate sentences. `evaluate_p1.py` reports both the original validation protocol and a sensitivity result after excluding validation rows that also occur verbatim in train.
- The sample audio folder contains byte-identical duplicate files. The end-to-end report exposes both raw and deduplicated metrics.
- Do not call the threshold-tuned validation result an independent or locked test. A larger speaker-separated test set is still required for a generalization claim.
- The runtime threshold remains `0.60` unless the newly generated sweep provides enough independent evidence to change it. A validation optimum is evidence of sensitivity, not proof that the same threshold generalizes.

## Build the release archive

Before packaging, place the pinned PhoWhisper snapshot in
`backend/ai_services/whisper/runtime_models/phowhisper-small`. It must contain
the model, processor, tokenizer, and generation configuration files. Then,
after generating both output folders:

```powershell
.venv-audio-p1\Scripts\python.exe -m backend.ai_services.whisper.package_p1
```

The packager validates required files, canonicalizes text files to LF, records
SHA-256 checksums, and creates a deterministic ZIP under
`backend/ai_services/whisper/releases/`. The archive contains both PhoBERT and
PhoWhisper and can be verified fully offline.
