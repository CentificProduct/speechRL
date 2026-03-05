# Emotion-GSRM

**Generative Speech Reward Model for Emotional Expressivity**

Emotion-GSRM is a generative reward model that evaluates and improves emotional expressivity in speech language models. It extends the [GSRM architecture](https://arxiv.org/abs/2602.13891) (Shen et al., 2026) from naturalness prediction to emotion-specific evaluation using a two-stage chain-of-thought (CoT) synthesis pipeline, 7-dimension emotion rubric, and public emotion speech datasets.

> **Status:** Research implementation targeting Interspeech 2026.

---

## Key Contributions

- **First generative reward model** specifically targeting emotional expressivity in speech
- **Novel acoustic feature pipeline** augmenting GSRM's prosodic features with voice quality (jitter, shimmer, HNR, spectral tilt), temporal (speech rate, pause patterns), and formant dynamics (F1/F2, vowel space area)
- **7-dimension emotion rubric** with anchored 1–5 Likert scales: Emotional Intensity, Appropriateness, Consistency, Valence Accuracy, Arousal Accuracy, Transition Smoothness, and Overall Quality
- **Reproducible pipeline** built entirely on public datasets (IEMOCAP, MSP-Podcast, RAVDESS, MELD), addressing the reproducibility gap in the original GSRM

---

## Architecture

```
Raw Audio ──► Acoustic Feature Extraction ──► Speaker Normalization ──► Discretization
                                                                              │
                                                                              ▼
                                                              ┌───────────────────────┐
Transcript + Context ──────────────────────────────────────►  │  Stage 1: Evidence Log │
                                                              │  (GPT-4o teacher)      │
                                                              └───────────┬───────────┘
                                                                          │
                                                                          ▼
                                                              ┌───────────────────────┐
                                            Oracle Scores ──► │  Stage 2: Judgment CoT │
                                                              │  (GPT-4o teacher)      │
                                                              └───────────┬───────────┘
                                                                          │
                                                                          ▼
                                                              ┌───────────────────────┐
                                                              │  SFT Fine-tuning       │
                                                              │  Qwen2.5-Omni-7B      │
                                                              └───────────┬───────────┘
                                                                          │
                                                                          ▼
                                                              ┌───────────────────────┐
                                              Raw Audio ────► │  Inference (K=16 avg)  │ ──► 7-D Scores
                                                              └───────────────────────┘
```

**Training:** GPT-4o generates CoT reasoning from acoustic features + transcripts. Qwen2.5-Omni-7B learns to replicate this reasoning directly from raw audio.

**Inference:** The student model takes raw audio, generates K=16 independent CoT responses at temperature 1.0 / top-p 0.6, and averages the parsed scores.

---

## Project Structure

```
speechRL/
├── features/
│   ├── __init__.py
│   └── acoustic.py              # Acoustic feature extraction (748 lines)
│                                  #   Prosodic (GSRM-retained), Voice Quality,
│                                  #   Temporal, Formant features + normalization
├── rubric/
│   └── __init__.py              # 7-dimension evaluation rubric (628 lines)
│                                  #   Scale anchors, prompt builders, score parsing
├── dataloader/
│   ├── __init__.py
│   ├── base.py                  # EmotionSample, BaseEmotionDataset, label maps
│   ├── iemocap.py               # IEMOCAP loader (12hrs, categorical + VAD)
│   ├── msp_podcast.py           # MSP-Podcast loader (100hrs, continuous VAD)
│   ├── ravdess.py               # RAVDESS loader (7K recordings, controlled)
│   └── meld.py                  # MELD loader (13K utterances, multi-party)
├── cot/
│   ├── __init__.py
│   ├── evidence.py              # Stage 1: per-dimension evidence generation
│   ├── judgment.py              # Stage 2: global CoT judgment synthesis
│   └── pipeline.py              # End-to-end synthesis orchestration
├── sft/
│   ├── __init__.py
│   ├── config.py                # TrainingConfig (all proposal hyperparameters)
│   ├── dataset.py               # SFT dataset formatting + collator
│   ├── trainer.py               # SWIFT / HuggingFace training backends
│   └── inference.py             # K-sample averaging inference engine
├── trail
|   ├── step0_validate.py            # Pipeline validation (no API calls)
|   ├── step1_synthesize.py          # CoT data synthesis runner
|   ├── step2_train.py               # SFT training runner
|   ├── step3_inference.py           # Inference & evaluation runner
|   ├── requirements.txt
└── README.md
```

**Total: ~6,700 lines of Python across 23 files.**

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For training (install separately):
pip install ms-swift          # SWIFT framework
pip install peft accelerate   # LoRA + distributed training
```

### 2. Download RAVDESS (Free, No License)

Download from [Zenodo](https://zenodo.org/record/1188976) and extract:

```
~/datasets/RAVDESS/
└── Audio_Speech_Actors_01-24/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

### 3. Validate the Pipeline

```bash
python step0_validate.py --ravdess_dir ~/datasets/RAVDESS/Audio_Speech_Actors_01-24/
```

This runs all components end-to-end on real audio with zero API calls. If it prints `ALL VALIDATION PASSED`, every dependency and data path is correct.

### 4. Synthesize Training Data

```bash
export OPENAI_API_KEY=sk-your-key-here

# Start small (20 samples, ~$1-2 in API costs)
python step1_synthesize.py \
    --ravdess_dir ~/datasets/RAVDESS/Audio_Speech_Actors_01-24/ \
    --max_samples 20 \
    --batch \
    --output_dir ./ravdess_cot_output
```

### 5. Fine-tune the Model

```bash
# Dry run first (prepares data, prints command, no GPU needed)
python step2_train.py \
    --data_path ./ravdess_cot_output/sft_training_data.jsonl \
    --dry_run

# Run training (needs GPU)
python step2_train.py \
    --data_path ./ravdess_cot_output/sft_training_data.jsonl \
    --num_gpus 1 \
    --epochs 10
```

### 6. Run Inference

```bash
python step3_inference.py \
    --model_path ./emotion_gsrm_checkpoints \
    --ravdess_dir ~/datasets/RAVDESS/Audio_Speech_Actors_01-24/ \
    --k 16
```

---

## Datasets

| Dataset | Size | Labels | Role | Access |
|---------|------|--------|------|--------|
| IEMOCAP | 12 hrs, 10K utt | Categorical + VAD (1-5) | Primary training | [USC license](https://sail.usc.edu/iemocap/iemocap_release.htm) |
| MSP-Podcast | 100+ hrs, 60K+ seg | Continuous VAD (1-7), 5+ annotators | Supplementary + OOD test | [UT Dallas academic license](https://www.lab-msp.com/MSP/MSP-Podcast.html) |
| RAVDESS | 7,356 recordings | 8 emotions, 2 intensities | Controlled ablation | [Open (Zenodo)](https://zenodo.org/record/1188976) |
| MELD | 13K+ utterances | 7 emotions + sentiment | OOD evaluation | [Open (GitHub)](https://github.com/declare-lab/MELD) |

---

## Acoustic Features

### Retained from GSRM
Vowel-level prosodic features extracted via Parselmouth with forced alignment (or energy-based fallback): pitch level, pitch variation, pitch slope, intensity level, intensity variation, and vowel duration.

### New for Emotion-GSRM

**Voice Quality** (via Parselmouth):
- **Jitter** — cycle-to-cycle F0 perturbation; increases under high arousal (anger, excitement)
- **Shimmer** — cycle-to-cycle amplitude perturbation; correlates with vocal strain
- **HNR** — harmonics-to-noise ratio; low = breathy/rough (sadness), high = clear (neutral)
- **Spectral tilt** — log-power slope; steep negative = sadness, flatter = anger/happiness

**Temporal** (via librosa):
- **Speech rate** — syllables/second via onset detection
- **Speech rate variation** — local rate std across 1s windows
- **Pause duration/frequency** — energy-based VAD pause detection

**Formant** (via Parselmouth/Praat):
- **F1/F2 means** — first and second formant frequencies
- **Vowel space area** — triangular VSA from F1/F2 extremes; expanded = clear emotion, compressed = reduced speech

All features undergo speaker-level z-normalization and quantile-based discretization into ordinal categories (very_low → very_high), following the GSRM methodology.

---

## Evaluation Rubric

Seven dimensions, each rated 1–5 with anchored descriptors:

| Dimension | Measures | Key Acoustic Indicators |
|-----------|----------|------------------------|
| Emotional Intensity | Flat → vivid expression | Pitch variation, jitter/shimmer, vowel space |
| Emotional Appropriateness | Context fit | Pitch contour, speech rate, spectral tilt |
| Emotional Consistency | Within-utterance coherence | Cross-segment prosodic variation |
| Valence Accuracy | Positive/negative alignment | Pitch level, HNR, formants |
| Arousal Accuracy | Energy calibration | Speech rate, intensity, jitter/shimmer |
| Transition Smoothness | Cross-turn naturalness | Pitch slope, pause patterns |
| Overall Emotional Quality | Holistic judgment | All features |

---

## Training Configuration

All hyperparameters follow the proposal specification:

| Parameter | Value |
|-----------|-------|
| Student model | Qwen2.5-Omni-7B |
| Teacher model | GPT-4o |
| Learning rate | 2×10⁻⁵ |
| Effective batch size | 32 |
| Epochs | 10 |
| LoRA rank / alpha | 64 / 128 |
| LR scheduler | Cosine with 5% warmup |
| Precision | BFloat16 |
| Inference K | 16 samples averaged |
| Inference temperature | 1.0 |
| Inference top-p | 0.6 |
| Framework | SWIFT (or HuggingFace fallback) |
| Target training samples | 5K–7K |

---

## What Changed from GSRM

| Aspect | GSRM | Emotion-GSRM |
|--------|------|---------------|
| Evaluation target | Single naturalness score | 7 emotion sub-dimensions |
| Acoustic features | 6 prosodic features | 6 prosodic + 11 new (voice quality, temporal, formant) |
| Evidence structure | Naturalness cues | Emotion-specific: context, cues, alignment, strengths, issues |
| Data | Proprietary custom collection | 4 public datasets |
| Conversational context | Not used | 2–3 preceding turns |
| Student model | Qwen2-Audio-7B | Qwen2.5-Omni-7B |
| Additional metrics | — | UAR, confusion matrices, systematic ablations |

---

## API Usage

### Programmatic Usage

```python
from emotion_gsrm.data.ravdess import RAVDESSDataset
from emotion_gsrm.data.base import DatasetSplit
from emotion_gsrm.features.acoustic import AcousticFeatureExtractor, FeatureNormalizer, format_features_for_prompt

# Load data
dataset = RAVDESSDataset("./RAVDESS", split=DatasetSplit.TRAIN).load()
sample = dataset[0]

# Extract features
extractor = AcousticFeatureExtractor()
features = extractor.extract(sample.audio_path, sample.utterance_id, sample.speaker_id)

# Normalize and format for prompts
normalizer = FeatureNormalizer()
normalizer.fit_speaker(sample.speaker_id, [features])
normalizer.fit_quantiles([features])
normalized = normalizer.normalize(features)
discretized = normalizer.discretize(normalized)
prompt_text = format_features_for_prompt(features, discretized)
```

### RAVDESS Controlled Pairs (for Ablation)

```python
from emotion_gsrm.data.base import CategoricalEmotion

dataset = RAVDESSDataset("./RAVDESS", split=DatasetSplit.TRAIN).load()

# Get matched pairs: same actor, same statement, different emotions
pairs = dataset.get_controlled_pairs(CategoricalEmotion.HAPPY, CategoricalEmotion.SAD)
# Get intensity pairs: same emotion, normal vs strong
intensity_pairs = dataset.get_intensity_pairs(CategoricalEmotion.ANGRY)
```

---

## Dependencies

| Package | Purpose | Required For |
|---------|---------|--------------|
| numpy | Numerical operations | All |
| librosa | Audio I/O, temporal features | Feature extraction |
| parselmouth | Praat bindings (pitch, formants, voice quality) | Feature extraction |
| soundfile | Audio file I/O backend | Feature extraction |
| openai | GPT-4o API calls | CoT synthesis (Step 1) |
| transformers | Model loading, tokenization | Training & inference |
| torch | PyTorch backend | Training & inference |
| peft | LoRA adapters | Training |
| ms-swift | SWIFT training framework | Training (recommended) |
| datasets | HuggingFace datasets | Training (optional) |
| wandb | Experiment tracking | Training (optional) |

---

## References

- Shen, M., et al. "GSRM: Generative Speech Reward Model for Speech RLHF." *arXiv:2602.13891* (2026).
- Busso, C., et al. "IEMOCAP: Interactive emotional dyadic motion capture database." *Language Resources and Evaluation* 42.4 (2008).
- Busso, C., et al. "The MSP-Podcast Corpus." *arXiv:2509.09791* (2025).
- Livingstone, S. R. & Russo, F. A. "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)." *PLoS ONE* 13.5 (2018).
- Poria, S., et al. "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations." *ACL* (2019).

---

## License

Research use. Dataset access is subject to individual dataset licenses (see Datasets section).

## Author

Atik Faysal — Department of Chemistry & Biochemistry, Rowan University
Advisor: Dr. Thomas M. Keck