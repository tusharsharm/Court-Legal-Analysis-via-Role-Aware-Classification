# Court-Legal-Analysis-via-Role-Aware-Classification
Final Year MSc Project
# CLARA — Court Legal Analysis via Role-Aware Classification

**CLARA** is a two-stage NLP pipeline designed for automated analysis of legal judgments. It combines rhetorical role labeling with intelligent summarization to make court documents more accessible and structured.

---

## Pipeline Overview
```
Raw Legal Document
        │
        ▼
┌─────────────────────┐
│   Step 1: CLARA-RR  │  ← Rhetorical Role Classification
│  (Role Labelling)   │     Labels each sentence by its legal function
└─────────────────────┘
        │
        ▼  (labelled sentences)
┌─────────────────────┐
│  Step 2: CLARA-SUMM │  ← Legal Summarization
│  (Summary Pipeline) │     Extracts + abstracts a structured summary
└─────────────────────┘
        │
        ▼
Structured Legal Summary (by role section)
```

---

## Step 1 — CLARA-RR: Rhetorical Role Labelling

**File:** `rhetoric_role_labelling_pipeline.ipynb`

### Purpose

CLARA-RR classifies every sentence in a legal judgment into one of four rhetorical roles. This enables structured downstream analysis by identifying *what function* each sentence serves in the judgment.

### Dataset

| Property | Detail |
|---|---|
| Dataset | LegalSeg: Unlocking the Structure of Indian Legal Judgments Through Rhetorical Role Classification
Shubham Kumar Nigam, Tanmay Dubey, Govind Sharma, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya |
| Documents | 7,120 |
| Sentences | ~1.4 million |
| Original Labels | 7 (LegalSeg schema) |
| Remapped Labels | 4 (CLARA-RR schema) |
| Baseline | Hier-BiLSTM-CRF, Macro-F1 = 0.77 |
| Hardware | Kaggle T4 (16 GB VRAM) |

### Label Remapping

The LegalSeg dataset uses 7 labels. CLARA-RR remaps these to 4 meaningful legal roles and drops the uninformative `None` class:

| LegalSeg Label | CLARA-RR Label |
|---|---|
| Facts | Facts |
| Reasoning | Ratio |
| Decision | Ruling_Present |
| AoP (Argument of Petitioner) | Argument |
| AoR (Argument of Respondent) | Argument |
| Issue | Argument |
| None | *(Dropped)* |

### Model Architecture

**Base Encoder:** `law-ai/InLegalBERT` — a BERT model pretrained on Indian legal text.

The full CLARA-RR model (`CLARAModel`) stacks several purpose-built modules on top of the encoder:

- **Context-Aware Tokenization** — Each sentence is encoded with a ±3 sentence context window, joined via `[SEP]` tokens, giving the model local discourse context.
- **RoleAwareAttention** — Multi-head self-attention over all sentences in a document, with optional role-conditioned embeddings for a second refinement pass.
- **SegmentTransformer** — A document-level Transformer that captures longer-range dependencies across sentence sequences.
- **PositionPriorBias** — An MLP that maps each sentence's relative position (0.0 → 1.0) to per-class emission biases, encoding the structural tendency of legal roles to appear at certain document positions.
- **LexiconEmissionBias** — An MLP over 12 hand-crafted legal lexicon features (keyword signals per role), adding domain-informed emission biases.
- **LearnedEmissionScaler** — Per-class learnable temperature scaling applied to final emissions.
- **PseudoLabelRefiner** — Iterative soft label refinement using attention over hidden states.
- **ShiftHead** — Binary head for detecting rhetorical role transitions between adjacent sentences.
- **CRF (Conditional Random Field)** — Final decoding layer that enforces valid label sequences. Structural priors are injected into the CRF transition matrix (e.g., `Ruling_Present` is discouraged from appearing mid-document).

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Max Token Length | 192 |
| Context Window | ±3 sentences |
| Batch Size | 2 documents |
| Gradient Accumulation | 16 steps (effective batch = 32) |
| Epochs | 10 |
| Learning Rate (BERT) | 2e-5 |
| Learning Rate (Heads) | 1e-4 |
| Warmup Ratio | 0.1 |
| BERT Layers Frozen | 0–7 (first 3 epochs), then progressively unfrozen |
| Optimizer | AdamW |
| Precision | Mixed (AMP with GradScaler) |

### Loss Functions

CLARA-RR uses a composite loss designed to handle class imbalance and structural properties of legal text:

| Loss Component | Weight | Purpose |
|---|---|---|
| Focal Loss | 0.5 | Down-weights easy examples; handles imbalance |
| Shift Loss | 0.3 | Penalizes missed rhetorical role transitions |
| Contrastive Loss (SupCon) | 0.3 | Pulls same-role sentence embeddings together |
| Ruling Confusion Loss | 0.4 | Adds extra penalty for misclassifying `Ruling_Present` |

Class weights are computed using **Effective Number Weighting** (Cui et al., 2019, β = 0.9999).

### Pipeline Cells Summary

| Cell | Description |
|---|---|
| Cell 0 | Header and task definition |
| Cell 1 | Package installation |
| Cell 2 | Imports, seeds, device setup |
| Cell 3 | Dataset and model paths (Kaggle) |
| Cell 4 | `Config` class — all hyperparameters and label maps |
| Cell 5 | Class weight computation |
| Cell 6 | CSV loading and LegalSeg → CLARA-RR label remapping |
| Cell 7 | Document list builder (groups sentences by `doc_id`) |
| Cell 8 | Tokenizer + `LegalSegDataset` (context encoding, lexicon features, positions) |
| Cell 9 | Model components: `PositionPriorBias`, `LexiconEmissionBias`, `RoleAwareAttention` |
| Cell 10 | Full `CLARAModel` assembly with CRF and structural priors |
| Cell 11 | BERT layer freeze/unfreeze helpers |
| Cell 12 | Training loop with AMP, gradient accumulation, progressive unfreezing |
| Cell 13 | Main training execution: data loading, model init, optimizer, scheduler |
| Cell 14 | Best model reload and final test evaluation |
| Cell 15 | Save results JSON (val F1, test F1, epoch history, baseline delta) |
| Cell 16 | Notes on 4-class vs 7-class schema differences |
| Cells 17–24 | Visualization suite: dataset distribution, training curves, confusion matrix, per-class P/R/F1, baseline comparison, position-role distribution, summary dashboard |
| Cell 25 | Output file manifest and final result printout |

### Outputs

| File | Description |
|---|---|
| `clara_legalseg_best.pt` | Best model checkpoint (by validation Macro-F1) |
| `clara_legalseg_results.json` | Full results summary (metrics, config, baseline delta) |
| `viz_01_dataset_distribution.png` | Label distribution before and after remapping |
| `viz_02_training_curves.png` | Training loss and validation F1 per epoch |
| `viz_03_confusion_matrix.png` | Test set confusion matrix (counts + recall-normalised) |
| `viz_04_perclass_metrics.png` | Per-class Precision / Recall / F1 bar chart |
| `viz_05_baseline_comparison.png` | CLARA-RR vs all LegalSeg baselines |
| `viz_06_position_distribution.png` | Role distribution by document position |
| `viz_07_dashboard.png` | Full results dashboard |

### Baseline Comparison

CLARA-RR is benchmarked against all systems reported in the LegalSeg paper:

| Model | Macro-F1 |
|---|---|
| MTL | 0.37 |
| RhetoricLLaMA | 0.09 |
| InLegalBERT (single sentence) | 0.49 |
| InLegalBERT (2-sentence context) | 0.55 |
| InLegalBERT (3-sentence context) | 0.58 |
| GNN | 0.54 |
| ToInLegalBERT | 0.62 |
| **Hier-BiLSTM-CRF (LegalSeg paper)** | **0.77** |
| **CLARA-RR v5 (ours)** | **TBD (target: > 0.77)** |

---

## Step 2 — CLARA-SUMM: Legal Summarization Pipeline

**File:** `summary-pipeline.ipynb`

### Purpose

CLARA-SUMM takes the rhetorical role-labelled sentences produced by CLARA-RR and generates structured legal summaries. It operates in two stages — extractive selection followed by abstractive rewriting — organized by rhetorical role sections.

### Dataset

| Property | Detail |
|---|---|
| Dataset | Malik et al. dataset (100 documents) |
| Format | `.txt` / `.tsv` files (tab-separated sentence + label) |
| Path | Kaggle: `legal-segmentation-100docs/` |

### NER Integration

Before summarization, a fine-tuned legal Named Entity Recognition (NER) model is applied to extract legal entities from the text. The pipeline supports two model formats:

- **spaCy model** (detected via `meta.json`) — loaded with optional GPU acceleration
- **HuggingFace model** (detected via `config.json`) — loaded as a `pipeline("ner", aggregation_strategy="simple")`

Long texts are chunked into 200-word segments before NER inference to avoid token length issues.

**NER Model Path:** `/kaggle/input/models/tusharsharma2911/ner-model-finetuned/`

### Stage 1 — Extractive Summarization (`LegalPegasusExtractiveSummarizer`)

Uses a Pegasus-based encoder to score and select the most informative sentences from each rhetorical role section.

**Key mechanisms:**

- **Adaptive Summary Length** — The target compression ratio is dynamically computed based on document length using a piecewise linear function:
  - ≤77 sentences: `summary% = -0.2444 × n + 40.54`
  - 78–122 sentences: `summary% = -0.1013 × n + 29.53`
  - >122 sentences: `summary% = -0.006 × n + 17.90`
  - Minimum: 10% regardless of length

- **Trigram Blocking** — Removes redundant sentences by checking trigram overlap against already-selected sentences. A sentence is dropped if its 3-gram overlap with any existing selected sentence exceeds the `redundancy_threshold` (0.3).

- **Pegasus Embeddings** — Sentence representations are extracted from the Pegasus encoder's last hidden state using mean pooling over token embeddings, enabling semantic similarity computation.

### Stage 2 — Abstractive Summarization (`AbstractiveSummarizer`)

Uses `google/pegasus-xsum` to generate a fluent, condensed abstractive summary for each rhetorical role section independently.

**Process:**
1. Extractive summary sentences are grouped by their rhetorical role label (Facts, Argument, Ratio, Ruling_Present).
2. Each group's sentences are concatenated into a section text.
3. Pegasus generates an abstractive summary per section (max 150 tokens, min 30 tokens, 4 beams, length penalty 2.0).
4. Section summaries are returned as a structured dictionary keyed by role label.

### Validation & Quality Metrics (`ValidationMetrics`)

Each processed document is evaluated on three dimensions:

| Metric | Description | Threshold |
|---|---|---|
| **Label Coverage** | Fraction of original rhetorical roles present in the summary | ≥ 0.8 |
| **Redundancy Score** | Mean pairwise cosine similarity between summary sentence embeddings | ≤ 0.3 |
| **Section Length** | Character count per role section in the summary | ≥ 50 chars |
| **Overall Quality** | Composite score combining coverage and redundancy | > 0.6 |

For abstractive validation, coherence, content preservation, and length ratio are additionally tracked.

### Pipeline Cells Summary

| Cell | Description |
|---|---|
| Cell 0 | HuggingFace token setup (`HF_TOKEN`) |
| Cell 1 | Environment setup: installs packages, sets Kaggle paths for NER model and dataset |
| Cell 2 | `CustomLegalNER` class — auto-detects and loads spaCy or HuggingFace NER model with chunked inference |
| Cell 3 | Instantiates the NER model |
| Cell 4 | `load_kaggle_dataset()` — reads `.txt`/`.tsv` files, parses sentence-label pairs, builds document list |
| Cell 5 | `LegalPegasusExtractiveSummarizer` — adaptive length, trigram blocking, Pegasus-based scoring |
| Cell 6 | `AbstractiveSummarizer` — role-section abstractive generation via `google/pegasus-xsum` |
| Cell 7 | `get_pegasus_embeddings()` — mean-pooled encoder embeddings for similarity computation |
| Cell 8 | `ValidationMetrics` class — label coverage, redundancy, section length, coherence evaluation |
| Cell 9 | Instantiates the validator |
| Cell 10 | `generate_visualizations()` — 6-panel validation dashboard |
| Cell 11 | `print_pipeline_summary()` — aggregated statistics across all documents |

### Outputs

| Output | Description |
|---|---|
| Visualization figure | 2×3 panel dashboard of validation metrics across all documents |
| Per-document results | Extractive + abstractive summaries with validation scores |
| Pipeline summary printout | Aggregate stats: success rate, avg quality, redundancy, coverage, coherence |

---

## Environment & Requirements

Both notebooks are designed to run on **Kaggle** with a **T4 GPU (16 GB VRAM)**.

### Common Dependencies
```
transformers
datasets
torch
pytorch-crf
scikit-learn
pandas
numpy
tqdm
spacy
sentence-transformers
matplotlib
seaborn
```

### CLARA-RR Specific
```
law-ai/InLegalBERT   # Base encoder (HuggingFace)
torchcrf             # CRF decoding layer
```

### CLARA-SUMM Specific
```
google/pegasus-xsum  # Abstractive summarization
t5-base              # Extractive scoring backbone
Custom Legal NER     # Fine-tuned NER model (spaCy or HuggingFace)
```

---

## Reproducibility Notes

- All random seeds are fixed at `SEED = 42` (Python, NumPy, PyTorch).
- CLARA-RR uses mixed precision training (`torch.cuda.amp`) for T4 memory efficiency.
- BERT layers 0–7 are frozen for the first 3 epochs, then progressively unfrozen at 0.5× LR from epoch 4 onward.
- CLARA-SUMM expects labelled sentence input; it is intended to consume output from CLARA-RR.

---

## Notes on Schema Differences

CLARA-RR on LegalSeg is reported as a **4-class experiment**, distinct from experiments on the Malik dataset which use a **7-class schema**:

| Experiment | Schema | Docs | Classes | Macro-F1 |
|---|---|---|---|---|
| Malik dataset | Full 7-class | 100 | 7 | 0.6254 |
| LegalSeg (CLARA-RR) | Reduced 4-class | 7,120 | 4 | TBD |
