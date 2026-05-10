# Empirical Comparison of IO, BIO, and BIOES Labeling Schemes for Transformer-Based Named Entity Recognition

NLP in Python, Final Project

Authors: **Ilya Mochalov**, **Goryunov Kirill**

## 1. Project description

This project empirically compares three sequence-labeling schemes — **IO**, **BIO**, and **BIOES** — for transformer-based Named Entity Recognition. All three are evaluated on the CoNLL-2003 English NER benchmark using a fine-tuned `bert-base-cased` model under identical training conditions, isolating the effect of the labeling scheme itself.

The project reports overall precision/recall/F1, per-entity F1 scores, token-level and tag-level confusion matrices, training-loss dynamics, and qualitative per-token inference examples that illustrate the theoretical strengths and weaknesses of each scheme.

The full discussion, related work, and detailed result analysis are provided in the accompanying paper: [`NLP_final_project.pdf`](NLP_final_project.pdf).

## 2. Repository structure

```
.
├── README.md                       — this file
├── NLP_final_project.pdf           — ACL two-column report
├── nlp-final-project-demo.ipynb    — main notebook
└── requirements.txt                — Python dependencies
```

## 3. Data

* **Dataset:** CoNLL-2003 English NER, loaded from the Hugging Face Hub mirror [`tomaarsen/conll2003`](https://huggingface.co/datasets/tomaarsen/conll2003).

* **Tag inventory (original BIO):** 9 tags — `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`.

* **Derived inventories:**
  * **IO** (5 tags): `B-X` is rewritten to `I-X`; `O` is kept. Cannot mark adjacent same-type entities.
  * **BIOES** (17 tags): a 1-token lookahead converts `B-X` followed by `I-X` into `B-X`, isolated `B-X` into `S-X`, the last `I-X` of a span into `E-X`, and inner `I-X` stays as `I-X`.

* No external data is required, the dataset is downloaded automatically by `datasets.load_dataset` on first run.

## 4. Model

* `bert-base-cased` (110M parameters) via `AutoModelForTokenClassification` with a freshly initialized 5 / 9 / 17-class classification head, one model per labeling scheme.

## 5. Reproducing the results

### Environment

Tested with **Python 3.10–3.12**, PyTorch ≥ 2.1, and the Hugging Face stack. CUDA, Apple MPS, and CPU back-ends are all supported (a CUDA GPU is strongly recommended).

```bash
git clone <repo-url>
cd <repo-folder>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the notebook

```bash
jupyter notebook nlp-final-project-demo.ipynb
```

Then **Cell / Run All**. The notebook is structured top-to-bottom and reproduces every figure and table in the report:

1. Imports, dataset loading, tokenizer setup, and `bert-base-cased` smoke test.
2. **BIO** baseline: subword-aware label alignment (`add_labels`), `Trainer` fine-tuning, span-level evaluation with `seqeval`.
3. **IO** branch: BIO -> IO conversion (`B-X -> I-X`), re-tokenization (`add_labels_io`), fine-tuning of an independent 5-class head.
4. **BIOES** branch: BIO -> BIOES conversion with one-token lookahead (`bio_seq_to_bioes`), re-tokenization (`add_labels_bioes`), fine-tuning of an independent 17-class head.
5. Comparative evaluation: overall and per-entity F1 tables, token-level entity-type confusion matrices for all three schemes, the full 17×17 BIOES tag-level confusion matrix, and joint training-loss curves.
6. Qualitative inference: per-token predictions on selected CoNLL-2003 test sentences and on hand-crafted free-text sentences via `pipeline("ner", aggregation_strategy="simple")`.

## 6. Report

The final report (4–6 pages, ACL two-column format) is included as [`NLP_final_project.pdf`](NLP_final_project.pdf) in the repository root. It contains the full motivation, related work, methodology, and discussion of the results summarized above.
