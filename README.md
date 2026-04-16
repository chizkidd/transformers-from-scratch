
# transformers-from-scratch

[![PyTorch](https://img.shields.io/badge/Py-Torch-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![View Notebooks](https://img.shields.io/badge/View-Live%20Notebooks-blue?logo=github)](https://chizkidd.github.io/transformers-from-scratch/)
[![Notebooks Deployment Status](https://github.com/chizkidd/transformers-from-scratch/actions/workflows/deploy-notebooks.yml/badge.svg)](https://github.com/chizkidd/transformers-from-scratch/actions/workflows/deploy-notebooks.yml)

A comprehensive collection of Transformer architectures implemented from scratch in PyTorch — from **"Attention Is All You Need"** to modern LLMs.

> **Note:** Specialized deep-dive into Attention-based architectures. For RNNs/CNNs, see [deep-learning-models](https://github.com/chizkidd/deep-learning-models).

## Why this repository?
First-principles implementations focusing on technical training details:
* **Manual Masking:** Causal/padding masks at tensor level
* **Custom Schedulers:** Noam LR (Learning Rate) scheduler from scratch
* **Numerical Stability:** FP16/AMP training with NaN handling

---

## Performance: RNN vs. Transformer (Multi30k `EN`→`DE`)

| Metric | RNN (LSTM) Baseline | Transformer (From Scratch) | Verdict |
| :--- | :---: | :---: | :--- |
| **BLEU** | 14.94 | **28.74** | +92% |
| **Perplexity** | 42.91 | **6.53** | Superior Certainty |
| **Bits/Character (BPC)** | 5.42 | **2.70** | Precise Alignment  |

>The following results were achieved on the **Multi30k (English-to-German)** translation task, comparing my [RNN/LSTM baseline](https://github.com/chizkidd/deep-learning-models/blob/main/pytorch/rnn/rnn-seq2seq.ipynb) against this repository's Transformer implementation (**AIAYN**).

---

## Model Catalog

### Encoder-Decoder: Neural Machine Translation (Attention Is All You Need)

| Task | Model & Dataset | Format | Classes | Metric | Description | Notebook |
|------|----------------|--------|---------|--------|-------------|----------|
| **1** | **[Transformer](https://arxiv.org/abs/1706.03762)**<br>[[Multi30k](https://huggingface.co/datasets/bentrevett/multi30k)] | Src: `[POS]...[EOS]` → Enc<br>Tgt: `[POS]...[EOS]` → Dec | N/A | BLEU | From-scratch Transformer: Multi-Head Attention, positional encodings, residuals, label smoothing for EN-DE translation. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](attention-is-all-you-need.ipynb) |

### Encoder-only: Representation Learning (BERT)

| Task | Model & Dataset | Format | Classes | Metric | Description | Notebook |
|------|----------------|--------|---------|--------|-------------|----------|
| **BERT-1** | **[BERT: Sentiment](https://arxiv.org/abs/1810.04805)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SST-2](https://huggingface.co/datasets/nyu-mll/glue)] | `[CLS] sentence [SEP]` | 2 (Neg/Pos) | Acc | BERT fine-tuned for binary sentiment via `[CLS]` token; pre-trained on WikiText-2 (MLM+NSP). | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-sentiment-analysis.ipynb) |
| **BERT-2** | **[BERT: QA](https://arxiv.org/abs/1810.04805)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SQuAD v2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)] | `[CLS] question [SEP] context [SEP]` | Span (start/end) | EM + F1 | BERT fine-tuned for extractive QA predicting answer span positions; handles unanswerable questions. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-question-answering.ipynb) |

### Decoder-only: Generative Pre-training (GPT)

#### GPT-1 (Fine-tuning)

| Task | Model & Dataset | Format | Classes | Metric | Description | Notebook |
|------|----------------|--------|---------|--------|-------------|----------|
| **1** | **[GPT-1: Sentiment](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SST-2](https://huggingface.co/datasets/nyu-mll/glue)] | `[BOS] sentence [CLF]` | 2 (Neg/Pos) | Acc | GPT-1 fine-tuned for binary sentiment via `[CLF]` token + auxiliary LM loss (λ=0.5). | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-sentiment-classification.ipynb) |
| **2** | **[GPT-1: NLI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SNLI](https://huggingface.co/datasets/stanfordnlp/snli)] | `[BOS] premise [SEP] hypothesis [CLF]` | 3 (E/N/C) | Acc + F1-m | GPT-1 fine-tuned for 3-class NLI with delimiter tokens + auxiliary LM regularization. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-natural-language-inference.ipynb) |
| **3** | **[GPT-1: Paraphrase](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [MRPC](https://huggingface.co/datasets/nyu-mll/glue)] | `[BOS] sent₁ [SEP] sent₂ [CLF]` | 2 (Para/Not) | Acc + F1-b | GPT-1 fine-tuned for paraphrase detection using sentence-pair format + auxiliary LM loss. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-sentence-similarity.ipynb) |
| **4** | **[GPT-1: QA](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>[[WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [RACE](https://huggingface.co/datasets/ehovy/race)] | `[BOS] article [SEP] question+opts [CLF]` | 4 (A/B/C/D) | Acc | GPT-1 fine-tuned for multiple-choice QA: select max-logit option from 4 candidates. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-multiple-choice-QA.ipynb) |

> **GPT-1 Notes:** All tasks use `[CLF]` token → linear head → softmax + auxiliary LM loss (λ=0.5). SNLI bridges single-sentence (SST-2) and pair-wise (MRPC) tasks.

#### GPT-2 (Zero-Shot)

| Task | Model & Dataset | Format | Classes | Metric | Description | Notebook |
|------|----------------|--------|---------|--------|-------------|----------|
| **1** | **[GPT-2: Zero-Shot](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**<br>[[WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)] | Task-specific prompts | Varies | Task-dep | WebText pre-trained decoder-only model demonstrating emergent zero-shot capabilities without fine-tuning. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt2.ipynb) |

---

## Installation
```bash
git clone https://github.com/chizkidd/transformers-from-scratch.git
cd transformers-from-scratch
pip install -r requirements.txt
```

## References
* Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Devlin et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
* Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models)
