# transformers-from-scratch

[![PyTorch](https://img.shields.io/badge/Py-Torch-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![View Notebooks](https://img.shields.io/badge/View-Live%20Notebooks-blue?logo=github)](https://chizkidd.github.io/transformers-from-scratch/)
[![Notebooks Deployment Status](https://github.com/chizkidd/transformers-from-scratch/actions/workflows/deploy-notebooks.yml/badge.svg)](https://github.com/chizkidd/transformers-from-scratch/actions/workflows/deploy-notebooks.yml)

A comprehensive, high-performance, collection of Transformer architectures implemented from scratch in PyTorch. This repository tracks the evolution of Attention mechanisms, from the original **"Attention Is All You Need"** paper to modern Large Language Models (LLMs).

> **Note:** This repository is a specialized deep-dive into Attention-based architectures. For my broader research into RNNs, and CNNs, see [deep-learning-models](https://github.com/chizkidd/deep-learning-models).

## Why this repository?
Unlike high-level libraries (Hugging Face), these implementations focus on the **first-principles math** and the technical hurdles of training Transformers:
* **Manual Masking:** Causal and Padding masks implemented at the tensor level.
* **Custom Schedulers:** Full implementation of the Noam Learning Rate scheduler.
* **Numerical Stability:** Robust solutions for `NaN` issues in half-precision (FP16/AMP) training.

---

## Performance Audit: RNN vs. Transformer
The following results were achieved on the **Multi30k (English-to-German)** translation task, comparing my [RNN/LSTM baseline](https://github.com/chizkidd/deep-learning-models/blob/main/pytorch/rnn/rnn-seq2seq.ipynb) against this repository's Transformer implementation (**AIAYN**).

| Metric | RNN (LSTM) Baseline (Seq2Seq) | Transformer (From Scratch) | Verdict |
| :--- | :---: | :---: | :--- |
| **BLEU Score** | 14.94 | **28.74** | **+92% Improvement** |
| **Perplexity (PPL)** | 42.91 | **6.53** | **Superior Certainty** |
| **Bits Per Char (BPC)** | 5.42 | **2.70** | **Precise Alignment** |

---

## Transformers

#### Encoder-Decoder: Neural Machine Translation (NMT)

| Task | Title & Dataset | Task Type | Input Format | Classes | Primary Metric | Description | Notebook |
|------|----------------|-----------|--------------|---------|---------------|-------------|----------|
| **1** | **[Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)**<br>📚 [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) | Neural Machine Translation (Seq2Seq) | Source: `[POS] tokens [EOS]` → Encoder<br>Target: `[POS] tokens [EOS]` → Decoder (teacher forcing) | N/A (generation) | BLEU Score | From-scratch implementation of the original Transformer architecture featuring Multi-Head Attention, positional encodings, residual connections, and label smoothing for English-German translation. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](attention-is-all-you-need.ipynb) |

### Pre-Training & Fine-Tuning

#### <u>Encoder-only</u>: Representation Learning & Classification. 

| Task | Title & Dataset | Task Type | Input Format | Classes | Primary Metric | Description | Notebook |
|------|----------------|-----------|--------------|---------|---------------|-------------|----------|
| **BERT-1** | **[BERT: Sentiment Classification](https://arxiv.org/abs/1810.04805)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SST-2 (GLUE)](https://huggingface.co/datasets/nyu-mll/glue) | Binary Sentiment Classification | `[CLS] sentence [SEP]` | 2 (Neg/Pos) | Accuracy | Fine-tuning pre-trained BERT (Masked LM + Next Sentence Prediction objectives) for binary sentiment analysis using the `[CLS]` token representation. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-sentiment-analysis.ipynb) |
| **BERT-2** | **[BERT: Extractive Question Answering](https://arxiv.org/abs/1810.04805)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SQuAD v2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | Span-based Question Answering | `[CLS] question [SEP] context [SEP]` | N/A (start/end positions) | Exact Match + F1 | Fine-tuning BERT for extractive QA by predicting answer span start/end token positions within context. Handles unanswerable questions via no-answer logit. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-question-answering.ipynb) |

#### <u>Decoder-only</u>: (Autoregressive) Generative Pre-training  & (Discriminative) Fine-Tuning

##### **GPT-1**

| Task | Title & Dataset | Task Type | Input Format | Classes | Primary Metric | Description | Notebook |
|------|----------------|-----------|--------------|---------|---------------|-------------|----------|
| **1** | **[GPT-1: Sentiment Classification](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SST-2 (GLUE)](https://huggingface.co/datasets/nyu-mll/glue) | Binary Sentiment Classification | `[BOS] sentence [CLF]` | 2 (Neg/Pos) | Accuracy | Fine-tuning for binary sentiment (positive/negative) with auxiliary LM loss. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-task1-sst2.ipynb) |
| **2** | **[GPT-1: Natural Language Inference](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | 3-Class NLI | `[BOS] premise [SEP] hypothesis [CLF]` | 3 (Entail/Neutral/Contradict) | Accuracy + F1-macro | Fine-tuning for entailment reasoning with delimiter tokens and auxiliary LM regularization. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-task2-snli.ipynb) |
| **3** | **[GPT-1: Paraphrase Detection](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [MRPC (GLUE)](https://huggingface.co/datasets/nyu-mll/glue) | Binary Sentence Similarity | `[BOS] sent₁ [SEP] sent₂ [CLF]` | 2 (Paraphrase/Not) | Accuracy + F1-binary | Discriminative fine-tuning for paraphrase detection using sentence-pair format with auxiliary LM loss. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-task3-mrpc.ipynb) |
| **4** | **[GPT-1: Reading Comprehension](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)**<br>📚 [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) + [RACE](https://huggingface.co/datasets/ehovy/race) | Multiple-Choice QA | `[BOS] article [SEP] question+options [CLF]` | 4 (A/B/C/D) | Accuracy | Multiple-choice QA where model selects highest-logit option from 4 candidates. Input concatenates context, question, and options. Pre-training on WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-task4-race.ipynb) |

>**Key Notes:**
>- **Task 2 (SNLI)** is the 3-class NLI task that bridges single-sentence (SST-2) and sentence-pair (MRPC) classification.
>- All tasks follow the same GPT-1 fine-tuning pattern: `[CLF]` token at the end → linear head → softmax.
>- Auxiliary LM loss ($\lambda = 0.5$) is used across all tasks per the original paper.


##### **GPT-2**

| Task | Title & Dataset | Task Type | Input Format | Classes | Primary Metric | Description | Notebook |
|------|----------------|-----------|--------------|---------|---------------|-------------|----------|
| **1** | **[GPT-2: Zero-Shot Multitask Learning](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**<br>📚 [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | Zero-Shot Transfer | Task-specific prompts (no fine-tuning) | Varies by task | Task-dependent | Larger decoder-only model trained on WebText; demonstrates emergent zero-shot capabilities across NLP tasks without task-specific fine-tuning. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt2-multitask.ipynb) |

---

## Installation & Usage
```bash
git clone [https://github.com/chizkidd/transformers-from-scratch.git](https://github.com/chizkidd/transformers-from-scratch.git)
cd transformers-from-scratch
pip install -r requirements.txt
```


## References
* Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
* Devlin, J. et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805).
* Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models).