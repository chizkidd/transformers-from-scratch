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

| Title | Dataset | Description | Notebooks |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) | Multi-Head Attention, Positional Encodings, and Label Smoothing from scratch. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](attention-is-all-you-need.ipynb) |

#### Encoder-only: Representation Learning & Classification

| Title | Dataset | Description | Notebooks |
| --- | --- | --- | --- |
| [BERT](https://arxiv.org/abs/1810.04805): Pre-training of Deep Bidirectional Transformers | [SST-2 (GLUE)](https://huggingface.co/datasets/nyu-mll/glue) | Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) for downstream sentiment analysis. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-sentiment-analysis.ipynb) |
| [BERT](https://arxiv.org/abs/1810.04805): Question Answering | [SQuAD v2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | Fine-tuning a pre-trained BERT encoder for extractive question answering. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](bert-question-answering.ipynb) |

#### Decoder-only: Generative Pre-training `GPT` (Autoregressive) & Discriminative Fine-Tuning

| Title | Dataset | Description | Notebooks |
| --- | --- | --- | --- |
| [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Sentiment Classification | [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), [SST-2 (GLUE)](https://huggingface.co/datasets/nyu-mll/glue) | Generative pre-training using a 12-layer decoder-only architecture followed by discriminative fine-tuning. Pre-training corpus: WikiText-2 (BookCorpus proxy). | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-sentiment-analysis.ipynb) |
| [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Natural Language Inference | [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | Fine-tuning for 3-class NLI (entailment/contradiction/neutral) using `[BOS] premise [SEP] hypothesis [CLF]` format with auxiliary LM loss. Pre-training: WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-nli-inference.ipynb) |
| [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Sentence Similarity | [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), [MRPC](https://huggingface.co/datasets/nyu-mll/glue) | Binary paraphrase detection (paraphrase/not) using delimiter token format `[BOS] sentence1 [SEP] sentence2 [CLF]` with discriminative fine-tuning. Pre-training: WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-sentence-similarity.ipynb) |
| [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf): Question Answering | [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), [RACE](https://huggingface.co/datasets/ehovy/race) | Multiple-choice QA with 4 options per question; input format `[BOS] context [SEP] question+answer_i [CLF]`, select highest logit option. Pre-training: WikiText-2 proxy. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt1-question-answering.ipynb) |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): Language Models are Unsupervised Multitask Learners | [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) | Zero-shot task transfer using a larger decoder-only model trained on the WebText dataset. | [![PyTorch](https://img.shields.io/badge/Py-Torch-red)](gpt2.ipynb) |

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