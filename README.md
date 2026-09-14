# 🩺 Medical Knowledge RAG Assistant with Unsloth Dynamic 4-bit Llama 3.2

An evidence-grounded medical Retrieval-Augmented Generation (RAG) system that combines authoritative WHO documents, semantic retrieval, Cross-Encoder reranking, confidence-based evidence filtering, and memory-efficient LLM inference using Unsloth Dynamic 4-bit quantization.

> **Educational project only — not a substitute for professional medical advice, diagnosis, or treatment.**

---

## 🚀 Project Overview

Large Language Models can generate fluent responses even when the underlying information is unsupported. This project explores a safer approach for domain-specific medical question answering by combining retrieval with guarded generation.

The system first searches an indexed collection of WHO medical documents for relevant evidence. Retrieved passages are filtered and reranked before being supplied to a quantized Llama 3.2 language model.

If the available evidence is insufficient, the system **does not invoke the LLM to generate a medical answer**.

The current prototype focuses primarily on:

- Hypertension
- Diabetes

---

## 🎯 Project Objectives

The project was designed to:

- build a complete Retrieval-Augmented Generation pipeline;
- use authoritative domain-specific medical documents;
- reduce unsupported LLM responses through evidence grounding;
- improve retrieval quality using Cross-Encoder reranking;
- identify unsupported questions before generation;
- run a capable language model efficiently on limited GPU hardware;
- provide transparent source and page information;
- evaluate retrieval and refusal behavior systematically;
- provide an interactive user-facing demonstration.

---

## 🧠 System Architecture

The complete pipeline follows this workflow:

```text
User Question
      ↓
Sentence Transformer Embedding
      ↓
FAISS Semantic Retrieval
      ↓
Similarity Filtering
      ↓
Cross-Encoder Reranking
      ↓
Confidence-Based Evidence Selection
      ↓
Is sufficient evidence available?
      ↓
 ┌───────────────┬───────────────────┐
 │ Yes           │ No                │
 ↓               ↓
Dynamic 4-bit    Controlled refusal
Llama 3.2
      ↓
Evidence-Grounded Answer
      ↓
Source + Page + Retrieval Metrics
🔍 Key Features
Retrieval-Augmented Generation

Medical documents are divided into overlapping chunks, embedded using a Sentence Transformer, and indexed using FAISS for semantic retrieval.

Cross-Encoder Reranking

Initial FAISS candidates are reranked using a Cross-Encoder model so that the most directly relevant evidence is prioritized.

Confidence-Based Evidence Filtering

Low-confidence passages are removed before generation. This prevents unrelated retrieved text from automatically reaching the LLM.

Guarded Generation

If no sufficiently relevant evidence remains after retrieval and reranking, the system skips LLM generation and returns a controlled insufficient-evidence response.

Source Transparency

Generated responses are accompanied by:

source document;
page number;
FAISS similarity score;
reranking confidence.
Memory-Efficient LLM Inference

The generation model uses:

Llama 3.2 3B Instruct with Unsloth Dynamic 4-bit quantization

The model required approximately 2.24 GB of allocated GPU memory after loading on a Tesla T4 during the experiment.

Interactive Demo

A Gradio interface allows users to enter questions and inspect:

evidence-grounded responses;
retrieved evidence;
retrieval confidence;
generation status;
response time.
🛠️ Technology Stack
Component	Technology
Language Model	Llama 3.2 3B Instruct
Quantization	Unsloth Dynamic 4-bit
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Search	FAISS
Reranking	Cross-Encoder MS MARCO MiniLM
Document Processing	PyPDF
Interface	Gradio
Framework	PyTorch / Transformers
Environment	Google Colab
GPU	NVIDIA Tesla T4
Knowledge Sources	World Health Organization documents
📚 Knowledge Base

The prototype currently indexes authoritative WHO documents covering:

hypertension;
diabetes.

The documents are processed page by page so that source and page metadata remain attached to each retrieval chunk.

The current knowledge base is intentionally limited. Questions about unrelated medical topics should therefore trigger the system's evidence-sufficiency protection rather than receive unsupported answers.

📊 Evaluation

The system was evaluated using a designed set of 10 questions containing:

supported hypertension questions;
supported diabetes questions;
unsupported medical questions.
Results
Evaluation Metric	Result
Evaluation questions	10
Source-routing accuracy	100%
Generation/refusal behavior accuracy	100%
Supported-query generation rate	100%
Unsupported-query refusal accuracy	100%
Average response time	4.82 seconds

These results describe performance only on the designed 10-question evaluation set. They should not be interpreted as general medical or clinical accuracy.

🖥️ Demo Results
Supported Query — Hypertension

The system retrieves relevant WHO hypertension evidence and generates an evidence-grounded response.

Supported Query — Diabetes

The system correctly routes a diabetes-related question to the WHO diabetes document and generates a grounded answer.

Unsupported Query — Asthma

Asthma information is not contained in the current indexed knowledge base.

Instead of allowing the language model to guess, the system detects insufficient evidence and skips generation.

🧪 Example Behavior
Supported Question
How can hypertension be prevented?

The system:

retrieves WHO hypertension passages;
filters weak candidates;
reranks the remaining evidence;
selects the strongest passage;
generates an evidence-grounded response.
Unsupported Question
What are the symptoms and treatment of asthma?

Because no sufficiently relevant evidence exists in the indexed documents:

LLM generation used: False

The system returns an insufficient-evidence message instead of producing an unsupported medical response.

📂 Repository Structure
medical-rag-unsloth-dynamic4bit/
│
├── Medical_RAG_Unsloth_Dynamic4bit_Month2_Task3.ipynb
│
├── README.md
│
└── assets/
    ├── demo_hypertension_supported.png
    ├── demo_diabetes_supported.png
    └── demo_asthma_refusal.png
▶️ Running the Project

The complete implementation is available in:

Medical_RAG_Unsloth_Dynamic4bit_Month2_Task3.ipynb

Recommended environment:

Google Colab
NVIDIA T4 GPU or equivalent

The notebook covers the complete workflow from dependency installation and document processing to retrieval, reranking, evaluation, and the interactive Gradio demo.

⚠️ Limitations

This project is a prototype and has several important limitations:

the indexed medical knowledge base is small;
only a limited number of medical topics are currently covered;
retrieval thresholds were selected empirically for the current document collection;
the evaluation set contains only 10 designed questions;
the application has not undergone clinical validation;
retrieval quality may change when additional documents are introduced;
the system should not be used for medical diagnosis or treatment decisions.
🛡️ Safety Design

A key design goal of this project is to reduce unsupported generation.

The system therefore uses:

semantic similarity filtering;
Cross-Encoder reranking;
evidence-confidence selection;
generation refusal when evidence is insufficient;
explicit source tracking;
a medical-use disclaimer.

This approach does not eliminate hallucination risk, but it introduces explicit safeguards before the LLM is allowed to generate a response.

🔮 Future Improvements

Planned extensions include:

expanding the authoritative medical knowledge base;
hybrid semantic + keyword retrieval;
larger and more diverse evaluation sets;
automated retrieval-quality benchmarking;
direct source-document citation links;
persistent web deployment;
conversational memory with evidence grounding;
improved chunking strategies;
retrieval parameter calibration;
additional hallucination and factuality evaluation.
💡 What I Learned

This project provided practical experience with:

Retrieval-Augmented Generation;
vector embeddings and semantic search;
FAISS indexing;
document chunking;
metadata-aware retrieval;
Cross-Encoder reranking;
confidence-based retrieval filtering;
LLM inference;
Dynamic 4-bit quantization;
GPU memory management;
RAG evaluation;
safety-oriented generation control;
Gradio application development.
👩‍💻 Author

Kaunain Gul Khalid

BS Artificial Intelligence
University of Malakand

Developed as part of a Generative AI internship project and extended into a portfolio-focused implementation of guarded, evidence-grounded RAG.
