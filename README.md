# Adaptive Context DAGs (AC-DAGs)
### Graph-Guided Multi-Modal Context Injection for Long-Horizon LLM Reasoning

> **Submitted to:** ReThesis: AI Research Blitz — Ascent Techfest  
> **Venue:** Scaler School of Technology, Bengaluru | May 2026  
> **Track:** AI / ML Research

---

## Abstract

Large Language Models suffer from **context dilution** in long-horizon reasoning tasks — when irrelevant or redundant information floods the context window, output quality degrades significantly. Standard Retrieval-Augmented Generation (RAG) systems retrieve flat, unordered chunk lists without modeling inter-chunk dependencies, producing incoherent reasoning chains. This worsens with multi-modal inputs (text + images) where modality alignment remains unsolved.

We propose **Adaptive Context DAGs (AC-DAGs)** — a novel inference-time pipeline that structures retrieved context as a Directed Acyclic Graph, learns dependency weights between chunks via a Graph Neural Network, and uses a Reinforcement Learning policy to traverse and prune the graph before injecting a fused context vector into any instruction-tuned LLM.

---

## Architecture

```
Query Input
     │
     ▼
┌─────────────────────────────────────────────┐
│  1. Encoder Layer                           │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │  SBERT   │  │   CLIP   │  │Timestamp  │ │
│  │  (text)  │  │ (images) │  │  enc.     │ │
│  └──────────┘  └──────────┘  └───────────┘ │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  2. DAG Builder                             │
│  Chunks → Nodes │ GraphSAGE GNN → Edges    │
│  Cosine similarity initializes edge weights │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  3. RL Policy (PPO)                         │
│  State: node emb + query emb               │
│  Action: next node selection + pruning      │
│  Reward: downstream LLM task success        │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  4. Aggregator                              │
│  Cross-modal multi-head attention           │
│  Selected path → fused context vector       │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│  5. LLM Interface                           │
│  Fused context → prefix injection           │
│  LLaMA-3 / Mistral (any instruction LLM)   │
└─────────────────────────────────────────────┘
     │
     ▼
  Reasoned LLM Output
```

---

## Core Pseudocode

```python
def inject_context(query, user_mem, history):
    # Stage 1: Retrieve multi-modal chunks
    chunks = retrieve_multi_modal(query)          # Vector + keyword search
    
    # Stage 2: Build DAG with GNN-learned edges
    graph = build_dag(chunks)                     # GraphSAGE edge weights
    
    # Stage 3: RL policy selects optimal path
    path = rl_policy.select_path(graph, query_emb)  # Max expected reasoning gain
    
    # Stage 4: Aggregate via cross-modal attention
    fused_ctx = aggregate(path)                   # Cross-modal attention
    
    # Stage 5: Inject and reason
    return llm_reason(fused_ctx, query)
```

---

## Innovation & Novelty

AC-DAGs introduce **three novel contributions** over vanilla RAG:

| Contribution | What it does | Why it matters |
|---|---|---|
| **Structured DAG context** | Models inter-chunk dependencies explicitly | Flat chunk lists miss relational structure |
| **RL-guided traversal** | Optimizes for reasoning quality, not retrieval similarity | Similarity ≠ usefulness for downstream reasoning |
| **Multi-modal fusion** | Timestamp-aware embeddings handle temporal drift | Text + image alignment is unsolved in RAG |

Unlike LangGraph or LlamaIndex pipelines, AC-DAGs **dynamically prune context at inference time** based on reasoning reward signals — directly solving context dilution in long-horizon planning.

---

## Projected Results

| Benchmark | Metric | Baseline (RAG) | AC-DAG Target | Gain |
|---|---|---|---|---|
| LongBench | Steps to failure | 8 | 16+ | **+2x** |
| HotpotQA (multi-modal) | F1 Score | 0.72 | 0.85 | **+18%** |
| Custom Planner | Success@10 | 45% | 80% | **+35pp** |

> Results are projected from architectural analysis. Full empirical validation is ongoing.

---

## Components & Stack

### Encoder Layer
- **SBERT** (`sentence-transformers`) — dense text embeddings
- **CLIP** (OpenAI) — image embeddings
- **Timestamp-aware positional encodings** — handles temporal modality drift

### DAG Builder
- Graph constructed dynamically per query
- Edges initialized via **cosine similarity**
- Refined by **2-layer GraphSAGE GNN** (PyTorch Geometric)

### RL Policy
- **PPO** (Proximal Policy Optimization) via Stable-Baselines3
- State: `current_node_embedding ⊕ query_embedding`
- Action: next node selection
- Reward: downstream LLM reasoning quality (task success score)

### Aggregator
- **Cross-modal multi-head attention** over selected DAG path
- Produces unified fused context vector

### LLM Interface
- Works with any instruction-tuned LLM: **LLaMA-3**, **Mistral**, etc.
- Fused context injected as a prefix
- Core pipeline: ~200 lines PyTorch

---

## Scalability Design

- **Hierarchical chunking**: coarse-to-fine retrieval reduces graph size
- **LoRA-fine-tuned RL policies**: enables deployment on edge devices
- **FAISS** for fast approximate nearest-neighbor vector search
- **Modality drift mitigation**: timestamp-aware embeddings for temporal misalignment

---

## Tools & Frameworks

```
PyTorch                    Core deep learning framework
Hugging Face Transformers  LLM backbone + tokenizers
sentence-transformers      SBERT text embeddings
CLIP (OpenAI)              Image embeddings
PyTorch Geometric          GraphSAGE GNN implementation
Stable-Baselines3          PPO reinforcement learning
FAISS                      Vector similarity search
LLaMA-3 / Mistral          Base instruction-tuned LLMs
LongBench                  Long-context evaluation suite
HotpotQA                   Multi-hop QA benchmark
```

---

## Repository Structure

```
ac-dag-rethesis/
├── README.md                  ← You are here
├── abstract/
│   └── AC_DAG_Abstract.pdf    ← Submission abstract (1-page)
├── src/
│   ├── encoder.py             ← SBERT + CLIP encoders
│   ├── dag_builder.py         ← Graph construction + GNN
│   ├── rl_policy.py           ← PPO traversal agent
│   ├── aggregator.py          ← Cross-modal attention
│   └── pipeline.py            ← End-to-end inject_context()
├── experiments/
│   └── benchmark_results.md   ← Evaluation results
└── requirements.txt           ← Dependencies
```

---

## Citation

```bibtex
@misc{acdags2026,
  title     = {Adaptive Context DAGs: Graph-Guided Multi-Modal Context Injection for Long-Horizon LLM Reasoning},
  author    = {Yashwanth R, Mohammad Ayaan Adil Ahmed},
  year      = {2026},
  note      = {ReThesis: AI Research Blitz, Ascent Techfest, Scaler School of Technology}
}
```

---

## Contact

**Yashwanth R** — ryashwanth429@gmail.com  
**Mohammand Ayaan Adil Ahmed** — m.ayaan.a.ahmed@gmail.com
ReThesis: AI Research Blitz | Ascent Techfest 2026
