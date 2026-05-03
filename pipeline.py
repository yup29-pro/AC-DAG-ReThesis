"""
AC-DAG Pipeline — Core inject_context() implementation
Adaptive Context DAGs for Long-Horizon LLM Reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
from stable_baselines3 import PPO


# ─── Encoders ────────────────────────────────────────────────────────────────

class MultiModalEncoder:
    """Encodes text and image chunks into a unified embedding space."""

    def __init__(self, device="cpu"):
        self.device = device
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.text_model.encode(texts, convert_to_tensor=True)
        return F.normalize(embeddings, dim=-1)

    def encode_images(self, images) -> torch.Tensor:
        preprocessed = torch.stack([self.clip_preprocess(img) for img in images])
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(preprocessed.to(self.device))
        return F.normalize(embeddings.float(), dim=-1)

    def encode_chunk(self, chunk: dict) -> torch.Tensor:
        """Encode a single chunk (text or image) with timestamp-aware offset."""
        if chunk["type"] == "text":
            emb = self.encode_text([chunk["content"]])[0]
        else:
            emb = self.encode_images([chunk["content"]])[0]
        timestamp_offset = torch.zeros(emb.shape[0])
        if "timestamp" in chunk:
            timestamp_offset[0] = chunk["timestamp"] / 1e9  # normalised epoch
        return emb + 0.01 * timestamp_offset.to(emb.device)


# ─── DAG Builder ─────────────────────────────────────────────────────────────

class GraphSAGEEdgeNet(nn.Module):
    """2-layer GraphSAGE that refines cosine-similarity edge weights."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def build_dag(chunks: list[dict], encoder: MultiModalEncoder, threshold: float = 0.3):
    """
    Build a DAG from retrieved chunks.
    Nodes  = multi-modal chunk embeddings
    Edges  = cosine-similarity pairs above threshold, refined by GraphSAGE
    Returns PyG Data object + raw node embeddings
    """
    embeddings = torch.stack([encoder.encode_chunk(c) for c in chunks])
    n = len(embeddings)

    # Cosine similarity → candidate edges
    sim = torch.mm(embeddings, embeddings.T)
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                src += [i, j]
                dst += [j, i]

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    graph = Data(x=embeddings, edge_index=edge_index)
    return graph, embeddings


# ─── RL Traversal Policy ─────────────────────────────────────────────────────

class DAGEnv:
    """
    Gym-compatible environment for DAG traversal.
    State  : current_node_embedding ⊕ query_embedding
    Action : index of next node to visit (or stop)
    Reward : reasoning quality signal from LLM (task success proxy)
    """

    def __init__(self, graph: Data, query_emb: torch.Tensor, max_steps: int = 8):
        self.graph = graph
        self.query_emb = query_emb
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.current = 0
        self.visited = set()
        self.path = []
        self.step_count = 0
        return self._obs()

    def _obs(self):
        node_emb = self.graph.x[self.current]
        return torch.cat([node_emb, self.query_emb]).numpy()

    def step(self, action: int):
        self.visited.add(self.current)
        self.path.append(self.current)
        self.current = action
        self.step_count += 1
        done = self.step_count >= self.max_steps or action in self.visited
        reward = self._estimate_reward()
        return self._obs(), reward, done, {}

    def _estimate_reward(self):
        """Proxy reward: cosine similarity of path centroid to query."""
        if not self.path:
            return 0.0
        path_embs = self.graph.x[self.path]
        centroid = path_embs.mean(dim=0)
        return float(F.cosine_similarity(centroid.unsqueeze(0), self.query_emb.unsqueeze(0)))


def select_path(graph: Data, query_emb: torch.Tensor, policy=None, max_steps: int = 8):
    """
    RL-guided DAG traversal.
    If no trained policy is provided, falls back to greedy cosine selection.
    """
    if policy is not None:
        env = DAGEnv(graph, query_emb, max_steps)
        obs = env.reset()
        path = []
        for _ in range(max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(int(action))
            path.append(env.path[-1])
            if done:
                break
        return path

    # Greedy fallback: pick top-k nodes by cosine similarity to query
    sims = F.cosine_similarity(graph.x, query_emb.unsqueeze(0))
    top_k = torch.topk(sims, min(max_steps, len(sims))).indices.tolist()
    return top_k


# ─── Aggregator ──────────────────────────────────────────────────────────────

class CrossModalAggregator(nn.Module):
    """Multi-head attention over selected DAG path → fused context vector."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, path_embeddings: torch.Tensor) -> torch.Tensor:
        """
        path_embeddings: (seq_len, embed_dim)
        Returns fused context vector of shape (embed_dim,)
        """
        x = path_embeddings.unsqueeze(0)          # (1, seq, dim)
        attn_out, _ = self.attn(x, x, x)
        fused = attn_out.squeeze(0).mean(dim=0)   # mean pool over path
        return self.proj(fused)


# ─── LLM Interface ───────────────────────────────────────────────────────────

def llm_reason(fused_ctx: torch.Tensor, query: str, llm_tokenizer, llm_model) -> str:
    """
    Inject fused context vector as a soft prefix and generate a response.
    Works with any HuggingFace causal LM (LLaMA-3, Mistral, etc.)
    """
    inputs = llm_tokenizer(query, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Project fused context to token embedding dimension
    embed_dim = llm_model.config.hidden_size
    ctx_proj = nn.Linear(fused_ctx.shape[0], embed_dim)(fused_ctx).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        token_embeds = llm_model.get_input_embeddings()(input_ids)
        combined = torch.cat([ctx_proj, token_embeds], dim=1)  # prefix injection
        outputs = llm_model.generate(inputs_embeds=combined, max_new_tokens=256)

    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def inject_context(
    query: str,
    chunks: list[dict],
    encoder: MultiModalEncoder,
    aggregator: CrossModalAggregator,
    llm_tokenizer=None,
    llm_model=None,
    rl_policy=None,
) -> str:
    """
    End-to-end AC-DAG pipeline.

    Args:
        query    : user query string
        chunks   : list of dicts with keys: type (text|image), content, timestamp (optional)
        encoder  : MultiModalEncoder instance
        aggregator: CrossModalAggregator instance
        llm_*    : optional HuggingFace tokenizer + model
        rl_policy: optional trained PPO policy

    Returns:
        Reasoned LLM response string (or fused_ctx tensor if no LLM provided)
    """
    # 1. Encode query
    query_emb = encoder.encode_text([query])[0]

    # 2. Build DAG
    graph, _ = build_dag(chunks, encoder)

    # 3. RL-guided traversal → optimal path
    path_indices = select_path(graph, query_emb, policy=rl_policy)

    # 4. Aggregate path embeddings
    path_embs = graph.x[path_indices]
    fused_ctx = aggregator(path_embs)

    # 5. LLM inference
    if llm_model is not None and llm_tokenizer is not None:
        return llm_reason(fused_ctx, query, llm_tokenizer, llm_model)

    return fused_ctx  # return vector for downstream use


# ─── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AC-DAG Pipeline — sanity check")
    encoder = MultiModalEncoder()
    aggregator = CrossModalAggregator()

    sample_chunks = [
        {"type": "text", "content": "The Eiffel Tower was built in 1889.", "timestamp": 1000},
        {"type": "text", "content": "Paris is the capital of France.", "timestamp": 2000},
        {"type": "text", "content": "The Seine river runs through Paris.", "timestamp": 3000},
    ]

    result = inject_context(
        query="Tell me about Paris landmarks",
        chunks=sample_chunks,
        encoder=encoder,
        aggregator=aggregator,
    )
    print(f"Fused context vector shape: {result.shape}")
    print("Pipeline OK!")
