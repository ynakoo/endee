"""
Custom Text Embedding Model
============================
A hand-built PyTorch nn.Module that produces 384-dimensional dense embeddings
from raw text. Instead of relying on the black-box sentence-transformers library,
we construct the full pipeline ourselves:

  1. Tokenizer  — HuggingFace AutoTokenizer (WordPiece vocabulary)
  2. Backbone   — A pre-trained MiniLM transformer loaded via AutoModel
  3. Pooling    — Our own mean-pooling layer (averages all token embeddings,
                  respecting the attention mask so padding tokens are ignored)
  4. Normalize  — L2 normalization so cosine similarity works correctly in Endee

Architecture Diagram:
  raw text → [Tokenizer] → input_ids + attention_mask
           → [MiniLM Transformer Backbone] → hidden states (batch, seq, 384)
           → [MeanPoolingLayer] → sentence embedding (batch, 384)
           → [L2 Normalize] → unit vector (batch, 384)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Custom Pooling Layer
# ---------------------------------------------------------------------------
class MeanPoolingLayer(nn.Module):
    """
    Averages all token-level hidden states into a single fixed-size vector,
    ignoring padding tokens via the attention mask.

    This is the same strategy used inside sentence-transformers, but written
    explicitly so you can see (and modify) every step.
    """

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: (batch, seq_len, hidden_dim)
        # attention_mask shape: (batch, seq_len)

        # Expand mask to match hidden_states dimensions: (batch, seq_len, 1)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Zero-out padding token embeddings, then sum across the sequence axis
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

        # Count non-padding tokens per sample (clamp to avoid division by zero)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Compute the mean
        return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Full Custom Embedding Model
# ---------------------------------------------------------------------------
class TextEmbeddingModel(nn.Module):
    """
    Our custom embedding model.

    Components:
        - backbone : a pre-trained transformer (MiniLM-L6-v2, 384-dim output)
        - pooler   : MeanPoolingLayer (our own code above)

    The forward() pass returns L2-normalized sentence embeddings ready for
    cosine similarity search inside the Endee vector database.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPoolingLayer()

        # Freeze the backbone so we don't accidentally update weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.eval()  # inference mode by default

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass:
          tokens → transformer → mean pool → L2 normalize
        """
        # Step 1: Run through the transformer backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Step 2: Extract the last hidden state (all token embeddings)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq, 384)

        # Step 3: Mean pool across the sequence dimension
        pooled = self.pooler(last_hidden_state, attention_mask)  # (batch, 384)

        # Step 4: L2 Normalize so cosine distance works correctly
        normalized = F.normalize(pooled, p=2, dim=1)

        return normalized


# ---------------------------------------------------------------------------
# Module-level singleton (loaded once on import)
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading custom embedding model: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TextEmbeddingModel(MODEL_NAME)
    print("Custom TextEmbeddingModel loaded successfully.")
except Exception as e:
    print(f"Error loading custom embedding model: {e}")
    tokenizer = None
    model = None


# ---------------------------------------------------------------------------
# Public API (same signature as before so app.py doesn't change)
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """
    Tokenizes the input text, runs it through our custom TextEmbeddingModel,
    and returns a 384-dimensional list of floats.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Custom embedding model is not loaded.")

    # Tokenize
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Forward pass through our custom model
    embedding = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    # Convert from (1, 384) tensor → flat Python list
    return embedding.squeeze(0).tolist()
