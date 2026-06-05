"""
Homework 10: Sinusoidal Position Encoding and RoPE.

Run:
    python rope_position_encoding_demo.py

This script implements:
1. Sinusoidal position encoding.
2. Two-dimensional rotation.
3. High-dimensional RoPE.
4. Comparison between E + pos and RoPE input forms.
5. A numerical experiment verifying RoPE's relative-position property.
"""

from __future__ import annotations

import numpy as np


def sinusoidal_position_encoding(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Return the classic Transformer sinusoidal position encoding."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")

    positions = np.arange(seq_len)[:, None]
    pair_ids = np.arange(0, dim, 2)[None, :]
    inv_freq = 1.0 / (base ** (pair_ids / dim))

    pe = np.zeros((seq_len, dim), dtype=np.float64)
    angles = positions * inv_freq
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def rotate_2d(x: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a 2D vector counterclockwise by theta radians."""
    if x.shape != (2,):
        raise ValueError("x must be a 2D vector")

    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )
    return rotation @ x


def rope_frequencies(dim: int, base: float = 10000.0) -> np.ndarray:
    """Return angular frequencies used by RoPE for each 2D channel pair."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    pair_ids = np.arange(0, dim, 2, dtype=np.float64)
    return 1.0 / (base ** (pair_ids / dim))


def apply_rope_one(x: np.ndarray, position: int, base: float = 10000.0) -> np.ndarray:
    """Apply RoPE to one vector at one sequence position."""
    if x.ndim != 1 or x.shape[0] % 2 != 0:
        raise ValueError("x must be a 1D vector with even dimension")

    dim = x.shape[0]
    freqs = rope_frequencies(dim, base)
    angles = position * freqs

    even = x[0::2]
    odd = x[1::2]

    out = np.empty_like(x, dtype=np.float64)
    out[0::2] = even * np.cos(angles) - odd * np.sin(angles)
    out[1::2] = even * np.sin(angles) + odd * np.cos(angles)
    return out


def apply_rope_batch(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Apply RoPE to a sequence of vectors shaped [seq_len, dim]."""
    if x.ndim != 2 or x.shape[1] % 2 != 0:
        raise ValueError("x must have shape [seq_len, even_dim]")

    return np.stack([apply_rope_one(vec, pos, base) for pos, vec in enumerate(x)], axis=0)


def attention_scores_e_plus_pos(embeddings: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Toy attention scores after adding sinusoidal position encoding to embeddings."""
    pos = sinusoidal_position_encoding(embeddings.shape[0], embeddings.shape[1], base)
    hidden = embeddings + pos
    return hidden @ hidden.T


def attention_scores_rope(q: np.ndarray, k: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Toy attention scores after applying RoPE to Q and K."""
    q_rope = apply_rope_batch(q, base)
    k_rope = apply_rope_batch(k, base)
    return q_rope @ k_rope.T


def verify_rope_relative_property(dim: int = 8, max_pos: int = 8, seed: int = 7) -> float:
    """
    Verify:
        <R_m q, R_n k> == <q, R_{n-m} k>

    The returned value is the maximum absolute numerical error.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=dim)
    k = rng.normal(size=dim)

    max_error = 0.0
    for m in range(max_pos):
        for n in range(max_pos):
            left = apply_rope_one(q, m) @ apply_rope_one(k, n)
            right = q @ apply_rope_one(k, n - m)
            max_error = max(max_error, abs(left - right))
    return max_error


def print_matrix(name: str, matrix: np.ndarray, precision: int = 4) -> None:
    print(f"\n{name}:")
    print(np.array2string(matrix, precision=precision, suppress_small=True))


def main() -> None:
    seq_len = 6
    dim = 8
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(seq_len, dim))

    pe = sinusoidal_position_encoding(seq_len, dim)
    rotated = rotate_2d(np.array([1.0, 0.0]), np.pi / 4)

    q = embeddings.copy()
    k = rng.normal(size=(seq_len, dim))
    rope_scores = attention_scores_rope(q, k)
    e_pos_scores = attention_scores_e_plus_pos(embeddings)
    max_error = verify_rope_relative_property(dim=dim, max_pos=10)

    print("Sinusoidal Position Encoding and RoPE demo")
    print("=" * 48)
    print_matrix("1) Sinusoidal position encoding, first 4 positions", pe[:4])
    print(f"\n2) Rotate [1, 0] by 45 degrees -> {rotated}")
    print_matrix("3) First token after high-dimensional RoPE", apply_rope_batch(q)[:1])
    print_matrix("4a) Scores from E + pos", e_pos_scores)
    print_matrix("4b) Scores from RoPE(Q), RoPE(K)", rope_scores)
    print(f"\n5) Max error in RoPE relative-position identity: {max_error:.3e}")
    print("\n6) Conclusion:")
    print("   E + pos injects position by addition, so content and position are mixed in the hidden state.")
    print("   RoPE injects position by rotating Q and K, so the dot product naturally contains n - m.")


if __name__ == "__main__":
    main()
