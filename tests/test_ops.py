"""Correctness tests for the CUDA ops against PyTorch / pure-Python references.

Skipped automatically when no CUDA device is available or the extension has
not been built. Run with:  pytest tests/
"""

import random

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA device not available", allow_module_level=True)

_C = pytest.importorskip("protein_search_gpu._C", reason="CUDA extension not built")

DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# BLOSUM62 in protein_search_gpu.datasets.AA_TO_IDX order
# (A R N D C Q E G H I L K M F P S T W Y V X U O B Z J); U, O, J scored as X.
BLOSUM62 = [
    [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0, 0, 0, -2, -1, 0],
    [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, -1, -1, -1, 0, -1],
    [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, -1, -1, -1, 3, 0, -1],
    [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, -1, -1, -1, 4, 1, -1],
    [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -2, -2, -2, -3, -3, -2],
    [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, -1, -1, -1, 0, 3, -1],
    [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, -1, -1, -1, 1, 4, -1],
    [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -1, -1, -1, -2, -1],
    [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, -1, -1, -1, 0, 0, -1],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -1, -1, -1, -3, -3, -1],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -1, -1, -1, -4, -3, -1],
    [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, -1, -1, -1, 0, 1, -1],
    [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -1, -1, -1, -3, -1, -1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -1, -1, -1, -3, -3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -2, -2, -2, -1, -2],
    [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0, 0, 0, -1, -1, 0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -2, -2, -2, -4, -3, -2],
    [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -1, -1, -1, -3, -2, -1],
    [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -1, -1, -1, -3, -2, -1],
    [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -1, -1, -1],
    [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, -1, -1, -1, 4, 1, -1],
    [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, -1, -1, -1, 1, 4, -1],
    [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -1, -1, -1],
]


def sw_reference(a, b, gap_open, gap_extend):
    """Full-matrix Gotoh local alignment score (gap of length L costs
    gap_open + (L - 1) * gap_extend)."""
    neg = -(10**9)
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    E = [[neg] * (m + 1) for _ in range(n + 1)]
    Fm = [[neg] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i][j] = max(E[i - 1][j] - gap_extend, H[i - 1][j] - gap_open)
            Fm[i][j] = max(Fm[i][j - 1] - gap_extend, H[i][j - 1] - gap_open)
            H[i][j] = max(0, H[i - 1][j - 1] + BLOSUM62[a[i - 1]][b[j - 1]], E[i][j], Fm[i][j])
            best = max(best, H[i][j])
    return best


def random_sequences(num, width, rng, min_len=0):
    seqs = torch.full((num, width), 20, dtype=torch.int8)  # pad with X
    lens = torch.empty(num, dtype=torch.int64)
    raw = []
    for r in range(num):
        length = rng.randint(min_len, width)
        s = [rng.randrange(26) for _ in range(length)]
        seqs[r, :length] = torch.tensor(s, dtype=torch.int8)
        lens[r] = length
        raw.append(s)
    return seqs, lens, raw


SHAPES = [(1, 1), (7, 13), (5, 4), (3, 1023), (33, 1280), (2, 4097)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.float16: dict(rtol=1e-2, atol=1e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=1e-2),
}


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_normalize_matches_torch(shape, dtype):
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    out = _C.l2_normalize(x)
    ref = F.normalize(x.float(), dim=1)
    assert out.dtype == dtype and out.shape == x.shape
    torch.testing.assert_close(out.float(), ref, **TOL[dtype])


def test_l2_normalize_zero_row_and_noncontiguous():
    x = torch.randn(6, 9, device=DEVICE)
    x[2] = 0
    out = _C.l2_normalize(x.t())  # non-contiguous [9, 6]
    torch.testing.assert_close(out, F.normalize(x.t(), dim=1))
    assert torch.all(_C.l2_normalize(x)[2] == 0)


def test_l2_normalize_misaligned_base_pointer():
    # dim % 4 == 0 but the base pointer is offset by one float, so the float4
    # path must not be taken.
    buf = torch.randn(5 * 8 + 1, device=DEVICE)
    x = buf[1:].view(5, 8)
    assert x.data_ptr() % 16 != 0
    torch.testing.assert_close(_C.l2_normalize(x), F.normalize(x, dim=1))


def test_l2_normalize_inplace():
    x = torch.randn(11, 37, device=DEVICE)
    ref = F.normalize(x, dim=1)
    out = _C.l2_normalize_inplace(x)
    assert out.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, ref)
    with pytest.raises(RuntimeError):
        _C.l2_normalize_inplace(torch.randn(4, 6, device=DEVICE).t())


# ---------------------------------------------------------------------------
# cosine_similarity (dot product of the given rows)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nq,nd,dim", [(1, 1, 1), (3, 257, 13), (4, 100, 64), (2, 33, 1281), (70000, 3, 5)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cosine_similarity_matches_matmul(nq, nd, dim, dtype):
    q = F.normalize(torch.randn(nq, dim, device=DEVICE), dim=1).to(dtype)
    db = F.normalize(torch.randn(nd, dim, device=DEVICE), dim=1).to(dtype)
    out = _C.cosine_similarity(q, db)
    ref = q.float() @ db.float().t()
    assert out.dtype == torch.float32 and out.shape == (nq, nd)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_cosine_similarity_single_query_and_misaligned():
    q = torch.randn(12, device=DEVICE)
    buf = torch.randn(9 * 12 + 1, device=DEVICE)
    db = buf[1:].view(9, 12)
    out = _C.cosine_similarity(q, db)
    assert out.shape == (9,)
    torch.testing.assert_close(out, db @ q, rtol=1e-5, atol=1e-5)


def test_cosine_similarity_rejects_bad_inputs():
    with pytest.raises(RuntimeError):
        _C.cosine_similarity(torch.randn(2, 4), torch.randn(3, 4))  # CPU
    with pytest.raises(RuntimeError):
        _C.cosine_similarity(torch.randn(2, 4, device=DEVICE), torch.randn(3, 5, device=DEVICE))
    with pytest.raises(RuntimeError):
        _C.cosine_similarity(torch.randn(2, 4, device=DEVICE, dtype=torch.float64),
                             torch.randn(3, 4, device=DEVICE, dtype=torch.float64))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------

def _check_topk(sims, k):
    values, indices = _C.topk(sims, k)
    ref_values, _ = torch.topk(sims, k, dim=1)
    assert values.shape == (sims.size(0), k) and indices.dtype == torch.int64
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    # Indices may differ from torch on ties, but must point at those values
    # and be distinct within each row.
    torch.testing.assert_close(sims.gather(1, indices), values, rtol=0, atol=0)
    for row in indices.cpu().tolist():
        assert len(set(row)) == k


@pytest.mark.parametrize("nq,n", [(1, 1), (3, 37), (5, 1000), (2, 100003), (64, 300)])
@pytest.mark.parametrize("k", [1, 5, 32])
def test_topk_matches_torch(nq, n, k):
    if k > n:
        pytest.skip("k > n")
    _check_topk(torch.randn(nq, n, device=DEVICE), k)


def test_topk_with_ties():
    sims = torch.randint(0, 4, (7, 5000), device=DEVICE).float()
    _check_topk(sims, 32)


def test_topk_rejects_large_k():
    sims = torch.randn(2, 100, device=DEVICE)
    with pytest.raises(RuntimeError):
        _C.topk(sims, 33)
    with pytest.raises(RuntimeError):
        _C.topk(torch.randn(2, 10, device=DEVICE), 11)


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 3, 8, 13, 1280])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_matches_index_select(dim, dtype):
    db_size = 50
    db = torch.randn(db_size, dim, device=DEVICE).to(dtype)
    idx = torch.randint(0, db_size, (4, 7), device=DEVICE)
    idx[0, 0] = -1
    idx[1, 3] = db_size
    idx[3, 6] = -5
    out = _C.gather(db, idx)

    valid = (idx >= 0) & (idx < db_size)
    ref = db.index_select(0, idx.clamp(0, db_size - 1).flatten()).view(4, 7, dim)
    ref = ref * valid.unsqueeze(-1).to(dtype)
    assert out.shape == (4, 7, dim) and out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Smith-Waterman
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gap_open,gap_extend", [(11, 1), (5, 2), (3, 3)])
def test_smith_waterman_all_pairs(gap_open, gap_extend):
    rng = random.Random(1234 + gap_open)
    q, ql, q_raw = random_sequences(5, 37, rng)
    d, dl, d_raw = random_sequences(7, 29, rng)
    scores = _C.smith_waterman(q.to(DEVICE), d.to(DEVICE), ql.to(DEVICE), dl.to(DEVICE),
                               gap_open, gap_extend)
    assert scores.shape == (5, 7) and scores.dtype == torch.float32
    ref = torch.tensor([[sw_reference(a, b, gap_open, gap_extend) for b in d_raw] for a in q_raw],
                       dtype=torch.float32)
    torch.testing.assert_close(scores.cpu(), ref, rtol=0, atol=0)


def test_smith_waterman_identity_and_padding():
    # A sequence aligned to itself scores the sum of its BLOSUM diagonal, and
    # padding beyond the given length is ignored.
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    seqs = torch.full((1, 32), 17, dtype=torch.int8)  # pad with W (high self-score)
    seqs[0, : len(s)] = torch.tensor(s, dtype=torch.int8)
    lens = torch.tensor([len(s)])
    score = _C.smith_waterman(seqs.to(DEVICE), seqs.to(DEVICE), lens.to(DEVICE), lens.to(DEVICE))
    assert score.item() == sum(BLOSUM62[a][a] for a in s)


def test_smith_waterman_batch_uses_pair_indices():
    rng = random.Random(99)
    q, ql, q_raw = random_sequences(4, 50, rng, min_len=1)
    d, dl, d_raw = random_sequences(6, 41, rng, min_len=1)
    pairs = torch.tensor([[3, 0], [0, 5], [2, 2], [0, 5], [1, 4]], dtype=torch.int64)
    scores = _C.smith_waterman_batch(q.to(DEVICE), d.to(DEVICE), ql.to(DEVICE), dl.to(DEVICE),
                                     pairs.to(DEVICE), 11, 1)
    ref = torch.tensor([sw_reference(q_raw[i], d_raw[j], 11, 1) for i, j in pairs.tolist()],
                       dtype=torch.float32)
    torch.testing.assert_close(scores.cpu(), ref, rtol=0, atol=0)

    bad = torch.tensor([[4, 0]], dtype=torch.int64, device=DEVICE)  # query index out of range
    with pytest.raises(RuntimeError):
        _C.smith_waterman_batch(q.to(DEVICE), d.to(DEVICE), ql.to(DEVICE), dl.to(DEVICE), bad, 11, 1)


def test_smith_waterman_rejects_bad_lengths():
    seqs = torch.zeros(2, 10, dtype=torch.int8, device=DEVICE)
    lens = torch.tensor([5, 11], device=DEVICE)  # 11 > width
    with pytest.raises(RuntimeError):
        _C.smith_waterman(seqs, seqs, lens, lens)
