"""
ProteinSearch-GPU: CUDA-accelerated semantic protein retrieval.

The GPU ops (l2_normalize, cosine_similarity, topk, gather, smith_waterman,
...) come from the compiled ``protein_search_gpu._C`` extension. If it has not
been built, the pure-Python modules (datasets, fasta, evaluate, encoder) still
import, and accessing an op raises an ImportError explaining how to build it.
"""

import torch  # noqa: F401  (loads libtorch before the extension)

_OPS = (
    "l2_normalize",
    "l2_normalize_inplace",
    "cosine_similarity",
    "topk",
    "gather",
    "smith_waterman",
    "smith_waterman_batch",
)

try:
    from ._C import (  # noqa: F401
        cosine_similarity,
        gather,
        l2_normalize,
        l2_normalize_inplace,
        smith_waterman,
        smith_waterman_batch,
        topk,
    )

    _EXTENSION_ERROR = None
except ImportError as exc:  # extension not built / built for another torch
    _EXTENSION_ERROR = exc


def __getattr__(name):
    if name in _OPS and _EXTENSION_ERROR is not None:
        raise ImportError(
            f"protein_search_gpu.{name} needs the compiled CUDA extension "
            "protein_search_gpu._C, which could not be imported "
            f"({_EXTENSION_ERROR}). Build it with the CUDA toolkit installed: "
            "`pip install --no-build-isolation -e .`"
        ) from _EXTENSION_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def extension_available() -> bool:
    """True if the compiled CUDA extension was imported successfully."""
    return _EXTENSION_ERROR is None


from .encoder import (  # noqa: E402
    ESM2Encoder,
    encode_fasta,
    load_embeddings,
    load_esm2_model,
    get_model_info,
)

from .datasets import (  # noqa: E402
    parse_fasta,
    sequence_to_indices,
    indices_to_sequence,
    ProteinDataset,
    ProteinPairDataset,
    PfamDataset,
    get_dataloader,
    load_swissprot,
    load_uniref50,
    create_train_val_split,
    collate_protein_batch,
)

from .benchmark import (  # noqa: E402
    benchmark_l2_normalize,
    benchmark_cosine_similarity,
    benchmark_topk,
    benchmark_gather,
    benchmark_smith_waterman,
    benchmark_pytorch_cosine,
    benchmark_pytorch_topk,
    run_full_benchmark,
    profile_memory,
)

from .evaluate import (  # noqa: E402
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    compute_retrieval_metrics,
    evaluate_pfam_families,
    evaluate_uniprot_annotations,
    exact_match_recovery,
    print_metrics,
    save_metrics,
    load_metrics,
)

__version__ = "0.1.0"
