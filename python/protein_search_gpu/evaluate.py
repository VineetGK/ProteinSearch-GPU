import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json

# Averaging convention: every metric is averaged over the queries that have at
# least one relevant item. Queries with no relevant items are skipped (they
# neither add to the numerator nor to the denominator).


def recall_at_k(
    retrieved: List[List[int]],
    relevant: List[List[int]],
    k: int
) -> float:
    total_recall = 0.0
    num_evaluated = 0
    
    for i in range(len(retrieved)):
        retrieved_k = set(retrieved[i][:k])
        relevant_set = set(relevant[i])
        
        if len(relevant_set) == 0:
            continue
            
        hits = len(retrieved_k & relevant_set)
        total_recall += hits / len(relevant_set)
        num_evaluated += 1
    
    return total_recall / num_evaluated if num_evaluated > 0 else 0.0


def precision_at_k(
    retrieved: List[List[int]],
    relevant: List[List[int]],
    k: int
) -> float:
    if k <= 0:
        return 0.0
    total_precision = 0.0
    num_evaluated = 0
    
    for i in range(len(retrieved)):
        retrieved_k = set(retrieved[i][:k])
        relevant_set = set(relevant[i])
        
        if len(relevant_set) == 0:
            continue
            
        hits = len(retrieved_k & relevant_set)
        total_precision += hits / k
        num_evaluated += 1
    
    return total_precision / num_evaluated if num_evaluated > 0 else 0.0


def mean_reciprocal_rank(
    retrieved: List[List[int]],
    relevant: List[List[int]]
) -> float:
    mrr = 0.0
    num_evaluated = 0
    
    for i in range(len(retrieved)):
        relevant_set = set(relevant[i])
        if not relevant_set:
            continue
        num_evaluated += 1
        for rank, idx in enumerate(retrieved[i], 1):
            if idx in relevant_set:
                mrr += 1.0 / rank
                break
    
    return mrr / num_evaluated if num_evaluated > 0 else 0.0


def ndcg_at_k(
    retrieved: List[List[int]],
    relevant: List[List[int]],
    k: int
) -> float:
    def dcg_at_k(rel_scores, k):
        return sum(
            (2**rel - 1) / np.log2(i + 2) 
            for i, rel in enumerate(rel_scores[:k])
        )
    
    total_ndcg = 0.0
    num_evaluated = 0
    
    for i in range(len(retrieved)):
        relevant_set = set(relevant[i])
        if not relevant_set:
            continue
        num_evaluated += 1
            
        rel_scores = [1 if idx in relevant_set else 0 for idx in retrieved[i][:k]]
        dcg = dcg_at_k(rel_scores, k)
        
        ideal_scores = [1] * min(len(relevant_set), k)
        idcg = dcg_at_k(ideal_scores, k)
        
        if idcg > 0:
            total_ndcg += dcg / idcg
    
    return total_ndcg / num_evaluated if num_evaluated > 0 else 0.0


def average_precision(
    retrieved: List[int],
    relevant: List[int]
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    
    hits = 0
    sum_prec = 0.0
    
    for i, idx in enumerate(retrieved):
        if idx in relevant_set:
            hits += 1
            sum_prec += hits / (i + 1)
    
    return sum_prec / len(relevant_set) if relevant_set else 0.0


def mean_average_precision(
    retrieved: List[List[int]],
    relevant: List[List[int]]
) -> float:
    aps = [average_precision(retrieved[i], relevant[i])
           for i in range(len(retrieved)) if len(relevant[i]) > 0]
    return float(np.mean(aps)) if aps else 0.0


def compute_retrieval_metrics(
    retrieved: List[List[int]],
    relevant: List[List[int]],
    k_values: List[int] = [1, 5, 10, 50, 100]
) -> Dict[str, float]:
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(retrieved, relevant, k)
        metrics[f'precision@{k}'] = precision_at_k(retrieved, relevant, k)
    
    metrics['mrr'] = mean_reciprocal_rank(retrieved, relevant)
    metrics['map'] = mean_average_precision(retrieved, relevant)
    
    for k in [10, 50]:
        if k in k_values:
            metrics[f'ndcg@{k}'] = ndcg_at_k(retrieved, relevant, k)
    
    return metrics


def evaluate_pfam_families(
    retrieved: List[List[int]],
    pfam_labels: List[str],
    query_pfam_labels: List[str]
) -> Dict[str, float]:
    relevant = []
    for q_pfam in query_pfam_labels:
        relevant_for_q = [i for i, p in enumerate(pfam_labels) if p == q_pfam]
        relevant.append(relevant_for_q)
    
    return compute_retrieval_metrics(retrieved, relevant)


def evaluate_uniprot_annotations(
    retrieved: List[List[int]],
    uniprot_annotations: Dict[int, List[str]],
    query_annotations: List[List[str]]
) -> Dict[str, float]:
    relevant = []
    for q_annots in query_annotations:
        relevant_for_q = []
        for i, annots in uniprot_annotations.items():
            if any(a in q_annots for a in annots):
                relevant_for_q.append(i)
        relevant.append(relevant_for_q)
    
    return compute_retrieval_metrics(retrieved, relevant)


def exact_match_recovery(
    retrieved: List[List[int]],
    query_ids: List[str],
    db_ids: List[str]
) -> float:
    exact_matches = 0
    for i, q_id in enumerate(query_ids):
        if q_id in [db_ids[idx] for idx in retrieved[i][:1]]:
            exact_matches += 1
    return exact_matches / len(query_ids) if query_ids else 0.0


def compute_ranking_metrics(
    query_embeddings: torch.Tensor,
    db_embeddings: torch.Tensor,
    query_ids: List[str],
    db_ids: List[str],
    query_pfam: Optional[List[str]] = None,
    db_pfam: Optional[List[str]] = None,
    query_go: Optional[List[List[str]]] = None,
    db_go: Optional[Dict[int, List[str]]] = None,
    k_values: List[int] = [1, 5, 10, 50, 100],
    use_cuda_kernels: bool = True
) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_cuda_kernels and device.type != 'cuda':
        raise RuntimeError("use_cuda_kernels=True requires a CUDA device")
    query_embeddings = query_embeddings.to(device=device, dtype=torch.float32)
    db_embeddings = db_embeddings.to(device=device, dtype=torch.float32)
    
    k = min(max(k_values), db_embeddings.size(0))
    
    if use_cuda_kernels:
        import protein_search_gpu as psg
        query_norm = psg.l2_normalize(query_embeddings)
        db_norm = psg.l2_normalize(db_embeddings)
        similarities = psg.cosine_similarity(query_norm, db_norm)
    else:
        query_norm = torch.nn.functional.normalize(query_embeddings, dim=1)
        db_norm = torch.nn.functional.normalize(db_embeddings, dim=1)
        similarities = torch.matmul(query_norm, db_norm.t())
    
    # The custom top-k kernel supports k <= 32; fall back to torch.topk above that.
    if use_cuda_kernels and k <= 32:
        _, indices = psg.topk(similarities, k)
    else:
        _, indices = torch.topk(similarities, k, dim=1)
    
    retrieved = indices.cpu().tolist()
    
    results = {}
    
    if query_pfam and db_pfam:
        results['pfam'] = evaluate_pfam_families(retrieved, db_pfam, query_pfam)
    
    if query_go and db_go:
        results['go'] = evaluate_uniprot_annotations(retrieved, db_go, query_go)
    
    results['exact_match'] = exact_match_recovery(retrieved, query_ids, db_ids)
    
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(f"\n{prefix}Retrieval Metrics:")
    print("-" * 40)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


def save_metrics(metrics: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    parser.add_argument("--retrieved", type=str, help="JSON file with retrieved indices")
    parser.add_argument("--relevant", type=str, help="JSON file with relevant indices")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.retrieved and args.relevant:
        with open(args.retrieved) as f:
            retrieved = json.load(f)
        with open(args.relevant) as f:
            relevant = json.load(f)
        
        metrics = compute_retrieval_metrics(retrieved, relevant)
        print_metrics(metrics)
        
        if args.output:
            save_metrics(metrics, args.output)