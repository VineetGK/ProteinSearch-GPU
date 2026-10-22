import torch
from transformers import EsmModel, EsmTokenizer
from typing import List, Tuple, Optional, Dict
import os
import numpy as np
from tqdm import tqdm


def _mean_pool_residues(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over residue tokens only.

    The ESM tokenizer wraps each sequence as <cls> residues <eos> (followed by
    padding), so the first token and the last attended token are excluded.
    """
    mask = attention_mask.clone()
    lengths = mask.sum(dim=1)  # includes <cls> and <eos>
    mask[:, 0] = 0
    rows = torch.arange(mask.size(0), device=mask.device)
    mask[rows, (lengths - 1).clamp(min=0)] = 0
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class ESM2Encoder:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda",
        half_precision: bool = True,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.half_precision = half_precision
        
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        if half_precision:
            self.model.half()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"Loaded {model_name} with embedding dim {self.embedding_dim}")
    
    @torch.no_grad()
    def encode(
        self,
        sequences: List[str],
        batch_size: int = 32,
        max_length: int = 1022,
        pooling: str = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        all_embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding"):
            batch = sequences[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
            
            if pooling == "mean":
                embeddings = _mean_pool_residues(hidden_states.float(), inputs["attention_mask"])
            elif pooling == "cls":
                embeddings = hidden_states[:, 0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            
            all_embeddings.append(embeddings.float().cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def encode_single(self, sequence: str, max_length: int = 1022) -> torch.Tensor:
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        embedding = _mean_pool_residues(hidden_states.float(), inputs["attention_mask"])
        
        return embedding.float().cpu().squeeze(0)
    
    def save_embeddings(
        self,
        embeddings: torch.Tensor,
        sequence_ids: List[str],
        output_path: str,
    ):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.savez_compressed(
            output_path,
            embeddings=embeddings.numpy(),
            sequence_ids=np.array(sequence_ids, dtype=object),
        )
    
    @staticmethod
    def load_embeddings(path: str) -> Tuple[torch.Tensor, List[str]]:
        data = np.load(path, allow_pickle=True)
        embeddings = torch.from_numpy(data['embeddings'])
        sequence_ids = data['sequence_ids'].tolist()
        return embeddings, sequence_ids


def load_esm2_model(model_name: str, device: str = "cuda", half: bool = True) -> ESM2Encoder:
    return ESM2Encoder(model_name, device, half)


def encode_fasta(
    fasta_path: str,
    output_path: str,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 32,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[str]]:
    from .fasta import read_fasta
    
    sequence_ids, sequences = read_fasta(fasta_path)
    
    encoder = ESM2Encoder(model_name, device)
    embeddings = encoder.encode(sequences, batch_size=batch_size)
    
    encoder.save_embeddings(embeddings, sequence_ids, output_path)
    
    return embeddings, sequence_ids


def load_embeddings(path: str) -> Tuple[torch.Tensor, List[str]]:
    return ESM2Encoder.load_embeddings(path)


def get_model_info(model_name: str) -> Dict:
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    return {
        "name": model_name,
        "embedding_dim": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
    }