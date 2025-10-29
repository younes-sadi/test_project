import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


@dataclass
class ProofExample:
    """Single proof example from NaturalProofs"""
    theorem_id: str
    statement: str
    proof: str
    domain: str
    definitions_used: List[str]
    proof_steps: List[str]
    
    
class NaturalProofsDataset(Dataset):
    """PyTorch Dataset for NaturalProofs"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_path: Path to NaturalProofs dataset directory
            split: 'train', 'val', or 'test'
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache preprocessed data
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load data
        self.examples = self._load_data()
        
        print(f"Loaded {len(self.examples)} {split} examples")
        
    def _load_data(self) -> List[ProofExample]:
        """Load and parse NaturalProofs data"""
        
        # Check for cached data
        if self.cache_dir:
            cache_file = self.cache_dir / f"{self.split}_processed.pkl"
            if cache_file.exists():
                print(f"Loading cached data from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        examples = []
        
        # Try different possible file structures
        possible_paths = [
            self.data_path / f"{self.split}.json",
            self.data_path / f"{self.split}.jsonl",
            self.data_path / self.split / "data.json",
        ]
        
        data_file = None
        for path in possible_paths:
            if path.exists():
                data_file = path
                break
                
        if data_file is None:
            raise FileNotFoundError(
                f"Could not find data file in any of: {possible_paths}"
            )
        
        print(f"Loading data from {data_file}")
        
        # Load JSON or JSONL
        if data_file.suffix == '.jsonl':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    examples.append(self._parse_example(data))
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        examples.append(self._parse_example(item))
                elif isinstance(data, dict):
                    for key, item in data.items():
                        examples.append(self._parse_example(item, key))
        
        # Cache processed data
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{self.split}_processed.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(examples, f)
            print(f"Cached data to {cache_file}")
        
        return examples
    
    def _parse_example(self, data: Dict, theorem_id: Optional[str] = None) -> ProofExample:
        """Parse a single example from raw data"""
        
        # Extract fields with various possible key names
        theorem_id = theorem_id or data.get('id') or data.get('theorem_id') or data.get('name')
        statement = data.get('statement') or data.get('theorem') or data.get('claim')
        proof = data.get('proof') or data.get('proof_text') or ""
        domain = data.get('domain') or data.get('category') or 'unknown'
        definitions = data.get('definitions_used') or data.get('definitions') or []
        
        # Extract proof steps
        proof_steps = self._extract_proof_steps(proof)
        
        return ProofExample(
            theorem_id=str(theorem_id),
            statement=self._clean_text(statement),
            proof=self._clean_text(proof),
            domain=domain,
            definitions_used=definitions,
            proof_steps=proof_steps
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize mathematical text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common mathematical symbols
        replacements = {
            '∀': 'forall',
            '∃': 'exists',
            '→': 'implies',
            '↔': 'iff',
            '∧': 'and',
            '∨': 'or',
            '¬': 'not',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '∈': 'in',
            '⊆': 'subset',
            '∪': 'union',
            '∩': 'intersect',
        }
        
        for symbol, replacement in replacements.items():
            text = text.replace(symbol, f' {replacement} ')
        
        # Remove extra whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_proof_steps(self, proof: str) -> List[str]:
        """Extract individual proof steps from proof text"""
        if not proof:
            return []
        
        # Split on common step indicators
        steps = []
        
        # Try numbered steps first (1., 2., etc.)
        numbered_pattern = r'(?:^|\n)\s*(\d+\.)\s*'
        parts = re.split(numbered_pattern, proof)
        
        if len(parts) > 2:  # Found numbered steps
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    step = parts[i+1].strip()
                    if step:
                        steps.append(step)
        else:
            # Try splitting on sentences
            sentences = re.split(r'[.!?]\s+', proof)
            steps = [s.strip() for s in sentences if s.strip()]
        
        return steps
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get a single example"""
        example = self.examples[idx]
        
        return {
            'theorem_id': example.theorem_id,
            'statement': example.statement,
            'proof': example.proof,
            'domain': example.domain,
            'proof_steps': example.proof_steps,
        }


class TacticDataset(Dataset):
    """Dataset that formats data as (context, goal, tactic) triples"""
    
    def __init__(
        self,
        naturalproofs_dataset: NaturalProofsDataset,
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            naturalproofs_dataset: Base NaturalProofs dataset
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.base_dataset = naturalproofs_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create tactic examples from proof steps
        self.tactic_examples = self._create_tactic_examples()
        
        print(f"Created {len(self.tactic_examples)} tactic examples")
    
    def _create_tactic_examples(self) -> List[Dict]:
        """Convert proofs into (context, goal, tactic) format"""
        tactic_examples = []
        
        for example in self.base_dataset.examples:
            # For each proof step, create a tactic prediction task
            context = f"Theorem: {example.statement}"
            goal = "Prove the theorem."
            
            for i, step in enumerate(example.proof_steps):
                tactic_examples.append({
                    'context': context,
                    'goal': goal,
                    'tactic': step,
                    'theorem_id': example.theorem_id,
                    'step_num': i
                })
                
                # Update context with this step for next prediction
                context += f" Step {i+1}: {step}"
        
        return tactic_examples
    
    def __len__(self) -> int:
        return len(self.tactic_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized tactic example"""
        example = self.tactic_examples[idx]
        
        # Format as input-output pair
        input_text = f"Context: {example['context']} Goal: {example['goal']}"
        output_text = example['tactic']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': output_encoding['input_ids'].squeeze(0),
            'theorem_id': example['theorem_id'],
            'step_num': example['step_num']
        }


def create_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_path: Path to NaturalProofs dataset
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        cache_dir: Directory to cache preprocessed data
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create base datasets
    train_base = NaturalProofsDataset(
        data_path, 
        split='train', 
        max_length=max_length,
        cache_dir=cache_dir
    )
    val_base = NaturalProofsDataset(
        data_path, 
        split='val', 
        max_length=max_length,
        cache_dir=cache_dir
    )
    test_base = NaturalProofsDataset(
        data_path, 
        split='test', 
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    # Create tactic datasets
    train_dataset = TacticDataset(train_base, tokenizer, max_length)
    val_dataset = TacticDataset(val_base, tokenizer, max_length)
    test_dataset = TacticDataset(test_base, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    
    # Example usage
    data_path = "./data/naturalproofs"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            tokenizer=tokenizer,
            batch_size=4,
            cache_dir="./cache"
        )
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure NaturalProofs data is in the correct location")
