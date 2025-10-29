"""
Data Loader for True ASTactic
Converts NaturalProofs to AST-based training format
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tactic_ast import (
    ASTNode, TacticNodeType, TacticParser, 
    TacticVocabulary, TacticGenerator
)


class NaturalProofsToAST:
    """Convert NaturalProofs natural language to AST format"""
    
    def __init__(self):
        self.parser = TacticParser()
        self.vocabulary = TacticVocabulary()
    
    def convert_proof_to_asts(self, proof: str) -> List[ASTNode]:
        """
        Convert natural language proof to list of AST tactics
        
        Args:
            proof: Natural language proof text
        
        Returns:
            List of AST nodes representing tactics
        """
        # Split proof into steps
        steps = self._extract_proof_steps(proof)
        
        # Parse each step into AST
        asts = []
        for step in steps:
            ast = self.parser.parse(step)
            asts.append(ast)
            
            # Add tokens to vocabulary
            self._add_ast_to_vocab(ast)
        
        return asts
    
    def _extract_proof_steps(self, proof: str) -> List[str]:
        """Extract individual steps from proof"""
        # Try numbered steps
        numbered_pattern = r'(?:^|\n)\s*(\d+\.)\s*([^\n]+)'
        matches = re.findall(numbered_pattern, proof)
        
        if matches:
            return [match[1].strip() for match in matches]
        
        # Try sentence-based splitting
        sentences = re.split(r'[.!]\s+', proof)
        steps = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not steps:
            # Fallback: split by newlines
            steps = [line.strip() for line in proof.split('\n') if line.strip()]
        
        return steps
    
    def _add_ast_to_vocab(self, ast: ASTNode):
        """Add AST tokens to vocabulary"""
        if ast.value:
            self.vocabulary.add_token(ast.value)
        
        for child in ast.children:
            self._add_ast_to_vocab(child)


class ASTacticDataset(Dataset):
    """Dataset for training ASTactic"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_context_length: int = 256,
        max_ast_size: int = 50,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_path: Path to NaturalProofs data
            split: 'train', 'val', or 'test'
            max_context_length: Max length for context tokens
            max_ast_size: Max number of nodes in AST
            cache_dir: Directory to cache processed data
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_context_length = max_context_length
        self.max_ast_size = max_ast_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Initialize converter and vocabulary
        self.converter = NaturalProofsToAST()
        
        # Load and process data
        self.examples = self._load_and_process()
        
        # Build vocabulary
        self.vocabulary = self.converter.vocabulary
        
        print(f"Loaded {len(self.examples)} {split} examples")
        print(f"Vocabulary size: {self.vocabulary.size()}")
    
    def _load_and_process(self) -> List[Dict]:
        """Load raw data and convert to AST format"""
        
        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"astactic_{self.split}.pkl"
            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Load raw NaturalProofs data
        data_file = self._find_data_file()
        
        if data_file is None:
            raise FileNotFoundError(f"Could not find {self.split} data")
        
        print(f"Loading data from {data_file}")
        
        # Parse JSON/JSONL
        raw_examples = self._load_json(data_file)
        
        # Convert to AST format
        processed_examples = []
        
        for i, raw in enumerate(raw_examples):
            if i % 100 == 0:
                print(f"Processing {i}/{len(raw_examples)}...")
            
            try:
                processed = self._process_example(raw)
                if processed:
                    processed_examples.append(processed)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                continue
        
        # Cache processed data
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"astactic_{self.split}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(processed_examples, f)
            print(f"Cached to {cache_file}")
        
        return processed_examples
    
    def _find_data_file(self) -> Optional[Path]:
        """Find data file for split"""
        possible_paths = [
            self.data_path / f"{self.split}.json",
            self.data_path / f"{self.split}.jsonl",
            self.data_path / self.split / "data.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON or JSONL file"""
        examples = []
        
        if file_path.suffix == '.jsonl':
            with open(file_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
        else:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                elif isinstance(data, dict):
                    examples = list(data.values())
        
        return examples
    
    def _process_example(self, raw: Dict) -> Optional[Dict]:
        """Convert raw example to AST training format"""
        
        # Extract fields
        theorem_id = raw.get('id') or raw.get('theorem_id') or 'unknown'
        statement = raw.get('statement') or raw.get('theorem') or ''
        proof = raw.get('proof') or raw.get('proof_text') or ''
        
        if not statement or not proof:
            return None
        
        # Convert proof to ASTs
        proof_asts = self.converter.convert_proof_to_asts(proof)
        
        if not proof_asts:
            return None
        
        # Tokenize context (theorem statement)
        context_text = f"Theorem: {statement}"
        context_tokens = self._tokenize_context(context_text)
        
        # Create training examples (one per proof step)
        return {
            'theorem_id': theorem_id,
            'context': context_text,
            'context_tokens': context_tokens,
            'proof_asts': proof_asts,
            'num_steps': len(proof_asts)
        }
    
    def _tokenize_context(self, text: str) -> List[int]:
        """Simple word-level tokenization"""
        # Split into words
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        # Convert to indices
        tokens = []
        for word in words:
            self.vocabulary.add_token(word)
            tokens.append(self.vocabulary.encode_token(word))
        
        # Truncate if needed
        if len(tokens) > self.max_context_length:
            tokens = tokens[:self.max_context_length]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get training example"""
        example = self.examples[idx]
        
        # For simplicity, return first AST as target
        # In full implementation, you'd iterate through all steps
        target_ast = example['proof_asts'][0] if example['proof_asts'] else None
        
        return {
            'theorem_id': example['theorem_id'],
            'context_tokens': torch.tensor(example['context_tokens'], dtype=torch.long),
            'target_ast': target_ast,
            'num_asts': len(example['proof_asts'])
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate batch of examples"""
    
    # Pad context tokens
    context_tokens = [item['context_tokens'] for item in batch]
    context_tokens_padded = pad_sequence(
        context_tokens,
        batch_first=True,
        padding_value=0
    )
    
    # Collect target ASTs
    target_asts = [item['target_ast'] for item in batch]
    
    return {
        'theorem_ids': [item['theorem_id'] for item in batch],
        'context_tokens': context_tokens_padded,
        'target_asts': target_asts,
        'num_asts': [item['num_asts'] for item in batch]
    }


def create_astactic_dataloaders(
    data_path: str,
    batch_size: int = 16,
    max_context_length: int = 256,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, TacticVocabulary]:
    """
    Create dataloaders for ASTactic training
    
    Args:
        data_path: Path to NaturalProofs data
        batch_size: Batch size
        max_context_length: Max context length
        num_workers: Number of workers
        cache_dir: Cache directory
    
    Returns:
        train_loader, val_loader, test_loader, vocabulary
    """
    
    # Create datasets
    train_dataset = ASTacticDataset(
        data_path=data_path,
        split='train',
        max_context_length=max_context_length,
        cache_dir=cache_dir
    )
    
    val_dataset = ASTacticDataset(
        data_path=data_path,
        split='val',
        max_context_length=max_context_length,
        cache_dir=cache_dir
    )
    
    test_dataset = ASTacticDataset(
        data_path=data_path,
        split='test',
        max_context_length=max_context_length,
        cache_dir=cache_dir
    )
    
    # Use same vocabulary for all splits
    vocabulary = train_dataset.vocabulary
    val_dataset.vocabulary = vocabulary
    test_dataset.vocabulary = vocabulary
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, vocabulary


if __name__ == "__main__":
    print("=== Testing ASTactic Data Loader ===\n")
    
    # Test on dummy data
    data_path = "./data/naturalproofs"
    
    try:
        train_loader, val_loader, test_loader, vocab = create_astactic_dataloaders(
            data_path=data_path,
            batch_size=4,
            cache_dir="./cache"
        )
        
        print(f"\nVocabulary size: {vocab.size()}")
        print(f"Number of node types: {len(TacticNodeType)}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Context tokens shape: {batch['context_tokens'].shape}")
        print(f"Number of examples: {len(batch['theorem_ids'])}")
        print(f"Target ASTs: {len(batch['target_asts'])}")
        
        # Show first example
        if batch['target_asts'][0]:
            print(f"\nFirst target AST: {batch['target_asts'][0].to_string()}")
        
        print("\n✓ Data loader test successful!")
        
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("Please ensure NaturalProofs data is available")
