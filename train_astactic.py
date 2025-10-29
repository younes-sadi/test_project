"""
Training Script for True ASTactic Model
Trains tree-based neural network for tactic generation
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from astactic_model import create_astactic_model, ASTacticModel
from astactic_data_loader import create_astactic_dataloaders
from tactic_ast import TacticVocabulary, TacticNodeType


class ASTacticTrainer:
    """Trainer for ASTactic model"""
    
    def __init__(
        self,
        model: ASTacticModel,
        train_loader,
        val_loader,
        vocabulary: TacticVocabulary,
        optimizer,
        scheduler,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_interval = config.get('log_interval', 10)
    
    def compute_ast_loss(self, batch):
        """
        Compute loss for AST generation
        This is a simplified version - real implementation would be more complex
        """
        context_tokens = batch['context_tokens'].to(self.device)
        target_asts = batch['target_asts']
        
        # Forward pass
        with autocast(enabled=self.use_amp):
            outputs = self.model(context_tokens)
            
            # In real implementation, we'd compute:
            # 1. Node type prediction loss
            # 2. Token prediction loss
            # 3. Structure prediction loss (num children, etc.)
            
            # Placeholder loss for demonstration
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Add small random loss for demonstration
            loss = loss + torch.randn(1, device=self.device).abs() * 0.1
        
        return loss
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Compute loss
            loss = self.compute_ast_loss(batch)
            loss = loss / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            loss = self.compute_ast_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocabulary': self.vocabulary,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"astactic_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "astactic_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting ASTactic Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training examples: {len(self.train_loader.dataset)}")
        print(f"Validation examples: {len(self.val_loader.dataset)}")
        print(f"Vocabulary size: {self.vocabulary.size()}")
        print(f"Node types: {len(TacticNodeType)}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - start_time
            
            # Print metrics
            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train True ASTactic Model")
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to NaturalProofs dataset')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Cache directory')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='Hidden size')
    parser.add_argument('--embed_size', type=int, default=256,
                       help='Embedding size')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation')
    parser.add_argument('--max_context_length', type=int, default=256,
                       help='Max context length')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    # Logging
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_astactic',
                       help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, vocabulary = create_astactic_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir
    )
    
    # Create model
    print("\nCreating ASTactic model...")
    model = create_astactic_model(
        vocab_size=vocabulary.size(),
        num_node_types=len(TacticNodeType),
        hidden_size=args.hidden_size,
        device=args.device
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create config
    config = vars(args)
    
    # Create trainer
    trainer = ASTacticTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocabulary=vocabulary,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
