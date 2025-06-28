import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Tuple

from ..data.dataset import ASVspoof2019Dataset
from ..data.preprocessing import create_data_loaders
from ..models.wav2vec_classisfier import Wav2Vec2Classifier
from ..training.config import TrainingConfig

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = Wav2Vec2Classifier(
            model_name=config.model_name,
            num_classes=config.num_classes,
            freeze_feature_extractor=config.freeze_feature_extractor,
            freeze_transformer_layers=config.freeze_transformer_layers
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.__dict__
            )
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (audio, targets, _) in enumerate(progress_bar):
            audio = audio.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_val
            )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for audio, targets, _ in tqdm(val_loader, desc="Validation"):
                audio = audio.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(audio)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config.model_save_dir, 'checkpoint_latest.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(
                self.config.model_save_dir, 'best_model.pth'
            )
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with validation loss: {val_metrics['loss']:.4f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint(epoch, val_metrics)
                
                # Early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Log metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1_score'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_score'],
                    'val_auc': val_metrics['auc'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                
                if self.config.use_wandb:
                    wandb.log(metrics)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                      f"Val AUC: {val_metrics['auc']:.4f}")

def main():
    """Main training function"""
    config = TrainingConfig()
    
    # Create directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Load dataset
    dataset = ASVspoof2019Dataset(
        audio_dir=config.data_dir,
        protocol_file=config.protocol_file,
        max_length=config.max_audio_length
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
