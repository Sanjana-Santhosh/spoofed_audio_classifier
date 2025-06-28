#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.dataset import ASVspoof2019Dataset
from src.data.preprocessing import create_data_loaders
from src.models.wav2vec_classifier import Wav2Vec2Classifier
from src.training.evaluate import ModelEvaluator
from src.training.config import TrainingConfig

def main():
    config = TrainingConfig()
    
    # Load test dataset
    dataset = ASVspoof2019Dataset(
        audio_dir=config.data_dir,
        protocol_file=config.protocol_file.replace('train.trn', 'eval.trl'),
        max_length=config.max_audio_length
    )
    
    _, _, test_loader = create_data_loaders(dataset, batch_size=config.batch_size)
    
    # Load model
    device = torch.device(config.device)
    model = Wav2Vec2Classifier().to(device)
    
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
