import torch
import librosa
import numpy as np
from typing import Tuple, Dict

from ..models.wav2vec_classifier import Wav2Vec2Classifier
from ..training.config import TrainingConfig

class SpoofingDetector:
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the spoofing detector
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training config (optional)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        self.model = Wav2Vec2Classifier()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class labels
        self.labels = {0: 'bonafide', 1: 'spoof'}
        
    def predict_audio_file(self, audio_path: str) -> Dict[str, float]:
        """
        Predict if audio file is bonafide or spoofed
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        return self.predict_audio_array(audio)
    
    def predict_audio_array(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Predict if audio array is bonafide or spoofed
        
        Args:
            audio: Audio array (numpy)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess audio
        max_length = 64000  # 4 seconds at 16kHz
        
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(audio_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
        
        # Format results
        pred_class = prediction.item()
        pred_prob = probabilities[0, pred_class].item()
        
        return {
            'prediction': self.labels[pred_class],
            'confidence': pred_prob,
            'bonafide_probability': probabilities[0, 0].item(),
            'spoof_probability': probabilities[0, 1].item()
        }

def main():
    """Example usage"""
    detector = SpoofingDetector('models/best_model.pth')
    
    # Test with audio file
    audio_file = 'path/to/test/audio.wav'
    result = detector.predict_audio_file(audio_file)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Bonafide probability: {result['bonafide_probability']:.3f}")
    print(f"Spoof probability: {result['spoof_probability']:.3f}")

if __name__ == "__main__":
    main()
