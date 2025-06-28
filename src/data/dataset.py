import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict

class ASVspoof2019Dataset(Dataset):
    def __init__(self, 
                 audio_dir: str,
                 protocol_file: str,
                 max_length: int = 64000,  # 4 seconds at 16kHz
                 transform=None):
        """
        ASVspoof2019 Dataset for loading audio and labels
        
        Args:
            audio_dir: Directory containing FLAC audio files
            protocol_file: Path to protocol file (.txt)
            max_length: Maximum audio length in samples
            transform: Optional transform to apply to audio
        """
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.transform = transform
        
        # Load protocol file
        self.samples = self._load_protocol(protocol_file)
        
        # Label mapping
        self.label_map = {"bonafide": 0, "spoof": 1}
        
    def _load_protocol(self, protocol_file: str) -> List[Dict]:
        """Load protocol file and extract file info"""
        samples = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    sample_info = {
                        'speaker_id': parts[0],
                        'filename': parts[1],
                        'system_id': parts[3],
                        'label': parts[4]
                    }
                    samples.append(sample_info)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        filename = sample['filename']
        label = self.label_map[sample['label']]
        
        # Load audio file
        audio_path = os.path.join(self.audio_dir, f"{filename}.flac")
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        audio_tensor = torch.FloatTensor(audio)
        
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
            
        return audio_tensor, label, filename
