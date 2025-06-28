import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from typing import Optional

from .predict import SpoofingDetector

class RealTimeDetector:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """
        Initialize real-time spoofing detector
        
        Args:
            model_path: Path to trained model
            sample_rate: Audio sample rate
        """
        self.detector = SpoofingDetector(model_path)
        self.sample_rate = sample_rate
        
    def record_and_predict(self, duration: float = 4.0) -> dict:
        """
        Record audio and predict if it's spoofed
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Prediction results
        """
        print(f"Recording for {duration} seconds...")
        print("Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        print("Recording completed. Analyzing...")
        
        # Flatten audio if needed
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        
        # Predict
        result = self.detector.predict_audio_array(audio_data)
        
        return result
    
    def save_recording(self, 
                      audio_data: np.ndarray, 
                      filename: Optional[str] = None) -> str:
        """
        Save recorded audio to file
        
        Args:
            audio_data: Audio array
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"recording_{np.random.randint(1000, 9999)}.wav"
        
        # Convert to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav.write(filename, self.sample_rate, audio_int16)
        return filename
    
    def interactive_mode(self):
        """
        Interactive mode for continuous testing
        """
        print("=== Real-time Spoofing Detection ===")
        print("Commands:")
        print("  'r' - Record and test")
        print("  'q' - Quit")
        print("=====================================")
        
        while True:
            command = input("\nEnter command (r/q): ").strip().lower()
            
            if command == 'q':
                print("Goodbye!")
                break
            elif command == 'r':
                try:
                    result = self.record_and_predict()
                    
                    print("\n--- Results ---")
                    print(f"Prediction: {result['prediction'].upper()}")
                    print(f"Confidence: {result['confidence']:.1%}")
                    print(f"Bonafide probability: {result['bonafide_probability']:.1%}")
                    print(f"Spoof probability: {result['spoof_probability']:.1%}")
                    print("---------------")
                    
                    # Color-coded output
                    if result['prediction'] == 'bonafide':
                        print("✅ GENUINE SPEECH DETECTED")
                    else:
                        print("⚠️  SYNTHETIC SPEECH DETECTED")
                        
                except Exception as e:
                    print(f"Error during recording/prediction: {e}")
            else:
                print("Invalid command. Use 'r' to record or 'q' to quit.")

def main():
    """Main function for interactive testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time spoofing detection')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--duration', type=float, default=4.0, 
                       help='Recording duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealTimeDetector(args.model)
    
    # Start interactive mode
    detector.interactive_mode()

if __name__ == "__main__":
    main()
 