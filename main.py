import argparse
import os
import torch
import librosa
import numpy as np
from transformers import AutoProcessor, Wav2Vec2ForCTC
from tqdm import tqdm

def transcribe_audio_chunks(input_file, output_file, chunk_size_seconds=600, device="mps"):
    """
    Transcribe a long audio file by splitting it into chunks.
    
    Args:
        input_file: Path to the input audio file
        output_file: Path to save the transcription
        chunk_size_seconds: Size of each audio chunk in seconds
        device: Device to run inference on ('mps' for Apple Silicon, 'cpu' for Intel Macs)
    """
    print(f"Loading model on {device}...")
    
    # Use MPS (Metal Performance Shaders) for Apple Silicon or CPU for Intel Macs
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS doesn't support bfloat16 yet
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    # Load model and processor
    model = Wav2Vec2ForCTC.from_pretrained(
        "reazon-research/japanese-wav2vec2-large-rs35kh",
        torch_dtype=dtype,
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("reazon-research/japanese-wav2vec2-large-rs35kh")
    
    # Load audio
    print(f"Loading audio file: {input_file}")
    audio, sr = librosa.load(input_file, sr=16_000)
    
    # Calculate chunk size in samples
    chunk_size = chunk_size_seconds * sr
    
    # Calculate number of chunks
    total_samples = len(audio)
    num_chunks = int(np.ceil(total_samples / chunk_size))
    
    print(f"Audio length: {total_samples/sr:.2f} seconds")
    print(f"Processing {num_chunks} chunks of {chunk_size_seconds} seconds each")
    
    transcription = []
    
    # Process each chunk
    for i in tqdm(range(num_chunks)):
        # Extract chunk
        start = i * chunk_size
        end = min(start + chunk_size, total_samples)
        chunk = audio[start:end]
        
        # Pad chunk (0.5 seconds on each side)
        chunk = np.pad(chunk, pad_width=int(0.5 * sr))
        
        # Process chunk
        input_values = processor(
            chunk,
            return_tensors="pt",
            sampling_rate=sr
        ).input_values.to(device).to(dtype)
        
        # Transcribe chunk
        with torch.inference_mode():
            logits = model(input_values).logits.cpu()
        
        predicted_ids = torch.argmax(logits, dim=-1)[0]
        chunk_transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        
        # Add to overall transcription
        transcription.append(chunk_transcription)
    
    # Combine all chunks
    full_transcription = " ".join(transcription)
    
    # Save transcription
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_transcription)
    
    print(f"Transcription saved to: {output_file}")
    return full_transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Japanese ASR using wav2vec2")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output transcription file")
    parser.add_argument("--chunk_size", "-c", type=int, default=60, 
                        help="Size of each audio chunk in seconds (default: 600)")
    parser.add_argument("--device", "-d", type=str, default="mps", 
                        choices=["mps", "cpu"], 
                        help="Device to run inference on (mps for Apple Silicon, cpu for Intel Macs)")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    transcribe_audio_chunks(args.input, args.output, args.chunk_size, args.device)