"""
Main script to extract audio from videos and transcribe with speaker diarization.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from extract_audio import extract_audio
from transcribe_audio import transcribe_with_diarization

# Load environment variables
load_dotenv()

# Configuration
VIDEO_DIR = "../data/videos"
OUTPUT_DIR = "../data/transcripts"
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

# Get API key
api_key = os.getenv('ASSEMBLYAI_API_KEY')
if not api_key:
    print("Error: ASSEMBLYAI_API_KEY not found in .env file")
    exit(1)

# Setup paths
video_dir = Path(VIDEO_DIR)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

if not video_dir.exists():
    print(f"Error: Video directory not found: {VIDEO_DIR}")
    exit(1)

# Find video files
video_files = [
    f for f in video_dir.iterdir()
    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
]

if not video_files:
    print(f"No video files found in {VIDEO_DIR}")
    exit(0)

print(f"Found {len(video_files)} video file(s)\n")

# Process each video
for video_file in video_files:
    print(f"Processing: {video_file.name}")

    # Extract audio
    audio_file = output_dir / f"{video_file.stem}.mp3"
    print(f"Extracting audio to {audio_file.name}")
    extract_audio(video_file, audio_file)

    # Transcribe with diarization
    print(f"Transcribing with speaker diarization...")
    try:
        results = transcribe_with_diarization(str(audio_file), api_key)

        # Save transcript
        transcript_file = output_dir / f"{video_file.stem}_transcript.txt"
        with open(transcript_file, 'w') as f:
            for utterance in results:
                f.write(f"Speaker {utterance['speaker']}: {utterance['text']}\n")

        print(f"Saved transcript to {transcript_file.name}")
        print()
    except Exception as e:
        print(f"ERROR during transcription: {e}")
        print(f"Skipping this file and continuing...")
        print()

print(f"Completed processing {len(video_files)} file(s)")
