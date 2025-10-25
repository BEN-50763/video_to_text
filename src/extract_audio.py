"""
Extract audio from video files using MoviePy library.
"""
import os
from moviepy import VideoFileClip


def extract_audio(video_path, output_path):
    """Extract audio from video file using MoviePy."""
    if os.path.exists(output_path):
        print("Audio already exists, skipping {video_path}")
    else:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(output_path))
        clip.close()
