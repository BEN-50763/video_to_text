"""
Transcribe audio files with speaker diarization using AssemblyAI.
"""

import assemblyai as aai


def transcribe_with_diarization(audio_path, api_key):
    """
    Transcribe audio file with speaker diarization.

    Args:
        audio_path: Path to audio file (local or URL)
        api_key: AssemblyAI API key

    Returns:
        List of dicts with speaker and text for each utterance

    Raises:
        Exception: If transcription fails
    """
    aai.settings.api_key = api_key

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config)

    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription failed: {transcript.error}")

    # Check if we got utterances
    if not transcript.utterances:
        raise Exception(f"Transcription completed but returned no utterances. Status: {transcript.status}")

    # Log success details
    print(f"Transcription completed successfully")
    print(f"Audio duration: {transcript.audio_duration}s ({transcript.audio_duration/60:.1f} min)")
    print(f"Utterances: {len(transcript.utterances)}")
    total_words = sum(len(utt.text.split()) for utt in transcript.utterances)
    print(f"Total words: {total_words}")

    results = []
    for utterance in transcript.utterances:
        results.append({
            'speaker': utterance.speaker,
            'text': utterance.text,
            'start': utterance.start,
            'end': utterance.end
        })

    return results


def transcribe_with_speaker_count(audio_path, api_key, min_speakers=None, max_speakers=None, expected_speakers=None):
    """
    Transcribe audio file with speaker diarization and speaker count constraints.

    Args:
        audio_path: Path to audio file (local or URL)
        api_key: AssemblyAI API key
        min_speakers: Minimum expected speakers
        max_speakers: Maximum expected speakers
        expected_speakers: Exact number of expected speakers

    Returns:
        List of dicts with speaker and text for each utterance
    """
    aai.settings.api_key = api_key

    if expected_speakers:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=expected_speakers
        )
    elif min_speakers or max_speakers:
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speaker_options=aai.SpeakerOptions(
                min_speakers_expected=min_speakers,
                max_speakers_expected=max_speakers
            )
        )
    else:
        config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config)

    results = []
    for utterance in transcript.utterances:
        results.append({
            'speaker': utterance.speaker,
            'text': utterance.text,
            'start': utterance.start,
            'end': utterance.end
        })

    return results
