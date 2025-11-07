# speech_to_text.py
import os
import sys
import json
import re
from dotenv import load_dotenv
from pathlib import Path
from elevenlabs.client import ElevenLabs

load_dotenv()

def transcribe_audio(audio_file_path):
    """
    Transcribe an MP3 file and save the output to text_output directory.

    Args:
        audio_file_path: Path to the MP3 file to transcribe
    """
    # Initialize ElevenLabs client
    elevenlabs = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )

    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        sys.exit(1)

    # Create text_output directory if it doesn't exist
    output_dir = Path("text_output")
    output_dir.mkdir(exist_ok=True)

    print(f"Transcribing: {audio_file_path}")

    # Open and read the audio file
    with open(audio_file_path, "rb") as audio_file:
        transcription = elevenlabs.speech_to_text.convert(
            file=audio_file,
            model_id="scribe_v1",  # Model to use, for now only "scribe_v1" is supported
            tag_audio_events=True,  # Tag audio events like laughter, applause, etc.
            language_code="eng",  # Language of the audio file. If set to None, the model will detect the language automatically.
            diarize=True,  # Whether to annotate who is speaking
        )

    # Generate output filename based on input filename
    input_filename = Path(audio_file_path).stem
    output_file = output_dir / f"{input_filename}_transcript.txt"

    # Save transcription to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(transcription))

    print(f"Transcription completed!")
    print(f"Output saved to: {output_file}")

    # Extract text content and save as JSON
    extract_text_to_json(output_file)

    return transcription

def extract_text_to_json(transcript_file_path):
    """
    Extract the text content from a transcript file and save it as JSON.

    Args:
        transcript_file_path: Path to the transcript .txt file
    """
    # Read the transcript file
    with open(transcript_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the text field using regex - capture only the text between the quotes before " words="
    match = re.search(r'text="(.*?)"\s+words=', content)

    if match:
        text_content = match.group(1)
    else:
        print(f"Warning: Could not extract text from {transcript_file_path}")
        return

    # Create JSON output
    json_data = {
        "transcript": text_content
    }

    # Generate JSON filename
    json_file = Path(transcript_file_path).with_suffix('.json')

    # Save to JSON file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON output saved to: {json_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python speech_and_text/speech_to_text.py <path_to_mp3_file>")
        print("Example: python speech_and_text/speech_to_text.py audio.mp3")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    transcribe_audio(audio_file_path)
