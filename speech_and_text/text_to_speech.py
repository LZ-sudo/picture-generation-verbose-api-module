# text_to_speech.py
import os
import sys
import json
from dotenv import load_dotenv
from pathlib import Path
from elevenlabs.client import ElevenLabs

load_dotenv()

def text_to_speech(json_file_path):
    """
    Convert text from a JSON file to speech and save to audio_output directory.

    Args:
        json_file_path: Path to the JSON file containing the text to convert
    """
    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        sys.exit(1)

    # Read the JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the response text
    if "response" not in data:
        print("Error: 'response' field not found in JSON file")
        sys.exit(1)

    text_content = data["response"]
    print(f"Converting text to speech: {text_content[:50]}...")

    # Initialize ElevenLabs client
    elevenlabs = ElevenLabs(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
    )

    # Create audio_output directory if it doesn't exist
    output_dir = Path("audio_output")
    output_dir.mkdir(exist_ok=True)

    # Convert text to speech
    audio = elevenlabs.text_to_speech.convert(
        text=text_content,
        voice_id="19STyYD15bswVz51nqLf",
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_96",
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.8,
            "speed": 0.85  # Range: 0.7 (slower) to 1.2 (faster)
        }
    )

    # Generate output filename based on input filename
    input_filename = Path(json_file_path).stem
    output_file = output_dir / f"{input_filename}_audio.mp3"

    # Save audio to file
    with open(output_file, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print(f"Audio generation completed!")
    print(f"Output saved to: {output_file}")

    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python python speech_and_text/text_to_speech.py <path_to_json_file>")
        print("Example: python python speech_and_text/text_to_speech.py text_input/response_1_test.json")
        sys.exit(1)

    json_file_path = sys.argv[1]
    text_to_speech(json_file_path)

