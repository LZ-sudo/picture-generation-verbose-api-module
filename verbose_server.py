"""
Verbose Service API (Speech-to-Text & Text-to-Speech)
======================================================
FastAPI server dedicated to speech services only.
Isolated from image processing to prevent event loop conflicts.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import re

from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Load shared environment variables
from dotenv import load_dotenv
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Initialize FastAPI app
app = FastAPI(
    title="Verbose Service (Speech)",
    description="Speech-to-text and text-to-speech services",
    version="1.0.0"
)


class SpeechToTextResponse(BaseModel):
    """Response model for speech-to-text"""
    success: bool
    transcript: Optional[str] = None
    transcript_file: Optional[str] = None
    json_file: Optional[str] = None
    error: Optional[str] = None


class TextToSpeechResponse(BaseModel):
    """Response model for text-to-speech"""
    success: bool
    audio_file_path: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - service info"""
    return {
        "status": "running",
        "service": "Verbose Service",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Verify API key is set
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ELEVENLABS_API_KEY not configured"
        )

    return {
        "status": "healthy",
        "service": "Verbose Service",
        "version": "1.0.0"
    }


@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text_endpoint(
    file: UploadFile = File(...)
):
    """
    Transcribe an audio file to text using ElevenLabs.

    Args:
        file: Audio file (MP3 format)

    Returns:
        Transcript text and paths to output files
    """
    temp_audio = None

    try:
        # Setup temp directory
        import tempfile
        system_temp = Path(tempfile.gettempdir())
        temp_dir = system_temp / "hks_spatial_stt"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Output directory for transcripts
        output_dir = temp_dir / "text_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

        # Save uploaded audio file
        temp_audio = temp_dir / f"audio_{timestamp}_{file.filename}"
        with open(temp_audio, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"Speech-to-text request received:")
        print(f"  Input: {temp_audio}")

        # Verify ElevenLabs API key
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")

        # Import ElevenLabs client
        from elevenlabs.client import ElevenLabs

        # Initialize ElevenLabs client
        elevenlabs = ElevenLabs(api_key=api_key)

        # Transcribe audio
        with open(temp_audio, "rb") as audio_file:
            transcription = elevenlabs.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )

        # Generate output filename
        input_filename = Path(temp_audio).stem
        transcript_file = output_dir / f"{input_filename}_transcript.txt"

        # Save transcription to file
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(str(transcription))

        print(f"Transcription completed!")
        print(f"Output saved to: {transcript_file}")

        # Extract text content and save as JSON
        with open(transcript_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the text field using regex
        # Try single quotes first (ElevenLabs uses single quotes)
        match = re.search(r"text='(.*?)'\s+words=", content)

        if not match:
            # Fallback: try double quotes
            match = re.search(r'text="(.*?)"\s+words=', content)

        if match:
            text_content = match.group(1)
        else:
            # If regex fails, return the full transcription object as fallback
            text_content = str(transcription)

        # Create JSON output
        json_data = {
            "transcript": text_content
        }

        # Generate JSON filename
        json_file = transcript_file.with_suffix('.json')

        # Save to JSON file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"JSON output saved to: {json_file}")

        return {
            "success": True,
            "transcript": text_content,
            "transcript_file": str(transcript_file),
            "json_file": str(json_file)
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in speech-to-text endpoint: {error_detail}")
        return {
            "success": False,
            "error": error_detail
        }

    finally:
        # Clean up temp audio file
        if temp_audio and temp_audio.exists():
            try:
                temp_audio.unlink()
            except:
                pass


@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech_endpoint(
    text: str = Body(..., embed=True)
):
    """
    Convert text to speech using ElevenLabs.

    Args:
        text: Text content to convert to speech

    Returns:
        Path to generated audio file
    """
    try:
        # Setup temp directory
        import tempfile
        system_temp = Path(tempfile.gettempdir())
        temp_dir = system_temp / "hks_spatial_tts"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Output directory for audio
        output_dir = temp_dir / "audio_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

        print(f"Text-to-speech request received:")
        print(f"  Text: {text[:100]}...")

        # Verify ElevenLabs API key
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY not configured")

        # Import ElevenLabs client
        from elevenlabs.client import ElevenLabs

        # Initialize ElevenLabs client
        elevenlabs = ElevenLabs(api_key=api_key)

        # Convert text to speech
        audio = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id="19STyYD15bswVz51nqLf",
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_96",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.8,
                "speed": 0.85
            }
        )

        # Generate output filename
        output_file = output_dir / f"audio_{timestamp}.mp3"

        # Save audio to file
        with open(output_file, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        print(f"Audio generation completed!")
        print(f"Output saved to: {output_file}")

        return {
            "success": True,
            "audio_file_path": str(output_file)
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in text-to-speech endpoint: {error_detail}")
        return {
            "success": False,
            "error": error_detail
        }


@app.get("/download-audio/{filename}")
async def download_audio_file(filename: str):
    """
    Download a generated audio file.

    Args:
        filename: Name of the audio file

    Returns:
        File response with the audio
    """
    import tempfile
    system_temp = Path(tempfile.gettempdir())
    temp_dir = system_temp / "hks_spatial_tts" / "audio_output"
    file_path = temp_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename
    )


@app.get("/download-transcript/{filename}")
async def download_transcript_file(filename: str):
    """
    Download a transcript file (text or json).

    Args:
        filename: Name of the transcript file

    Returns:
        File response with the transcript
    """
    import tempfile
    system_temp = Path(tempfile.gettempdir())
    temp_dir = system_temp / "hks_spatial_stt" / "text_output"
    file_path = temp_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Determine media type based on extension
    media_type = "application/json" if filename.endswith(".json") else "text/plain"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("VERBOSE_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("VERBOSE_SERVICE_PORT", "8003"))

    print(f"Starting Verbose Service (Speech) on {host}:{port}")
    print("Dedicated server for speech-to-text and text-to-speech")
    print("Isolated from image processing to prevent conflicts")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=120  # 2 minutes keep-alive (shorter than image service)
    )
