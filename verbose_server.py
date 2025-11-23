"""
Verbose Service API - WebSocket Voice Services
===============================================
Simple WebSocket server for speech-to-text and text-to-speech.
- Speech-to-Text: User speaks → Server transcribes → Returns text
- Text-to-Speech: Chatbot text → Server generates audio → Streams back

Architecture:
Uses plain FastAPI WebSocket (no FastRTC) for maximum simplicity.
Separate message types for STT and TTS operations.
No event loop blocking with async/await throughout.
"""

import os
import io
import base64
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
    title="Verbose Service (WebSocket Voice)",
    description="Simple WebSocket for speech-to-text and text-to-speech",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint - service info"""
    return {
        "status": "running",
        "service": "Verbose Service",
        "version": "2.0.0"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ELEVENLABS_API_KEY not configured"
        )

    return {
        "status": "healthy",
        "service": "Verbose Service (WebSocket)",
        "version": "2.0.0"
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice services.

    Protocol:
    Client sends:
    - {"type": "stt", "audio": "base64_encoded_audio_data"} → Transcribe audio
    - {"type": "tts", "text": "Text to speak"} → Generate and stream audio

    Server responds:
    - {"type": "transcript", "text": "transcribed text"} → Transcription result
    - {"type": "audio_chunk", "data": "base64_audio_chunk"} → TTS audio chunk
    - {"type": "audio_end"} → TTS streaming complete
    - {"type": "error", "message": "error description"} → Error occurred
    """
    from elevenlabs.client import ElevenLabs
    from starlette.concurrency import run_in_threadpool

    await websocket.accept()
    print("[WebSocket] Client connected")

    # Initialize ElevenLabs client
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        await websocket.send_json({
            "type": "error",
            "message": "ELEVENLABS_API_KEY not configured"
        })
        await websocket.close()
        return

    elevenlabs = ElevenLabs(api_key=api_key)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "stt":
                # Speech-to-Text request
                await handle_stt(websocket, elevenlabs, data)

            elif message_type == "tts":
                # Text-to-Speech request
                await handle_tts(websocket, elevenlabs, data)

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass


async def handle_stt(websocket: WebSocket, elevenlabs, data: dict):
    """
    Handle Speech-to-Text request.

    Args:
        websocket: WebSocket connection
        elevenlabs: ElevenLabs client
        data: Message data with base64-encoded audio
    """
    from starlette.concurrency import run_in_threadpool

    try:
        # Decode base64 audio data
        audio_b64 = data.get("audio")
        if not audio_b64:
            await websocket.send_json({
                "type": "error",
                "message": "No audio data provided"
            })
            return

        audio_bytes = base64.b64decode(audio_b64)
        print(f"[STT] Received audio: {len(audio_bytes)} bytes")

        # Create file-like object for ElevenLabs
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.webm"

        # Transcribe audio (blocking call, run in thread pool)
        def transcribe():
            return elevenlabs.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                language_code="eng"
            )

        print("[STT] Transcribing...")
        transcription = await run_in_threadpool(transcribe)

        # Extract text from transcription
        if hasattr(transcription, 'text'):
            text = transcription.text
        else:
            import re
            text_str = str(transcription)
            match = re.search(r"text='(.*?)'", text_str)
            text = match.group(1) if match else text_str

        print(f"[STT] SUCCESS Transcript: '{text}'")

        # Send transcript back to client
        await websocket.send_json({
            "type": "transcript",
            "text": text
        })

    except Exception as e:
        print(f"[STT] ERROR: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": f"STT error: {str(e)}"
        })


async def handle_tts(websocket: WebSocket, elevenlabs, data: dict):
    """
    Handle Text-to-Speech request.

    Args:
        websocket: WebSocket connection
        elevenlabs: ElevenLabs client
        data: Message data with text to convert
    """
    from starlette.concurrency import run_in_threadpool

    try:
        text = data.get("text")
        if not text:
            await websocket.send_json({
                "type": "error",
                "message": "No text provided"
            })
            return

        print(f"[TTS] Generating speech for: '{text[:50]}...'")

        # Generate TTS audio (blocking call, run in thread pool)
        # Using flash model and lower quality output to reduce credit consumption
        def generate_tts():
            return elevenlabs.text_to_speech.convert(
                text=text,
                voice_id="19STyYD15bswVz51nqLf",
                model_id="eleven_flash_v2_5",  # Flash model - fastest and cheapest
                output_format="mp3_22050_32",  # Lower quality = fewer credits, change to mp3_44100_96 for prod
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,  # Slightly reduced for flash
                    "speed": 0.85
                }
            )

        audio_stream = await run_in_threadpool(generate_tts)

        # Stream audio chunks to client
        chunk_count = 0
        for chunk in audio_stream:
            # Encode chunk as base64 and send
            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
            await websocket.send_json({
                "type": "audio_chunk",
                "data": chunk_b64
            })
            chunk_count += 1

        # Send completion message
        await websocket.send_json({
            "type": "audio_end"
        })

        print(f"[TTS] SUCCESS Streamed {chunk_count} audio chunks")

    except Exception as e:
        print(f"[TTS] ERROR: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({
            "type": "error",
            "message": f"TTS error: {str(e)}"
        })


# =============================================================================
# Legacy REST Endpoints (deprecated)
# =============================================================================

@app.post("/speech-to-text")
def speech_to_text_legacy():
    """Legacy REST endpoint - use WebSocket instead"""
    raise HTTPException(
        status_code=410,
        detail="Use WebSocket at ws://localhost:8003/ws with message type 'stt'"
    )


@app.post("/text-to-speech")
def text_to_speech_legacy():
    """Legacy REST endpoint - use WebSocket instead"""
    raise HTTPException(
        status_code=410,
        detail="Use WebSocket at ws://localhost:8003/ws with message type 'tts'"
    )


# =============================================================================
# Server Startup
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("VERBOSE_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("VERBOSE_SERVICE_PORT", "8003"))

    print("=" * 80)
    print("Verbose Service - WebSocket Voice Services")
    print("=" * 80)
    print(f"Server URL: http://{host}:{port}")
    print(f"WebSocket: ws://{host}:{port}/ws")
    print(f"Health check: http://{host}:{port}/health")
    print("=" * 80)
    print("Protocol:")
    print("  - Send: {type: 'stt', audio: 'base64...'} -> Get transcript")
    print("  - Send: {type: 'tts', text: '...'} -> Get audio chunks")
    print("=" * 80)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=300
    )
