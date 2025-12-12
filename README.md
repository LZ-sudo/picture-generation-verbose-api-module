# Picture Generation & Verbose API Module

A comprehensive microservices module for AI-powered spatial design analysis, object detection, image transformation, and voice interaction. This module provides REST and WebSocket APIs for processing interior design images with LLM-based recommendations, Florence-2 object detection, and Nano Banana Pro image editing.

## Overview

This module consists of three main microservices:

1. **Image Generation Service** - AI-powered image transformation based on design recommendations
2. **Object Detection Service** - Florence-2 based spatial coordinate identification
3. **Verbose Service** - WebSocket-based speech-to-text and text-to-speech

## Setup (Standalone)

### Prerequisites

- Python 3.8+
- pip
- Virtual environment support

### Installation

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/LZ-sudo/picture-generation-verbose-api-module.git

   cd picture-generation-verbose-api-module
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv myenv

   # Activate on Windows
   myenv\Scripts\activate

   # Activate on macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory with the following variables:
   ```env
   # API Keys
   NANOBANANA_API_KEY=your_google_gemini_api_key
   ELEVENLABS_API_KEY=your_elevenlabs_api_key

   # Service Ports (optional, defaults shown)
   IMAGE_GEN_SERVICE_HOST=127.0.0.1
   IMAGE_GEN_SERVICE_PORT=8002
   VERBOSE_SERVICE_HOST=127.0.0.1
   VERBOSE_SERVICE_PORT=8003
   DETECTION_SERVICE_HOST=127.0.0.1
   DETECTION_SERVICE_PORT=8004
   ```

5. **Initialize the interior-segment-labeler submodule (if needed):**
   ```bash
   git submodule update --init --recursive
   ```

## Running the Services

### Start Individual Services

**Image Generation Service:**
```bash
python image_server.py
```
Runs on `http://127.0.0.1:8002`

**Object Detection Service:**
```bash
python detection_server.py
```
Runs on `http://127.0.0.1:8004`

**Verbose Service (WebSocket):**
```bash
python verbose_server.py
```
Runs on `http://127.0.0.1:8003` with WebSocket at `ws://127.0.0.1:8003/ws`

## Scripts Documentation

### Core API Servers

#### [image_server.py](image_server.py)
FastAPI microservice for image transformation.

**Purpose:** Receives an original image and JSON analysis with design recommendations, then orchestrates sequential image editing using Nano Banana Pro (Gemini 3 Pro).

**Endpoints:**
- `GET /` - Service info and status
- `GET /health` - Health check (validates API key)
- `POST /transform` - Transform image based on recommendations
  - Parameters: `file` (image), `analysis_json` (JSON string)
  - Returns: Path to transformed image
- `GET /download/{filename}` - Download transformed image

**Features:**
- Asyncio subprocess execution to prevent threading deadlocks
- Support for long-running transformations (30+ minute timeout)
- Automatic temp file cleanup

#### [detection_server.py](detection_server.py)
FastAPI microservice for object detection and spatial coordinate identification.

**Purpose:** Uses Florence-2 model to detect objects mentioned in design recommendations and add bounding box coordinates to the analysis JSON.

**Endpoints:**
- `GET /` - Service info
- `GET /health` - Health check (validates Florence-2 availability)
- `POST /identify` - Identify objects and add bounding boxes
  - Parameters: `file` (image), `analysis_json` (JSON string)
  - Returns: Updated analysis JSON with bounding box coordinates

**Features:**
- Lazy-loaded Florence-2 detector (singleton pattern)
- Normalized bounding box coordinates (0-1 range)
- Automatic intermediate file cleanup

#### [verbose_server.py](verbose_server.py)
FastAPI WebSocket server for voice services.

**Purpose:** Provides real-time speech-to-text and text-to-speech capabilities using ElevenLabs API.

**Endpoints:**
- `GET /` - Service info
- `GET /health` - Health check (validates API key)
- `WebSocket /ws` - Main WebSocket endpoint for voice services

**WebSocket Protocol:**
- Send: `{"type": "stt", "audio": "base64_encoded_audio"}` → Receive transcript
- Send: `{"type": "tts", "text": "Text to speak"}` → Receive audio chunks
- Receive: `{"type": "transcript", "text": "..."}` - Transcription result
- Receive: `{"type": "audio_chunk", "data": "base64..."}` - TTS audio chunk
- Receive: `{"type": "audio_end"}` - TTS streaming complete
- Receive: `{"type": "error", "message": "..."}` - Error message

**Features:**
- Async WebSocket communication
- Base64-encoded audio streaming
- ElevenLabs Scribe v1 for STT, Flash v2.5 for TTS

### Standalone Scripts

#### [transform_image.py](transform_image.py)
Sequential image editing orchestrator for batch processing.

**Purpose:** Processes multiple design recommendations sequentially, applying each edit to the result of the previous edit.

**Usage:**
```bash
python transform_image.py <image> <editing_prompts.json> [output] [--keep-intermediates]
```

**Examples:**
```bash
# Basic usage
python transform_image.py room.jpg prompts.json

# Custom output path
python transform_image.py room.jpg prompts.json final_room.jpg

# Keep intermediate files for debugging
python transform_image.py room.jpg prompts.json output.jpg --keep-intermediates
```

**Input Format (editing_prompts.json):**
```json
{
  "analysis_summary": {"total_issues": 3},
  "issues": [
    {
      "item": "Coffee Table",
      "issue": "Description of issue...",
      "guideline_reference": "Reference...",
      "recommendation": "Paint the legs dark charcoal grey..."
    }
  ]
}
```

**Features:**
- Sequential editing pipeline
- Automatic intermediate file management
- Enhanced prompts with object attention guidance
- Uses Gemini 3 Pro Image Preview (Nano Banana Pro)

#### [identify_changes.py](identify_changes.py)
Batch object detection script for offline processing.

**Purpose:** Processes analysis JSON to identify and locate objects in images using Florence-2 detection, adding bounding box coordinates to each recommendation.

**Usage:**
```bash
python identify_changes.py <analysis.json> <image.jpg> [options]
```

**Examples:**
```bash
# Basic usage
python identify_changes.py analysis.json room.jpg

# Custom output file
python identify_changes.py analysis.json room.jpg --output results.json

# Custom output directory
python identify_changes.py analysis.json room.jpg --output-dir results/

# Print to console only
python identify_changes.py analysis.json room.jpg --no-save
```

**Features:**
- Reuses single detector instance for all detections (efficient)
- Normalized bounding box coordinates
- IOU-based detection filtering
- Automatic detector cleanup

#### [nanobanana_edit.py](nanobanana_edit.py)
Core image editing wrapper for Gemini 3 Pro API.

**Purpose:** Single-image editing using Google's Gemini 3 Pro Image Preview model with optional reference/highlight images.

**Usage:**
```bash
python nanobanana_edit.py <original_image> <prompt.json> [output.png]
```

**Examples:**
```bash
# Basic editing
python nanobanana_edit.py original.jpg prompt.json

# With custom output
python nanobanana_edit.py original.jpg prompt.json edited.png
```

**Prompt JSON Format (basic):**
```json
{
  "prompt": "Edit description here"
}
```

**Prompt JSON Format (with reference image):**
```json
{
  "prompt": "Replace the highlighted areas with a modern sofa",
  "reference_image": "results/room_highlight.jpg"
}
```

**Features:**
- Multi-image input support (reference + original)
- Gemini 3 Pro Image Preview model
- Image-only response mode

### Speech & Text Utilities

#### [speech_and_text/speech_to_text.py](speech_and_text/speech_to_text.py)
Standalone speech-to-text transcription utility.

**Purpose:** Transcribe audio files using ElevenLabs Scribe v1 model.

**Usage:**
```bash
python speech_and_text/speech_to_text.py <path_to_audio_file>
```

**Example:**
```bash
python speech_and_text/speech_to_text.py audio.mp3
```

**Output:**
- `text_output/{filename}_transcript.txt` - Full transcript with metadata
- `text_output/{filename}_transcript.json` - Extracted text in JSON format

**Features:**
- Audio event tagging (laughter, applause, etc.)
- Speaker diarization
- Automatic text extraction to JSON

#### [speech_and_text/text_to_speech.py](speech_and_text/text_to_speech.py)
Standalone text-to-speech generation utility.

**Purpose:** Convert text from JSON files to speech audio using ElevenLabs.

**Usage:**
```bash
python speech_and_text/text_to_speech.py <path_to_json_file>
```

**Example:**
```bash
python speech_and_text/text_to_speech.py text_input/response.json
```

**Input Format:**
```json
{
  "response": "Text to convert to speech"
}
```

**Output:**
- `audio_output/{filename}_audio.mp3` - Generated audio file

**Features:**
- ElevenLabs Turbo v2.5 model
- Configurable voice settings (stability, similarity, speed)
- High-quality MP3 output

## API Examples

### Image Transformation

```python
import requests

url = "http://127.0.0.1:8002/transform"
files = {"file": open("room.jpg", "rb")}
data = {
    "analysis_json": json.dumps({
        "issues": [
            {
                "item": "Wall paint",
                "recommendation": "Paint walls sage green"
            }
        ]
    })
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result["transformed_image_path"])
```

### Object Detection

```python
import requests

url = "http://127.0.0.1:8004/identify"
files = {"file": open("room.jpg", "rb")}
data = {
    "analysis_json": json.dumps({
        "issues": [
            {
                "item": "Sofa",
                "recommendation": "Replace old sofa"
            }
        ]
    })
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result["analysis_with_boxes"])
```

### WebSocket Voice (JavaScript)

```javascript
const ws = new WebSocket('ws://127.0.0.1:8003/ws');

// Speech-to-Text
ws.send(JSON.stringify({
  type: 'stt',
  audio: base64AudioData
}));

// Text-to-Speech
ws.send(JSON.stringify({
  type: 'tts',
  text: 'Hello, this is a test'
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'transcript') {
    console.log('Transcript:', data.text);
  } else if (data.type === 'audio_chunk') {
    // Play audio chunk
    playAudioChunk(data.data);
  }
};
```

## Directory Structure

```
picture-generation-verbose-api-module/
├── image_server.py                    # Image transformation API server
├── detection_server.py                # Object detection API server
├── verbose_server.py                  # WebSocket voice services server
├── transform_image.py                 # Batch image transformation script
├── identify_changes.py                # Batch object detection script
├── nanobanana_edit.py                 # Single image editing wrapper
├── speech_and_text/
│   ├── speech_to_text.py             # Audio transcription utility
│   └── text_to_speech.py             # Text-to-speech generation utility
├── interior-segment-labeler/          # Florence-2 detection submodule
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (create this)
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## Troubleshooting

### Common Issues

**API Key Errors:**
- Ensure `.env` file exists and contains valid API keys
- Check that `NANOBANANA_API_KEY` and `ELEVENLABS_API_KEY` are set

**Import Errors:**
- Activate virtual environment: `myenv\Scripts\activate` (Windows) or `source myenv/bin/activate` (macOS/Linux)
- Reinstall dependencies: `pip install -r requirements.txt`

**Florence-2 Model Not Loading:**
- Ensure `interior-segment-labeler` submodule is initialized
- Check available system memory (model requires ~2GB)

**Long Transformation Times:**
- Gemini 3 Pro can take 30+ minutes for complex edits
- Use `--keep-intermediates` flag to debug progress

**WebSocket Connection Issues:**
- Verify service is running: `curl http://127.0.0.1:8003/health`
- Check firewall settings allow local connections

## Dependencies

Key packages (see [requirements.txt](requirements.txt) for full list):
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for Florence-2
- `transformers` - Hugging Face models
- `google-genai` - Gemini 3 Pro API
- `elevenlabs` - Voice services API
- `opencv-python` - Image processing
- `pillow` - Image handling

## License

This module is part of the HKS Spatial project.

## Contributing

For development:
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## Support

For issues or questions, please check:
- Service health endpoints (`/health`)
- Log output from running services
- API documentation at `http://localhost:{port}/docs` when services are running
