# Picture Generation & Verbose API Module

A comprehensive microservices module for AI-powered spatial design analysis, object detection, image transformation, and voice interaction. This module provides REST and WebSocket APIs for processing interior design images with LLM-based recommendations, Florence-2 object detection, segmentation-based masking, and Gemini 2.5 Flash image editing (Nano Banana).

## Overview

This module consists of three main microservices:

1. **Image Generation Service** - Segmentation-based AI image transformation with Gemini 2.5 Flash
2. **Object Detection Service** - Florence-2 based spatial coordinate identification
3. **Verbose Service** - WebSocket-based speech-to-text and text-to-speech

## Key Features

- **Segmentation-Based Editing**: Uses Florence-2 to segment specific objects before editing for precise, targeted transformations
- **Sequential Processing**: Applies multiple edits in sequence, using each output as input for the next edit
- **Multi-Image Context**: Sends both segmented highlight masks and original images to guide AI editing
- **Real-Time Voice Services**: WebSocket-based speech-to-text and text-to-speech with ElevenLabs
- **Microservice Architecture**: Independent services that can run separately or together

## Setup (Standalone)

### Prerequisites

- Python 3.8+
- pip
- Virtual environment support
- Git (for submodule management)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd picture-generation-verbose-api-module
   ```

2. **Initialize the segmentation submodule:**
   ```bash
   git submodule update --init --recursive
   ```
   This will clone the `interior-segment-labeler` submodule required for object segmentation.

3. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv myenv

   # Activate on Windows
   myenv\Scripts\activate

   # Activate on macOS/Linux
   source myenv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables:**
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
FastAPI microservice for segmentation-based image transformation.

**Purpose:** Receives an original image and JSON analysis with design recommendations, then orchestrates sequential image editing using segmentation + Gemini 2.5 Flash.

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
- Integrated segmentation pipeline

**Workflow:**
1. Receives image and analysis JSON
2. Calls `transform_image.py` as subprocess
3. Returns path to final transformed image

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
- IOU-based detection filtering

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
- Optimized credit usage with lower quality settings

### Standalone Scripts

#### [transform_image.py](transform_image.py)
Sequential segmentation-based image editing orchestrator for batch processing.

**Purpose:** Processes multiple design recommendations sequentially using a 3-step pipeline: segment → edit → update. Each edit uses the output of the previous edit as input.

**Workflow:**
1. **Segment**: Uses Florence-2 to detect and highlight the target object
2. **Edit**: Sends both the highlight mask and original image to Gemini 2.5 Flash
3. **Update**: Uses the edited image as input for the next recommendation

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
- **Segmentation-first approach**: Accurately identifies target objects before editing
- **Multi-image context**: Provides both highlight mask and original image to AI
- **Sequential pipeline**: Each edit builds on the previous result
- **Automatic intermediate file management**
- **Uses Gemini 2.5 Flash Image** for fast, efficient editing
- **Debug mode**: `--keep-intermediates` preserves all intermediate files

**Important Notes:**
- Requires `interior-segment-labeler` submodule for segmentation
- Creates `intermediate_files/` directory during processing
- Each issue is processed in 3 steps: segment → edit → update
- Segmentation uses Florence-2 detection with item name as prompt

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
- Normalized bounding box coordinates (0-1 range)
- IOU-based detection filtering (threshold: 0.7)
- Automatic detector cleanup
- Filters action words and duplicates

**Output Format:**
Adds `bounding_box_coordinates` to each issue:
```json
{
  "item": "Coffee Table",
  "recommendation": "...",
  "bounding_box_coordinates": {
    "format": "normalized",
    "count": 2,
    "detections": [
      {
        "label": "coffee table",
        "bbox": [0.2, 0.3, 0.5, 0.6],
        "center": [0.35, 0.45],
        "confidence": 0.95
      }
    ]
  }
}
```

#### [nanobanana_edit.py](nanobanana_edit.py)
Core image editing wrapper for Gemini 2.5 Flash API.

**Purpose:** Single-image editing using Google's Gemini 2.5 Flash Image model with optional reference/highlight images for guided editing.

**Usage:**
```bash
python nanobanana_edit.py <original_image> <prompt.json> [output.png]
```

**Examples:**
```bash
# Basic editing (no reference image)
python nanobanana_edit.py original.jpg prompt.json

# With segmentation highlight mask (RECOMMENDED)
python nanobanana_edit.py original.jpg prompt.json edited.png
```

**Prompt JSON Format (basic):**
```json
{
  "prompt": "Edit description here"
}
```

**Prompt JSON Format (with reference/highlight image):**
```json
{
  "prompt": "Based on the highlighted areas in the reference image: Replace with modern sofa",
  "reference_image": "results/room_annotated.jpg"
}
```

**Features:**
- **Multi-image input support**: Sends reference image first, then original
- **Gemini 2.5 Flash Image model**: Fast and efficient
- **Image-only response mode**: Returns only edited image, no text
- **Segmentation integration**: Works with highlight masks from `segment_image.py`

**Image Order:**
Based on testing, the optimal order for multi-image input is:
1. Reference/highlight image (visual guide showing what to edit)
2. Original clean image (what actually gets edited)
3. Text prompt (instructions for the edit)

**Workflow with Segmentation:**
```bash
# 1. Segment the object to create highlight mask
python interior-segment-labeler/segment_image.py room.jpg

# 2. Creates: results/room_annotated.jpg (highlight) and results/room_original.jpg
# 3. Edit using both images
python nanobanana_edit.py results/room_original.jpg prompt.json
# (where prompt.json references results/room_annotated.jpg)
```

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
- Language detection (defaults to English)

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
- Configurable voice settings (stability: 0.5, similarity: 0.8, speed: 0.85)
- High-quality MP3 output (44.1kHz, 96kbps)

## Complete Workflow Example

### End-to-End Image Transformation

```bash
# 1. Start with an interior design image
image="room.jpg"

# 2. Create analysis JSON with recommendations
cat > prompts.json << EOF
{
  "analysis_summary": {"total_issues": 2},
  "issues": [
    {
      "item": "Sofa",
      "issue": "Color doesn't match guidelines",
      "guideline_reference": "Section 2.3",
      "recommendation": "Replace with a modern sage green sofa"
    },
    {
      "item": "Wall paint",
      "issue": "Too bright",
      "guideline_reference": "Section 1.1",
      "recommendation": "Paint walls a soft beige color"
    }
  ]
}
EOF

# 3. Run transformation (includes automatic segmentation)
python transform_image.py room.jpg prompts.json final_room.jpg --keep-intermediates

# 4. Optionally add bounding boxes for frontend visualization
python identify_changes.py prompts.json room.jpg --output prompts_with_boxes.json
```

### Using the Microservices

```bash
# 1. Start all services (in separate terminals)
python image_server.py      # Terminal 1
python detection_server.py  # Terminal 2
python verbose_server.py    # Terminal 3

# 2. Use the services via API (see API Examples below)
```

## API Examples

### Image Transformation

```python
import requests
import json

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

if result["success"]:
    print(f"Transformed image: {result['transformed_image_path']}")

    # Download the image
    filename = result['transformed_image_path'].split('/')[-1]
    download_url = f"http://127.0.0.1:8002/download/{filename}"
    img_response = requests.get(download_url)

    with open("transformed.jpg", "wb") as f:
        f.write(img_response.content)
else:
    print(f"Error: {result['error']}")
```

### Object Detection

```python
import requests
import json

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

if result["success"]:
    analysis = result["analysis_with_boxes"]
    for issue in analysis["issues"]:
        coords = issue.get("bounding_box_coordinates", {})
        print(f"Item: {issue['item']}")
        print(f"Detections: {coords['count']}")
        for det in coords.get("detections", []):
            print(f"  - {det['label']}: {det['bbox']}")
else:
    print(f"Error: {result['error']}")
```

### WebSocket Voice (JavaScript)

```javascript
const ws = new WebSocket('ws://127.0.0.1:8003/ws');

// Convert audio blob to base64
function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Speech-to-Text
async function transcribeAudio(audioBlob) {
  const arrayBuffer = await audioBlob.arrayBuffer();
  const base64Audio = arrayBufferToBase64(arrayBuffer);

  ws.send(JSON.stringify({
    type: 'stt',
    audio: base64Audio
  }));
}

// Text-to-Speech
function speakText(text) {
  ws.send(JSON.stringify({
    type: 'tts',
    text: text
  }));
}

// Handle responses
const audioChunks = [];
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'transcript') {
    console.log('Transcript:', data.text);
  } else if (data.type === 'audio_chunk') {
    audioChunks.push(data.data);
  } else if (data.type === 'audio_end') {
    // Combine and play audio chunks
    playAudio(audioChunks);
    audioChunks.length = 0;
  } else if (data.type === 'error') {
    console.error('Error:', data.message);
  }
};
```

## Directory Structure

```
picture-generation-verbose-api-module/
├── image_server.py                    # Image transformation API server
├── detection_server.py                # Object detection API server
├── verbose_server.py                  # WebSocket voice services server
├── transform_image.py                 # Batch segmentation + editing script
├── identify_changes.py                # Batch object detection script
├── nanobanana_edit.py                 # Single image editing wrapper
├── speech_and_text/
│   ├── speech_to_text.py             # Audio transcription utility
│   └── text_to_speech.py             # Text-to-speech generation utility
├── interior-segment-labeler/          # Florence-2 detection & segmentation submodule
│   ├── segment_image.py              # Object segmentation script
│   ├── florence2_detector.py         # Florence-2 detector implementation
│   └── ...                           # Additional segmentation utilities
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (create this)
├── .gitignore                        # Git ignore rules
├── .gitmodules                       # Git submodule configuration
└── README.md                         # This file

Generated directories (auto-created):
├── intermediate_files/               # Temporary files during transformation
├── audio_output/                     # TTS generated audio files
├── text_output/                      # STT transcription files
└── results/                          # Segmentation output files
```

## Architecture & Design

### Segmentation-Based Editing Pipeline

The master branch uses a sophisticated segmentation-first approach:

1. **Object Detection**: Florence-2 model identifies target objects in the image
2. **Highlight Generation**: Creates annotated images with highlighted regions
3. **Multi-Image Context**: Sends both highlight mask and original to Gemini
4. **Precise Editing**: AI focuses on highlighted areas for accurate transformations

**Benefits:**
- More precise edits (AI knows exactly what to change)
- Better preservation of non-target areas
- Visual guidance reduces AI hallucination
- Works well with complex scenes

**Trade-offs:**
- Additional processing time for segmentation
- Requires accurate object detection
- More complex pipeline

### Model Selection: Gemini 2.5 Flash

The master branch uses Gemini 2.5 Flash Image model:

**Advantages:**
- Faster processing than Gemini 3 Pro
- Lower API costs
- Good quality for most use cases
- Better availability

**When to Use:**
- Production deployments requiring speed
- Budget-conscious projects
- Standard quality requirements

**When to Consider Alternatives:**
- Need highest quality edits → Use Gemini 3 Pro (nanobanana_pro_updated branch)
- Simple edits without segmentation → Use base_nanobanana_bypass_segmentation branch

## Troubleshooting

### Common Issues

**Segmentation Failures:**
- Ensure `interior-segment-labeler` submodule is initialized: `git submodule update --init --recursive`
- Check item names in prompts match objects in image
- Try more specific item names (e.g., "leather sofa" instead of "furniture")

**API Key Errors:**
- Ensure `.env` file exists and contains valid API keys
- Check that `NANOBANANA_API_KEY` and `ELEVENLABS_API_KEY` are set
- Verify API keys have sufficient credits/quota

**Import Errors:**
- Activate virtual environment: `myenv\Scripts\activate` (Windows) or `source myenv/bin/activate` (macOS/Linux)
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Florence-2 Model Not Loading:**
- Ensure `interior-segment-labeler` submodule is initialized
- Check available system memory (model requires ~2GB)
- Try running detection script directly to isolate issue

**Long Transformation Times:**
- Each edit includes segmentation + API call (expect 30-60s per issue)
- Use `--keep-intermediates` flag to monitor progress
- Check intermediate files to see where pipeline stalls

**Subprocess Timeouts:**
- Default timeouts: 120s for segmentation, 180s for editing
- For complex images, you may need to increase timeouts in code
- Check stderr output for specific error messages

**WebSocket Connection Issues:**
- Verify service is running: `curl http://127.0.0.1:8003/health`
- Check firewall settings allow local connections
- Ensure port 8003 is not in use by another process

**Intermediate Files Not Cleaned Up:**
- Use `--keep-intermediates` flag to debug
- Manual cleanup: Delete `intermediate_files/` directory
- Check disk space if transformations fail

## Dependencies

Key packages (see [requirements.txt](requirements.txt) for full list):

**Core Framework:**
- `fastapi==0.115.6` - REST API framework
- `uvicorn==0.32.1` - ASGI server
- `pydantic==2.10.5` - Data validation

**AI/ML:**
- `torch==2.6.0` - PyTorch for Florence-2
- `torchvision==0.21.0` - Vision utilities
- `transformers==4.57.3` - Hugging Face models
- `google-genai==1.55.0` - Gemini API client
- `elevenlabs==2.26.1` - Voice services API

**Image Processing:**
- `opencv-python==4.12.0.88` - Image processing
- `pillow==12.0.0` - Image handling
- `segment-anything-hq==0.3` - Segmentation models
- `timm==1.0.22` - Model architectures

**Utilities:**
- `python-dotenv==1.2.1` - Environment variables
- `requests==2.32.5` - HTTP client
- `numpy==2.2.6` - Numerical operations

## Branch Comparison

This repository has multiple branches for different use cases:

| Branch | Segmentation | AI Model | Best For |
|--------|-------------|----------|----------|
| **master** (current) | ✅ Yes | Gemini 2.5 Flash | Production, balanced speed/quality |
| nanobanana_pro_updated | ✅ Yes | Gemini 3 Pro | Highest quality edits |
| base_nanobanana_bypass_segmentation | ❌ No | Gemini 3 Pro | Simple edits, faster processing |

**Switch branches:**
```bash
# View branches
git branch -a

# Switch to another branch
git checkout nanobanana_pro_updated
```

## Performance Tips

1. **Batch Processing**: Process multiple images sequentially to reuse loaded models
2. **Intermediate Files**: Use `--keep-intermediates` only for debugging
3. **Image Size**: Resize large images before processing to reduce API costs
4. **Prompt Quality**: More specific prompts = better segmentation = better edits
5. **Service Deployment**: Run services on machines with GPU for faster Florence-2 detection

## License

This module is part of the HKS Spatial project.

## Contributing

For development:
1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Ensure all services start and health checks pass
5. Submit a pull request

## Support

For issues or questions:
- Check service health endpoints (`/health`)
- Review log output from running services
- Inspect intermediate files with `--keep-intermediates`
- API documentation available at `http://localhost:{port}/docs` when services are running
- Check GitHub issues for known problems

## Additional Resources

- **Interior Segment Labeler**: See `interior-segment-labeler/` for segmentation details
- **API Documentation**: Start services and visit `/docs` endpoints
- **Example Images**: Test with sample interior design images
- **Branch Documentation**: Each branch may have specific setup requirements
