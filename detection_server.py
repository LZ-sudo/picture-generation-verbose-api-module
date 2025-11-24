#!/usr/bin/env python3
"""
Object Detection Microservice API
==================================
FastAPI server for Florence-2 object detection and spatial coordinate identification.
Processes analysis JSON files and adds bounding box coordinates to recommendations.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import time
import shutil

from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add interior-segment-labeler to path for imports
script_dir = Path(__file__).parent
segment_labeler_dir = script_dir / "interior-segment-labeler"
sys.path.insert(0, str(segment_labeler_dir))

# Import Florence-2 detector
try:
    from label_changes import Florence2Detector
    FLORENCE2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Florence2Detector: {e}")
    FLORENCE2_AVAILABLE = False

# Load shared environment variables
from dotenv import load_dotenv
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection Service",
    description="Florence-2 based object detection for spatial coordinate identification",
    version="1.0.0"
)

# Global detector instance (lazy loaded)
_detector = None


def get_detector():
    """Get or initialize the Florence-2 detector (singleton)"""
    global _detector
    if _detector is None:
        if not FLORENCE2_AVAILABLE:
            raise RuntimeError("Florence-2 detector not available")
        print("Initializing Florence-2 detector...")
        _detector = Florence2Detector()
        print("[OK] Florence-2 detector initialized")
    return _detector


class IdentifyResponse(BaseModel):
    """Response model for identification"""
    success: bool
    analysis_with_boxes: Optional[Dict[str, Any]] = None
    detected_count: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    florence2_loaded: bool


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - service info"""
    return {
        "status": "running",
        "service": "Object Detection",
        "version": "1.0.0",
        "florence2_loaded": _detector is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Try to initialize detector to verify it works
        detector = get_detector()
        return {
            "status": "healthy",
            "service": "Object Detection",
            "version": "1.0.0",
            "florence2_loaded": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_objects(
    file: UploadFile = File(...),
    analysis_json: str = Body(...)
):
    """
    Identify objects in image based on analysis recommendations.
    Adds bounding box coordinates to each issue in the analysis JSON.

    Args:
        file: Original image file
        analysis_json: JSON string containing issues with recommendations

    Returns:
        Updated analysis JSON with bounding_box_coordinates added to each issue
    """
    start_time = time.time()
    temp_image = None
    intermediate_dir = None

    try:
        # Parse and validate JSON
        analysis_data = json.loads(analysis_json)

        if "issues" not in analysis_data:
            raise ValueError("JSON must contain 'issues' array")

        issues = analysis_data.get('issues', [])
        if len(issues) == 0:
            return {
                "success": True,
                "analysis_with_boxes": analysis_data,
                "detected_count": 0,
                "processing_time": time.time() - start_time
            }

        # Setup temp directory
        import tempfile
        system_temp = Path(tempfile.gettempdir())
        temp_dir = system_temp / "hks_spatial_detection"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Create intermediate directory for processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        intermediate_dir = temp_dir / f"intermediate_{timestamp}"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded image
        temp_image = intermediate_dir / f"_temp_{file.filename}"
        with open(temp_image, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"\n{'='*60}")
        print(f"Detection request received:")
        print(f"  Image: {file.filename}")
        print(f"  Issues: {len(issues)}")
        print(f"  Temp dir: {intermediate_dir}")
        print(f"{'='*60}\n")

        # Get detector
        detector = get_detector()

        # Process each recommendation
        total_detections = 0

        for idx, issue in enumerate(issues):
            recommendation = issue.get('recommendation', '')
            item = issue.get('item', 'Unknown')

            if not recommendation:
                # No recommendation, skip
                analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                    'format': 'normalized',
                    'detections': [],
                    'count': 0
                }
                continue

            print(f"[{idx + 1}/{len(issues)}] Processing: {item}")
            print(f"  Recommendation: {recommendation[:60]}{'...' if len(recommendation) > 60 else ''}")

            # Create prompt file for this recommendation
            prompt_file = intermediate_dir / f"{temp_image.stem}_prompt.json"
            with open(prompt_file, 'w') as f:
                json.dump({"prompts": [recommendation]}, f, indent=2)

            try:
                # Run detection
                result = detector.detect(temp_image, [recommendation])

                if result and result.get('count', 0) > 0:
                    # Extract bounding box coordinates
                    bboxes = []
                    for det in result.get('detections', []):
                        bbox_info = {
                            'label': det['label'],
                            'bbox': det['bbox_normalized'],
                            'center': det['center_normalized'],
                            'confidence': det.get('confidence', 1.0)
                        }
                        bboxes.append(bbox_info)

                    # Add to analysis data
                    analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                        'format': 'normalized',
                        'detections': bboxes,
                        'count': len(bboxes)
                    }

                    total_detections += len(bboxes)
                    print(f"  [OK] Found {len(bboxes)} detection(s)")
                else:
                    # No detections
                    analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                        'format': 'normalized',
                        'detections': [],
                        'count': 0
                    }
                    print(f"  [SKIP] No objects detected")

            except Exception as e:
                print(f"  [ERROR] Detection failed: {e}")
                # Add empty coordinates on error
                analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                    'format': 'normalized',
                    'detections': [],
                    'count': 0,
                    'error': str(e)
                }

            # Clean up prompt file
            prompt_file.unlink(missing_ok=True)

        processing_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"[OK] Detection complete!")
        print(f"  Total detections: {total_detections}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "analysis_with_boxes": analysis_data,
            "detected_count": total_detections,
            "processing_time": processing_time
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON: {str(e)}"
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in identify endpoint: {error_detail}")
        return {
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }
    finally:
        # Clean up temp files
        if intermediate_dir and intermediate_dir.exists():
            try:
                shutil.rmtree(intermediate_dir)
                print(f"[OK] Cleaned up: {intermediate_dir}")
            except Exception as e:
                print(f"[!] Warning: Could not clean up {intermediate_dir}: {e}")


if __name__ == "__main__":
    # Get port from environment or default
    port = int(os.getenv("DETECTION_SERVICE_PORT", 8004))
    host = os.getenv("DETECTION_SERVICE_HOST", "127.0.0.1")

    print(f"\n{'='*60}")
    print(f"Starting Object Detection Service")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Florence-2: {'Available' if FLORENCE2_AVAILABLE else 'Not Available'}")
    print(f"{'='*60}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
