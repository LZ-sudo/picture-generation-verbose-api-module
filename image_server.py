"""
Picture Generation Microservice API
====================================
FastAPI server for image transformation only.
Uses asyncio subprocess to avoid threading deadlocks with nested subprocesses.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import asyncio

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
    title="Picture Generation Service",
    description="Image transformation based on JSON recommendations",
    version="1.0.0"
)


class TransformResponse(BaseModel):
    """Response model for transformation"""
    success: bool
    transformed_image_path: Optional[str] = None
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
        "service": "Picture Generation",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Verify API key is set
    api_key = os.getenv("NANOBANANA_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="NANOBANANA_API_KEY not configured"
        )

    return {
        "status": "healthy",
        "service": "Picture Generation",
        "version": "1.0.0"
    }


@app.post("/transform", response_model=TransformResponse)
async def transform_image(
    file: UploadFile = File(...),
    analysis_json: str = Body(...)
):
    """
    Transform an image based on JSON analysis recommendations.
    Uses asyncio subprocess to run transform_image.py as isolated process.

    Args:
        file: Original image file
        analysis_json: JSON string containing issues with recommendations

    Returns:
        Path to transformed image
    """
    temp_original = None
    temp_json_file = None
    output_file = None

    try:
        # Parse and validate JSON
        analysis_data = json.loads(analysis_json)

        if "issues" not in analysis_data:
            raise ValueError("JSON must contain 'issues' array")

        # Setup temp directory
        import tempfile
        system_temp = Path(tempfile.gettempdir())
        temp_dir = system_temp / "hks_spatial_transform"
        temp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

        # Save uploaded image
        temp_original = temp_dir / f"original_{timestamp}_{file.filename}"
        with open(temp_original, "wb") as f:
            content = await file.read()
            f.write(content)

        # Save JSON to file
        temp_json_file = temp_dir / f"prompts_{timestamp}.json"
        with open(temp_json_file, "w") as f:
            json.dump(analysis_data, f, indent=2)

        # Output path
        output_file = temp_dir / f"transformed_{timestamp}.jpg"

        print(f"Transform request received:")
        print(f"  Input: {temp_original}")
        print(f"  Prompts: {temp_json_file}")
        print(f"  Output: {output_file}")
        print(f"  Issues: {len(analysis_data.get('issues', []))}")

        # Get Python executable from venv
        script_dir = Path(__file__).parent
        venv_dir = script_dir / "myenv"

        # Windows or Unix
        python_exe = venv_dir / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = venv_dir / "bin" / "python"

        if not python_exe.exists():
            raise RuntimeError(f"Virtual environment not found at {venv_dir}")

        transform_script = script_dir / "transform_image.py"

        # Run transform_image.py as isolated subprocess using asyncio
        print(f"Starting transformation subprocess...")
        process = await asyncio.create_subprocess_exec(
            str(python_exe),
            str(transform_script),
            str(temp_original.resolve()),
            str(temp_json_file.resolve()),
            str(output_file.resolve()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(script_dir)  # Run in picture-generation directory
        )

        # Wait for completion and capture output
        stdout, stderr = await process.communicate()

        # Decode output
        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        # Log output for debugging
        if stdout_text:
            print("Transform stdout:")
            print(stdout_text[-1000:])  # Last 1000 chars to avoid spam

        if process.returncode != 0:
            print(f"Transform failed with return code {process.returncode}")
            if stderr_text:
                print(f"Transform stderr:\n{stderr_text}")
            raise RuntimeError(
                f"Transformation failed (exit code {process.returncode})\n"
                f"Error: {stderr_text[:500]}"
            )

        if not output_file.exists():
            raise RuntimeError(f"Transformation did not create output file: {output_file}")

        print(f"[OK] Transformation complete: {output_file}")

        return {
            "success": True,
            "transformed_image_path": str(output_file)
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON: {str(e)}"
        }
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in transform endpoint: {error_detail}")
        return {
            "success": False,
            "error": error_detail
        }

    finally:
        # Clean up temp input files (keep output for download)
        if temp_original and temp_original.exists():
            try:
                temp_original.unlink()
            except:
                pass
        if temp_json_file and temp_json_file.exists():
            try:
                temp_json_file.unlink()
            except:
                pass


@app.get("/download/{filename}")
async def download_transformed_image(filename: str):
    """
    Download a transformed image file.

    Args:
        filename: Name of the transformed image file

    Returns:
        File response with the image
    """
    import tempfile
    system_temp = Path(tempfile.gettempdir())
    temp_dir = system_temp / "hks_spatial_transform"
    file_path = temp_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=filename
    )


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("IMAGE_GEN_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("IMAGE_GEN_SERVICE_PORT", "8002"))

    print(f"Starting Picture Generation Service on {host}:{port}")
    print("Using asyncio subprocess to avoid threading deadlocks")
    print("Long-running transformations supported (30+ minutes)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=3600  # 1 hour keep-alive for long transformations
    )
