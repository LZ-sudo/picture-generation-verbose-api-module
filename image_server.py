"""
Picture Generation Microservice API
====================================
FastAPI server for image transformation only.
Uses asyncio subprocess to avoid threading deadlocks with nested subprocesses.
Supports async job pattern (primary) with sync fallback for compatibility.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import asyncio
import uuid
from enum import Enum

from fastapi import FastAPI, HTTPException, File, UploadFile, Body, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
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
    description="Image transformation based on JSON recommendations (Async + Sync modes)",
    version="2.0.0"
)

# Job storage (in-memory for now, could be Redis in production)
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

jobs: Dict[str, Dict[str, Any]] = {}


class TransformResponse(BaseModel):
    """Response model for transformation"""
    success: bool
    transformed_image_path: Optional[str] = None
    error: Optional[str] = None


class JobSubmitResponse(BaseModel):
    """Response when submitting async job"""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[TransformResponse] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    async_mode: bool = True


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - service info"""
    return {
        "status": "running",
        "service": "Picture Generation",
        "version": "2.0.0",
        "async_mode": True
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
        "version": "2.0.0",
        "async_mode": True
    }


async def run_transformation(
    image_content: bytes,
    filename: str,
    analysis_data: dict,
    job_id: Optional[str] = None
) -> TransformResponse:
    """
    Core transformation logic (used by both sync and async endpoints).

    Args:
        image_content: Image file bytes
        filename: Original filename
        analysis_data: Parsed analysis JSON
        job_id: Optional job ID for tracking

    Returns:
        TransformResponse with result or error
    """
    temp_original = None
    temp_json_file = None
    output_file = None

    try:
        # Validate JSON structure
        if "issues" not in analysis_data:
            raise ValueError("JSON must contain 'issues' array")

        # Setup temp directory
        import tempfile
        system_temp = Path(tempfile.gettempdir())
        temp_dir = system_temp / "hks_spatial_transform"
        temp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        job_prefix = f"{job_id}_" if job_id else ""

        # Save uploaded image
        temp_original = temp_dir / f"original_{job_prefix}{timestamp}_{filename}"
        with open(temp_original, "wb") as f:
            f.write(image_content)

        # Save JSON to file
        temp_json_file = temp_dir / f"prompts_{job_prefix}{timestamp}.json"
        with open(temp_json_file, "w") as f:
            json.dump(analysis_data, f, indent=2)

        # Output path
        output_file = temp_dir / f"transformed_{job_prefix}{timestamp}.jpg"

        print(f"Transform request {'[Job: ' + job_id + ']' if job_id else ''} received:")
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

        # Update job status if async
        if job_id and job_id in jobs:
            jobs[job_id]["status"] = JobStatus.PROCESSING
            jobs[job_id]["progress"] = "Running transformation subprocess..."

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

        return TransformResponse(
            success=True,
            transformed_image_path=str(output_file)
        )

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in transformation: {error_detail}")
        return TransformResponse(
            success=False,
            error=error_detail
        )

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


@app.post("/transform", response_model=TransformResponse)
async def transform_image_sync(
    file: UploadFile = File(...),
    analysis_json: str = Body(...)
):
    """
    LEGACY SYNC ENDPOINT: Transform an image (blocking, for localhost use).
    For tunnel/production use, prefer /transform/async endpoint.

    Args:
        file: Original image file
        analysis_json: JSON string containing issues with recommendations

    Returns:
        Path to transformed image (waits for completion)
    """
    try:
        # Parse and validate JSON
        analysis_data = json.loads(analysis_json)

        # Read file content
        content = await file.read()

        # Run transformation using shared logic
        result = await run_transformation(
            image_content=content,
            filename=file.filename or "image.jpg",
            analysis_data=analysis_data,
            job_id=None
        )

        return result

    except json.JSONDecodeError as e:
        return TransformResponse(
            success=False,
            error=f"Invalid JSON: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in transform endpoint: {error_detail}")
        return TransformResponse(
            success=False,
            error=error_detail
        )


@app.post("/transform/async", response_model=JobSubmitResponse)
async def transform_image_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_json: str = Body(...)
):
    """
    ASYNC ENDPOINT: Submit transformation job (returns immediately with job_id).
    Use /transform/status/{job_id} to poll for completion.
    Recommended for tunnel/production use to avoid gateway timeouts.

    Args:
        background_tasks: FastAPI background tasks
        file: Original image file
        analysis_json: JSON string containing issues with recommendations

    Returns:
        Job ID for polling status
    """
    try:
        # Parse and validate JSON
        analysis_data = json.loads(analysis_json)

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Read file content
        content = await file.read()

        # Create job record
        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": "Job queued",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }

        # Schedule background task
        async def process_job():
            try:
                result = await run_transformation(
                    image_content=content,
                    filename=file.filename or "image.jpg",
                    analysis_data=analysis_data,
                    job_id=job_id
                )

                # Update job with result
                jobs[job_id]["status"] = JobStatus.COMPLETED if result.success else JobStatus.FAILED
                jobs[job_id]["result"] = result
                jobs[job_id]["error"] = result.error
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                jobs[job_id]["progress"] = "Transformation complete" if result.success else "Transformation failed"

            except Exception as e:
                import traceback
                error_detail = f"{str(e)}\n{traceback.format_exc()}"
                jobs[job_id]["status"] = JobStatus.FAILED
                jobs[job_id]["error"] = error_detail
                jobs[job_id]["completed_at"] = datetime.now().isoformat()
                jobs[job_id]["progress"] = "Job failed with exception"

        # Add to background tasks
        background_tasks.add_task(process_job)

        print(f"[ASYNC] Job {job_id} submitted for transformation")

        return JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Transformation job submitted. Poll /transform/status/{job_id} for progress."
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"ERROR in async transform endpoint: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/transform/status/{job_id}", response_model=JobStatusResponse)
async def get_transformation_status(job_id: str):
    """
    Check status of async transformation job.

    Args:
        job_id: Job ID returned from /transform/async

    Returns:
        Current job status and result (if completed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        completed_at=job.get("completed_at")
    )


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
