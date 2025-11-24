#!/usr/bin/env python3
"""
Identify Changes Script
Processes LLM analysis JSON to identify and locate objects in images using Florence-2 detection.
Adds bounding box coordinates to each recommendation in the analysis.

Usage:
    python identify_changes.py analysis.json image.jpg
    python identify_changes.py analysis.json image.jpg --output results.json
    python identify_changes.py analysis.json image.jpg --output-dir results/
"""

import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path


def extract_recommendations(analysis_data):
    """
    Extract recommendations from analysis JSON.

    Args:
        analysis_data: Parsed analysis JSON data

    Returns:
        List of tuples (issue_index, item, recommendation)
    """
    recommendations = []

    issues = analysis_data.get('issues', [])
    for idx, issue in enumerate(issues):
        item = issue.get('item', 'Unknown')
        recommendation = issue.get('recommendation', '')

        if recommendation:
            recommendations.append((idx, item, recommendation))

    return recommendations


def run_label_changes(image_path, recommendation_text, temp_image_path, output_dir):
    """
    Run label_changes.py with a single recommendation.

    Args:
        image_path: Original image path
        recommendation_text: Recommendation text to detect
        temp_image_path: Temporary copy of image (for prompt file)
        output_dir: Directory for output files

    Returns:
        Dictionary with detection results, or None if detection failed
    """
    temp_image_path = Path(temp_image_path).resolve()
    output_dir = Path(output_dir).resolve()

    # Create prompt file next to the temporary image
    # label_changes.py expects {image_stem}_prompt.json next to the image
    temp_prompt = temp_image_path.parent / f"{temp_image_path.stem}_prompt.json"
    prompt_data = {
        "prompts": [recommendation_text]
    }

    with open(temp_prompt, 'w') as f:
        json.dump(prompt_data, f, indent=2)

    print(f"  > Created prompt file: {temp_prompt.name}")
    print(f"    Prompt content: {recommendation_text}")

    # Prepare command to run label_changes.py
    label_changes_script = Path(__file__).parent / "interior-segment-labeler" / "label_changes.py"

    if not label_changes_script.exists():
        print(f"  [ERROR] label_changes.py not found at {label_changes_script}")
        return None

    cmd = [
        sys.executable,
        str(label_changes_script),
        str(temp_image_path),  # Already resolved to absolute path
        "--output-dir", str(output_dir),  # Already resolved to absolute path
        "--no-image",  # Skip image generation
        "--format", "normalized"  # Use normalized coordinates for frontend
    ]

    try:
        # Run label_changes.py
        print(f"  > Running detection...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
            cwd=str(label_changes_script.parent)
        )

        # Debug output
        print(f"    Return code: {result.returncode}")
        if result.stdout:
            print(f"    STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"    STDERR:\n{result.stderr}")

        # Check for errors
        if result.returncode != 0:
            if result.stderr:
                print(f"  [ERROR] {result.stderr.strip()}")
            if "No objects detected" in result.stdout or "No valid objects" in result.stdout:
                print(f"  [SKIP] No objects detected for this recommendation")
            return None

        # Read the generated detections JSON
        detections_file = output_dir / f"{temp_image_path.stem}_detections.json"

        print(f"    Looking for detections file: {detections_file}")
        print(f"    File exists: {detections_file.exists()}")

        if detections_file.exists():
            with open(detections_file, 'r') as f:
                detections_data = json.load(f)
            print(f"    Loaded detections: {detections_data.get('count', 0)} objects")
            return detections_data
        else:
            print(f"  [!] Warning: Detection file not generated")
            # List files in output directory for debugging
            print(f"    Files in output dir:")
            for f in output_dir.iterdir():
                print(f"      - {f.name}")
            return None

    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_analysis(analysis_path, image_path, output_path=None, output_dir=None):
    """
    Process analysis JSON and add bounding box coordinates to each issue.

    Args:
        analysis_path: Path to analysis JSON file
        image_path: Path to input image
        output_path: Path to save updated JSON (optional)
        output_dir: Directory for output files

    Returns:
        Updated analysis data with bounding box coordinates
    """
    image_path = Path(image_path)
    analysis_path = Path(analysis_path)

    # Load analysis JSON
    print(f"\n{'='*60}")
    print(f"Processing Analysis")
    print(f"{'='*60}")
    print(f"Analysis: {analysis_path.name}")
    print(f"Image: {image_path.name}")
    print(f"{'='*60}\n")

    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)

    # Extract recommendations
    recommendations = extract_recommendations(analysis_data)
    total = len(recommendations)

    if total == 0:
        print("[!] No recommendations found in analysis")
        return analysis_data

    print(f"Found {total} recommendations to process\n")

    # Create intermediate_files_identification directory
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = image_path.parent

    intermediate_dir = base_dir / "intermediate_files_identification"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Copy image to intermediate directory for processing
    temp_image_name = f"_temp_{image_path.name}"
    temp_image_path = intermediate_dir / temp_image_name
    shutil.copy2(image_path, temp_image_path)

    print(f"Created temporary workspace: {intermediate_dir}\n")

    try:
        # Process each recommendation sequentially
        for idx, item, recommendation in recommendations:
            print(f"[{idx + 1}/{total}] Processing: {item}")
            print(f"  Recommendation: {recommendation[:60]}{'...' if len(recommendation) > 60 else ''}")

            # Run detection
            detections = run_label_changes(
                image_path,
                recommendation,
                temp_image_path,
                intermediate_dir
            )

            if detections and detections.get('count', 0) > 0:
                # Extract only the bounding box coordinates
                bboxes = []
                for det in detections.get('detections', []):
                    bbox_info = {
                        'label': det['label'],
                        'bbox': det['bbox'],  # Normalized coordinates
                        'center': det['center'],
                        'confidence': det.get('confidence', 1.0)
                    }
                    bboxes.append(bbox_info)

                # Add to analysis data
                analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                    'format': 'normalized',
                    'detections': bboxes,
                    'count': len(bboxes)
                }

                print(f"  [OK] Found {len(bboxes)} detection(s)")
            else:
                # No detections found
                analysis_data['issues'][idx]['bounding_box_coordinates'] = {
                    'format': 'normalized',
                    'detections': [],
                    'count': 0
                }
                print(f"  [SKIP] No objects detected")

            print()

    finally:
        # Always clean up intermediate directory
        print(f"\nCleaning up intermediate files...")
        try:
            shutil.rmtree(intermediate_dir)
            print(f"[OK] Removed: {intermediate_dir}")
        except Exception as e:
            print(f"[!] Warning: Could not remove intermediate directory: {e}")

    # Save updated analysis
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"\n{'='*60}")
        print(f"[OK] Updated analysis saved to: {output_path}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"[OK] Processing complete!")
        print(f"{'='*60}\n")

    return analysis_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process LLM analysis to identify objects and add bounding box coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - updates analysis JSON with bounding boxes
  python identify_changes.py analysis.json room.jpg

  # Save to custom output file
  python identify_changes.py analysis.json room.jpg --output results.json

  # Use custom output directory
  python identify_changes.py analysis.json room.jpg --output-dir results/

  # Print to console only
  python identify_changes.py analysis.json room.jpg --no-save

Output:
  - Updated analysis JSON with "bounding_box_coordinates" added to each issue
  - Coordinates are in normalized format (0-1 range) for frontend use

Process:
  1. Reads analysis JSON and extracts all recommendations
  2. Processes each recommendation sequentially (one at a time)
  3. Runs Florence-2 detection via label_changes.py
  4. Adds bounding box coordinates to each issue
  5. Returns updated analysis JSON
        """
    )

    parser.add_argument('analysis', help='Path to analysis JSON file')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: {analysis_stem}_with_boxes.json)')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--no-save', action='store_true', help='Print to console only (do not save file)')

    args = parser.parse_args()

    # Validate paths
    analysis_path = Path(args.analysis)
    if not analysis_path.exists():
        print(f"Error: Analysis file not found: {analysis_path}")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Determine output path
    if args.no_save:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        # Default: save as {analysis_stem}_with_boxes.json
        if args.output_dir:
            output_path = Path(args.output_dir) / f"{analysis_path.stem}_with_boxes.json"
        else:
            output_path = analysis_path.parent / f"{analysis_path.stem}_with_boxes.json"

    # Process analysis
    updated_data = process_analysis(
        analysis_path,
        image_path,
        output_path,
        args.output_dir
    )

    # Print to console if not saving
    if args.no_save:
        print(json.dumps(updated_data, indent=2))


if __name__ == '__main__':
    main()
