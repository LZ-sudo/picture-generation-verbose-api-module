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
from pathlib import Path

# Import Florence2Detector from the shared module
sys.path.insert(0, str(Path(__file__).parent / "interior-segment-labeler"))
from florence2_detector import Florence2Detector, filter_detections


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


def run_detection(detector, image_path, recommendation_text):
    """
    Run Florence-2 detection directly using the detector instance.

    Args:
        detector: Florence2Detector instance (reused across calls)
        image_path: Path to input image
        recommendation_text: Recommendation text to detect

    Returns:
        Dictionary with detection results, or None if detection failed
    """
    try:
        print(f"  > Running detection...")
        print(f"    Prompt content: {recommendation_text}")

        # Parse the recommendation text into vocabulary
        # The recommendation is the text we want to detect
        vocabulary = [recommendation_text]

        # Run detection directly
        result = detector.detect(image_path, vocabulary)

        # Filter detections to remove action words and duplicates
        if result['count'] > 0:
            filtered_detections = filter_detections(result['detections'], iou_threshold=0.7)
            result['detections'] = filtered_detections
            result['count'] = len(filtered_detections)

        if result['count'] == 0:
            print(f"  [SKIP] No objects detected for this recommendation")
            return None

        print(f"  [OK] Found {result['count']} detection(s)")
        return result

    except Exception as e:
        print(f"  [ERROR] Unexpected error during detection: {e}")
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

    # Initialize Florence-2 detector ONCE for all detections
    # This is the key optimization - model stays loaded in memory
    print("Initializing Florence-2 detector (one-time setup)...")
    detector = Florence2Detector()
    print(f"[OK] Detector ready. Will process {total} recommendations.\n")

    try:
        # Process each recommendation sequentially
        for idx, item, recommendation in recommendations:
            print(f"[{idx + 1}/{total}] Processing: {item}")
            print(f"  Recommendation: {recommendation[:60]}{'...' if len(recommendation) > 60 else ''}")

            # Run detection using the shared detector instance
            detections = run_detection(
                detector,
                image_path,
                recommendation
            )

            if detections and detections.get('count', 0) > 0:
                # Extract only the bounding box coordinates
                bboxes = []
                for det in detections.get('detections', []):
                    bbox_info = {
                        'label': det['label'],
                        'bbox': det['bbox_normalized'],  # Normalized coordinates
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
        # Clean up detector to free GPU/CPU memory
        print(f"\nCleaning up detector...")
        del detector
        print(f"[OK] Detector cleaned up")

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
