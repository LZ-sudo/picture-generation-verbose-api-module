#!/usr/bin/env python3
"""
Transform Image - Sequential Image Editing Orchestrator

Processes multiple editing recommendations sequentially by:
1. Segmenting each item from the image
2. Sending to Nano Banana for editing
3. Using the output as input for the next edit

Usage:
    python transform_image.py <original_image> <editing_prompts.json> [output_path]
"""

import sys
import json
import subprocess
import shutil
import os
from pathlib import Path
from datetime import datetime


class ImageTransformer:
    """Orchestrates sequential image editing with segmentation and Nano Banana."""
    
    def __init__(self, original_image_path, prompts_json_path, output_path=None, keep_intermediates=False):
        """
        Initialize the transformer.

        Args:
            original_image_path: Path to the original image
            prompts_json_path: Path to image_editing_prompts.json
            output_path: Optional custom output path for final image
            keep_intermediates: If True, don't delete intermediate files (useful for debugging)
        """
        self.keep_intermediates = keep_intermediates
        self.original_image = Path(original_image_path)
        self.prompts_json = Path(prompts_json_path)
        
        # Validate inputs
        if not self.original_image.exists():
            raise FileNotFoundError(f"Original image not found: {original_image_path}")
        if not self.prompts_json.exists():
            raise FileNotFoundError(f"Prompts JSON not found: {prompts_json_path}")
        
        # Load editing prompts
        with open(self.prompts_json, 'r') as f:
            self.editing_data = json.load(f)
        
        self.issues = self.editing_data.get('issues', [])
        if not self.issues:
            raise ValueError("No issues found in prompts JSON")

        # Setup directories - create relative to this script's location, not CWD
        script_dir = Path(__file__).parent
        self.intermediate_dir = script_dir / 'intermediate_files'
        self.intermediate_dir.mkdir(exist_ok=True)
        
        # Setup output path
        if output_path:
            self.final_output = Path(output_path)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.final_output = Path(f"{self.original_image.stem}_transformed_{timestamp}.jpg")
        
    
        print(f"Image Transformation Pipeline")
        print(f"Original image: {self.original_image}")
        print(f"Prompts file: {self.prompts_json}")
        print(f"Issues to process: {len(self.issues)}")
        print(f"Final output: {self.final_output}")
    
    def transform(self):
        """Execute the sequential transformation pipeline."""
        
        # Track current image state (starts with original)
        current_image = self.original_image
        
        try:
            # Process each issue sequentially
            for i, issue in enumerate(self.issues, start=1):
                print(f"Processing Issue {i}/{len(self.issues)}: {issue['item']}")
                
                # Step 1: Segment the item
                print(f"\n[Step 1/3] Segmenting {issue['item']}...")
                segmentation_result = self._segment_item(current_image, issue['item'], i)
                
                if not segmentation_result:
                    print(f"[WARN] Segmentation failed for {issue['item']}, skipping...")
                    continue
                
                # Step 2: Edit with Nano Banana
                print(f"\n[Step 2/3] Applying edit with Nano Banana...")
                edited_image = self._edit_with_nanobanana(
                    segmentation_result['original_image'],
                    segmentation_result['highlight_image'],
                    issue['recommendation'],
                    i
                )
                
                if not edited_image:
                    print(f"[WARN] Nano Banana edit failed for {issue['item']}, skipping...")
                    continue
                
                # Step 3: Update current image for next iteration
                print(f"\n[Step 3/3] Updating current image state...")
                current_image = edited_image
                print(f"[OK] Issue {i} completed successfully")
            
            # Copy final result to output location
            print(f"Finalizing...")
            shutil.copy(current_image, self.final_output)
            print(f"[OK] Final edited image saved: {self.final_output}")

            # Cleanup intermediate files
            self._cleanup()

            print(f"[OK] Transformation complete!")
            print(f"Processed: {len(self.issues)} issues")
            print(f"Output: {self.final_output}")
            print()
            
            return self.final_output
            
        except KeyboardInterrupt:
            print("Process interrupted by user")
            self._cleanup()
            sys.exit(1)
        except Exception as e:
            print(f"Error during transformation: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup()
            sys.exit(1)
    
    def _segment_item(self, image_path, item_name, iteration):
        """
        Segment a specific item from the image by calling segment_image.py.

        Args:
            image_path: Path to current image
            item_name: Name of item to segment (e.g., "Coffee Table")
            iteration: Current iteration number

        Returns:
            Dict with paths to generated files, or None if failed
        """
        # Copy current image to intermediate directory
        intermediate_image = self.intermediate_dir / f"current_{iteration:02d}.jpg"
        shutil.copy(image_path, intermediate_image)

        # Create segmentation prompt JSON in same directory as image
        seg_prompt_path = self.intermediate_dir / f"{intermediate_image.stem}_prompt.json"
        seg_prompt_data = {"prompts": [item_name]}

        with open(seg_prompt_path, 'w') as f:
            json.dump(seg_prompt_data, f, indent=2)

        print(f"  Created segmentation prompt for '{item_name}'")

        # Setup output directory
        seg_output_dir = self.intermediate_dir / f"seg_output_{iteration:02d}"
        seg_output_dir.mkdir(exist_ok=True)

        print(f"  Running segmentation...")
        try:
            # Create environment with UTF-8 encoding to prevent Unicode errors
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Build absolute path to segment_image.py relative to this script
            script_dir = Path(__file__).parent
            segment_script = script_dir / 'interior-segment-labeler' / 'segment_image.py'

            if not segment_script.exists():
                print(f"  [ERROR] Segment script not found: {segment_script}")
                return None

            # Call segment_image.py script with correct working directory
            # Use absolute paths for all arguments
            # Use DEVNULL to prevent deadlock from output buffer filling
            result = subprocess.run(
                [
                    sys.executable,
                    str(segment_script),  # Use absolute path
                    str(intermediate_image.resolve()),  # Absolute path to input
                    '--output',
                    str(seg_output_dir.resolve())  # Absolute path to output dir
                ],
                stdout=subprocess.DEVNULL,  # Discard stdout to prevent buffer deadlock
                stderr=subprocess.PIPE,  # Capture only stderr for errors
                text=True,
                env=env,  # Use modified environment
                cwd=str(script_dir),  # Run in picture-generation directory for relative imports
                timeout=120
            )

            if result.returncode != 0:
                print(f"  [ERROR] Segmentation failed (return code: {result.returncode})")
                print(f"  STDERR: {result.stderr}")
                return None

            # Find generated files
            image_stem = intermediate_image.stem
            annotated_path = seg_output_dir / f"{image_stem}_annotated.jpg"

            if not annotated_path.exists():
                print(f"  [ERROR] Annotated image not found: {annotated_path}")
                return None

            print(f"  [OK] Segmentation complete")
            print(f"    - Annotated: {annotated_path}")
            print(f"    - Using original: {intermediate_image}")

            return {
                'original_image': intermediate_image,  # Use the copy we made
                'highlight_image': annotated_path,      # Use annotated as reference
                'output_dir': seg_output_dir
            }

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Segmentation timed out")
            return None
        except Exception as e:
            print(f"  [ERROR] Segmentation error: {e}")
            return None
    
    def _edit_with_nanobanana(self, original_image, highlight_image, recommendation, iteration):
        """
        Edit image using Nano Banana by calling nanobanana_edit.py.

        Args:
            original_image: Path to current clean image (to be edited)
            highlight_image: Path to highlight/mask image (shows what to edit)
            recommendation: Text recommendation for edit
            iteration: Current iteration number

        Returns:
            Path to edited image, or None if failed
        """
        # Create prompt JSON
        nb_prompt_path = self.intermediate_dir / f"nb_prompt_{iteration:02d}.json"
        enhanced_prompt = f"Based on the highlighted areas in the reference image: {recommendation}"

        # Use absolute paths for the reference image
        nb_prompt_data = {
            "prompt": enhanced_prompt,
            "reference_image": str(highlight_image.resolve())  # Convert to absolute path
        }

        with open(nb_prompt_path, 'w') as f:
            json.dump(nb_prompt_data, f, indent=2)

        print(f"  Calling Nano Banana API...")
        print(f"  Prompt: {enhanced_prompt}")

        # Output path
        edited_path = self.intermediate_dir / f"edited_{iteration:02d}.jpg"

        try:
            # Create environment with UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Build absolute path to nanobanana_edit.py relative to this script
            script_dir = Path(__file__).parent
            nanobanana_script = script_dir / 'nanobanana_edit.py'

            if not nanobanana_script.exists():
                print(f"  [ERROR] Nanobanana script not found: {nanobanana_script}")
                return None

            # Call nanobanana_edit.py script with correct working directory
            # Use absolute paths for all arguments
            # Use DEVNULL to prevent deadlock from output buffer filling
            result = subprocess.run(
                [
                    sys.executable,
                    str(nanobanana_script),  # Use absolute path
                    str(original_image.resolve()),  # Absolute path to input image
                    str(nb_prompt_path.resolve()),  # Absolute path to prompt JSON
                    str(edited_path.resolve())  # Absolute path to output image
                ],
                stdout=subprocess.DEVNULL,  # Discard stdout to prevent buffer deadlock
                stderr=subprocess.PIPE,  # Capture only stderr for errors
                text=True,
                env=env,
                cwd=str(script_dir),  # Run in picture-generation directory for relative imports
                timeout=180
            )

            if result.returncode != 0:
                print(f"  [ERROR] Nano Banana failed (return code: {result.returncode})")
                print(f"  STDERR: {result.stderr}")
                return None

            if not edited_path.exists():
                print(f"  [ERROR] Edited image not found")
                return None

            print(f"  [OK] Edit complete: {edited_path}")
            return edited_path

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Nano Banana timed out")
            return None
        except Exception as e:
            print(f"  [ERROR] Nano Banana error: {e}")
            return None
    
    def _cleanup(self):
        """Clean up intermediate files."""
        if self.keep_intermediates:
            print(f"Keeping intermediate files for debugging: {self.intermediate_dir}/")
            return

        print(f"\nCleaning up intermediate files...")
        try:
            if self.intermediate_dir.exists():
                shutil.rmtree(self.intermediate_dir)
                print(f"Deleted: {self.intermediate_dir}/")
        except Exception as e:
            print(f"Warning: Could not clean up intermediate files: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python transform_image.py <image> <editing_prompts.json> [output] [--keep-intermediates]")
        print("\nExamples:")
        print("  python transform_image.py room.jpg image_editing_prompts.json")
        print("  python transform_image.py room.jpg prompts.json final_room.jpg")
        print("  python transform_image.py room.jpg prompts.json final.jpg --keep-intermediates")
        print("\nInput Format (editing_prompts.json):")
        print('''{
  "analysis_summary": {"total_issues": 3},
  "issues": [
    {
      "item": "Coffee Table",
      "issue": "Description of the issue...",
      "guideline_reference": "Reference...",
      "recommendation": "Paint the legs a dark charcoal grey..."
    },
    ...
  ]
}''')
        sys.exit(1)

    # Parse arguments
    original_image = sys.argv[1]
    prompts_json = sys.argv[2]

    # Check for optional arguments
    output_path = None
    keep_intermediates = False

    for i in range(3, len(sys.argv)):
        if sys.argv[i] == '--keep-intermediates':
            keep_intermediates = True
        else:
            output_path = sys.argv[i]

    # Create and run transformer
    transformer = ImageTransformer(original_image, prompts_json, output_path, keep_intermediates)
    transformer.transform()


if __name__ == '__main__':
    main()