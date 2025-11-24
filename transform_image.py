#!/usr/bin/env python3
"""
Transform Image - Sequential Image Editing Orchestrator

Processes multiple editing recommendations sequentially by:
1. Enhancing the prompt with the item name (object of attention)
2. Sending to Nano Banana Pro (Gemini 3 Pro) for editing
3. Using the output as input for the next edit

Note: This version uses Gemini 3 Pro Image Preview which can perform
      intelligent editing without requiring segmentation masks.

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
    """Orchestrates sequential image editing with Nano Banana Pro (Gemini 3 Pro)."""
    
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

                # Step 1: Edit with Nano Banana Pro (no segmentation needed)
                print(f"\n[Step 1/2] Applying edit with Nano Banana Pro...")
                edited_image = self._edit_with_nanobanana(
                    current_image,
                    issue['item'],
                    issue['recommendation'],
                    i
                )

                if not edited_image:
                    print(f"[WARN] Nano Banana Pro edit failed for {issue['item']}, skipping...")
                    continue

                # Step 2: Update current image for next iteration
                print(f"\n[Step 2/2] Updating current image state...")
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
    
    
    def _edit_with_nanobanana(self, current_image, item_name, recommendation, iteration):
        """
        Edit image using Nano Banana Pro (Gemini 3 Pro) by calling nanobanana_edit.py.

        Note: This version does not require segmentation. The item name is added
              directly to the prompt to guide the AI on what to focus on.

        Args:
            current_image: Path to current image (to be edited)
            item_name: Name of the item to focus on (e.g., "Wall paint", "Floor rug")
            recommendation: Text recommendation for edit
            iteration: Current iteration number

        Returns:
            Path to edited image, or None if failed
        """
        # Copy current image to intermediate directory for tracking
        intermediate_image = self.intermediate_dir / f"current_{iteration:02d}.jpg"
        shutil.copy(current_image, intermediate_image)

        # Create enhanced prompt with item name as object of attention
        # Following the new format: "The object of attention is {item name}. {recommendation}."
        enhanced_prompt = f"The object of attention is {item_name}. {recommendation}. Keep the perspective of the original image."

        # Create prompt JSON (no reference image needed)
        nb_prompt_path = self.intermediate_dir / f"nb_prompt_{iteration:02d}.json"
        nb_prompt_data = {
            "prompt": enhanced_prompt
        }

        with open(nb_prompt_path, 'w') as f:
            json.dump(nb_prompt_data, f, indent=2)

        print(f"  Calling Nano Banana Pro API...")
        print(f"  Item: {item_name}")
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
                    str(intermediate_image.resolve()),  # Absolute path to input image
                    str(nb_prompt_path.resolve()),  # Absolute path to prompt JSON
                    str(edited_path.resolve())  # Absolute path to output image
                ],
                stdout=subprocess.DEVNULL,  # Discard stdout to prevent buffer deadlock
                stderr=subprocess.PIPE,  # Capture only stderr for errors
                text=True,
                env=env,
                cwd=str(script_dir),  # Run in picture-generation directory for relative imports
                timeout=240  # Increased timeout for Gemini 3 Pro
            )

            if result.returncode != 0:
                print(f"  [ERROR] Nano Banana Pro failed (return code: {result.returncode})")
                print(f"  STDERR: {result.stderr}")
                return None

            if not edited_path.exists():
                print(f"  [ERROR] Edited image not found")
                return None

            print(f"  [OK] Edit complete: {edited_path}")
            return edited_path

        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Nano Banana Pro timed out")
            return None
        except Exception as e:
            print(f"  [ERROR] Nano Banana Pro error: {e}")
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
        print("\nNote: This version uses Gemini 3 Pro Image Preview (Nano Banana Pro)")
        print("      and does NOT require segmentation. The item name is included")
        print("      in the prompt to guide the AI on what to focus on.")
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