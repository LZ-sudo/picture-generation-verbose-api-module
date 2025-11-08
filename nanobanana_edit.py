import os
import json
import sys
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

def edit_image_with_prompt(image_path, prompt, reference_image_path=None):
    """
    Edit an image using the Gemini API based on a text prompt.
    Supports multi-image input for better context.

    IMPORTANT: Based on testing, for best results with highlight masks:
    - Send highlight image FIRST (visual guide showing where to edit)
    - Send original clean image SECOND (what actually gets edited)
    - Gemini seems to use the highlight as a mask and edits the clean image

    Args:
        image_path: Path to the ORIGINAL clean image to be edited
        prompt: Text description of the edit to apply
        reference_image_path: Optional path to HIGHLIGHT/MASK image showing what areas to edit

    Returns:
        PIL.Image: The edited image
    """

    # Validate image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate reference image if provided
    if reference_image_path and not os.path.exists(reference_image_path):
        raise FileNotFoundError(f"Reference image file not found: {reference_image_path}")

    # Initialize the Gemini client
    # API key should be set in NANOBANANA_API_KEY environment variable
    api_key = os.environ.get('NANOBANANA_API_KEY')
    if not api_key:
        raise ValueError("NANOBANANA_API_KEY environment variable not set. Please set it in your .env file.")

    client = genai.Client(api_key=api_key)

    # Load the image(s)
    try:
        input_image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Build contents list for API call
    # IMPORTANT: Based on testing, we need to put the reference (highlight) FIRST
    # and the original image SECOND for best results
    contents = []

    # Add reference image FIRST if provided (this becomes the visual guide)
    if reference_image_path:
        try:
            reference_image = Image.open(reference_image_path)
            contents.append(reference_image)
        except Exception as e:
            raise ValueError(f"Failed to load reference image: {e}")

    # Add the main input image SECOND (or first if no reference)
    contents.append(input_image)

    # Add prompt at the end
    contents.append(prompt)

    # Call the API to edit the image
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
        )
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")

    # Extract the edited image from the response
    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]

    if not image_parts:
        raise RuntimeError("No image data returned from API")

    # Convert bytes to PIL Image
    edited_image = Image.open(BytesIO(image_parts[0]))

    return edited_image


def main():
    """
    Main function to handle CLI usage.
    Pass the ORIGINAL clean image as the main argument.
    Optionally specify a HIGHLIGHT/MASK image in the JSON to show what to edit.
    """
    if len(sys.argv) < 3:
        print("Usage: python nanobanana_edit.py <original_image> <prompt.json> [output.png]")
        print("\nExamples:")
        print("  # Basic editing (no mask):")
        print("  python nanobanana_edit.py original.jpg prompt.json")
        print("\n  # With highlight mask (RECOMMENDED for precise edits):")
        print("  python nanobanana_edit.py original.jpg prompt.json")
        print("  (where prompt.json specifies highlight image as reference)")
        print("\nPrompt JSON format (basic):")
        print('{\n  "prompt": "Edit description here"\n}')
        print("\nPrompt JSON format (with highlight mask - RECOMMENDED):")
        print('{\n  "prompt": "Replace the highlighted yellow areas with a modern sofa",')
        print('  "reference_image": "results/room_highlight.jpg"\n}')
        print("\nWorkflow:")
        print("  1. Run segmentation: python segment_image.py room.jpg")
        print("  2. Creates: room_original.jpg and room_highlight.jpg")
        print("  3. Edit: python nanobanana_edit.py results/room_original.jpg prompt.json")
        sys.exit(1)

    image_path = sys.argv[1]
    json_file = sys.argv[2]

    # Generate output filename based on input filename
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    else:
        # Extract filename and extension from input image path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        extension = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_edited{extension}"

    # Load JSON prompt and optional reference image
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            prompt = data.get('prompt')
            if not prompt:
                print("Error: JSON file must contain 'prompt' field")
                sys.exit(1)
            reference_image_path = data.get('reference_image')
    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        sys.exit(1)

    # Edit the image
    try:
        print(f"Loading image from: {image_path}")
        if reference_image_path:
            print(f"Loading reference image from: {reference_image_path}")
        print(f"Applying edit prompt: {prompt}")
        print("Calling Gemini API...")

        edited_image = edit_image_with_prompt(image_path, prompt, reference_image_path)

        # Save the edited image
        edited_image.save(output_path)
        print(f"Successfully saved edited image to: {output_path}")

        return edited_image

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
