import os
from rembg import remove

def process_images_in_folder(input_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-files
        if not os.path.isfile(input_path):
            continue

        # Process common image formats
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        print(f"Processing: {filename}...")

        try:
            # Read input image
            with open(input_path, 'rb') as i:
                input_image = i.read()

            # Generate mask
            output_image = remove(input_image, only_mask=True)

            # Create output filename
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_mask.png")

            # Save mask
            with open(output_path, 'wb') as o:
                o.write(output_image)

            print(f"Saved mask to: {output_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    input_folder = "images"
    output_folder = "masks"

    process_images_in_folder(input_folder, output_folder)
