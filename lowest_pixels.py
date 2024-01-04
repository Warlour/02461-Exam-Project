from PIL import Image
import os

def estimate_quality(file_path):
    try:
        # Get file size in bytes
        file_size = os.path.getsize(file_path)

        # Open the image to get its resolution
        with Image.open(file_path) as img:
            width, height = img.size

        # Calculate an estimate of image quality based on file size and resolution
        quality_estimate = file_size / (width * height)
        return quality_estimate, (width, height)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return float('inf'), None

def get_lowest_quality_and_dimensions(folder_path):
    lowest_quality = float('inf')
    lowest_dimensions = None

    # Recursively traverse through all subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                file_path = os.path.join(root, filename)

                # Estimate image quality and get dimensions
                quality_estimate, dimensions = estimate_quality(file_path)

                # Update lowest quality
                lowest_quality = min(lowest_quality, quality_estimate)

                # Update lowest dimensions
                if dimensions is not None:
                    if lowest_dimensions is None or dimensions < lowest_dimensions:
                        lowest_dimensions = dimensions

    return lowest_quality, lowest_dimensions

# Specify the path to your main folder
main_folder_path = 'data/FER2013'

# Get the lowest quality estimate and dimensions
lowest_quality, lowest_dimensions = get_lowest_quality_and_dimensions(main_folder_path)

# Print the result
if lowest_quality != float('inf'):
    print(f"The estimated lowest image quality in the folder is: {lowest_quality}")
    print(f"The lowest dimensions among all images: {lowest_dimensions}")
else:
    print("No images found or error occurred during processing.")
