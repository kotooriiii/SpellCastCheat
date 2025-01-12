import os

# Define the common image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}


def rename_images_in_subdirs(root_dir):
    """Rename images in root_dir and its subdirectories to have 'image_' prefix with subdir-specific counters."""
    for root, _, files in os.walk(root_dir):
        # Initialize the counter for the current subdirectory
        subdir_counter = 1

        for file in files:
            # Check if the file is an image
            if any(file.lower().endswith(ext) for ext in image_extensions):
                current_path = os.path.join(root, file)

                # Check if the file already has the 'image_' prefix
                if not file.startswith("image_"):
                    while True:  # Loop until a unique filename is found
                        new_filename = f"image_{subdir_counter}.png"
                        new_path = os.path.join(root, new_filename)

                        if not os.path.exists(new_path):  # Check if the file already exists
                            break
                        subdir_counter += 1  # Increment the counter if the file exists

                    # Rename the file
                    try:
                        os.rename(current_path, new_path)
                        print(f"Renamed: {current_path} -> {new_path}")
                        subdir_counter += 1  # Increment for the next new file
                    except Exception as e:
                        print(f"Error renaming {current_path}: {e}")


# Specify the root directory to start renaming
root_directory = "../../dataset"

# Call the function
rename_images_in_subdirs(root_directory)
