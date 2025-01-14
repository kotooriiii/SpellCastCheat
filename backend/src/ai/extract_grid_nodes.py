import cv2
import numpy as np


def extract_grid_nodes(image):
    # Convert the Pillow Image to a NumPy array
    image_np = np.array(image)

    # If the image is in RGBA or RGB mode, convert it to BGR for OpenCV
    if image.mode in ("RGBA", "RGB"):
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image.mode == "L":  # Grayscale
        pass  # No conversion needed for grayscale

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make the grid more distinguishable
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find edges in the image to detect the grid
    edges = cv2.Canny(thresh, 50, 150, apertureSize=7)

    # Find contours to detect the grid lines
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours that are unlikely to be grid lines
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Get bounding boxes of the grid contours
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Calculate the center of each bounding box (center_x, center_y)
    centers = [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes]

    # Define tolerance value for misalignment in the y-axis (row grouping)
    y_tolerance = 20  # Adjust this value based on how much misalignment is acceptable

    # Sort bounding boxes based on center coordinates: first by center_y, then by center_x
    sorted_bounding_boxes = sorted(zip(bounding_boxes, centers), key=lambda item: (item[1][1], item[1][0]))

    # Unzip back into sorted bounding boxes and centers
    sorted_bounding_boxes, _ = zip(*sorted_bounding_boxes)

    # Group bounding boxes by proximity in the y-axis (vertical rows)
    grouped_rows = []
    current_row = []
    last_y = None

    for box, center in zip(sorted_bounding_boxes, centers):
        _, y, w, h = box
        center_y = center[1]

        if last_y is None or abs(center_y - last_y) <= y_tolerance:
            # If it's close enough in the y-axis, it's considered part of the same row
            current_row.append(box)
        else:
            # If the y distance is too large, start a new row
            grouped_rows.append(current_row)
            current_row = [box]

        last_y = center_y

    # Add the last row to the grouped rows
    if current_row:
        grouped_rows.append(current_row)

    # Sort each row based on x (left to right)
    for row in grouped_rows:
        row.sort(key=lambda box: box[0])  # Sort by the x-coordinate (left to right)

    # Now you can extract each node, including padding
    cell_padding = 5  # Adjust this value for more or less padding
    nodes = []
    for row in grouped_rows:
        for x, y, w, h in row:
            # Apply padding
            x1 = max(x - cell_padding, 0)
            y1 = max(y - cell_padding, 0)
            x2 = min(x + w + cell_padding, image_np.shape[1])
            y2 = min(y + h + cell_padding, image_np.shape[0])

            # Crop the node
            node = image_np[y1:y2, x1:x2]

            nodes.append(node)
            # Save the node image (optional)
            # cv2.imwrite(f'node_{len(nodes)}.png', node)

    return nodes


def preprocess_nodes(nodes):
    """Preprocess the extracted nodes for model input."""
    processed = []
    for i, node in enumerate(nodes):
        # Resize to 128x128 and normalize (if required)

        # Ensure each node has consistent
        node_resized = cv2.resize(node, (128, 128), interpolation=cv2.INTER_CUBIC)  # Resize to 128x128

        # Normalize the node: Scale pixel values to range [0, 1]
        # node_normalized = node_resized.astype('float32') / 255.0

        processed.append(node_resized)

        # Save the node image (optional)
        # cv2.imwrite(f'node_{i}.png', node_normalized)

    return np.array(processed)  # Return as NumPy array
