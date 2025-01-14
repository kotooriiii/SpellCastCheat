import cv2

from ai import extract_grid_nodes

file_path = 'C:\\Users\\Kotori\\Desktop\\del\\Screenshot_1.png'  # Change this

if __name__ == "__main__":
    extract_grid_nodes.extract_grid_nodes(cv2.imread(file_path), should_write_to_io=True)
