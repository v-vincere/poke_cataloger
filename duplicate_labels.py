# This script takes a master set of bounding boxes and applies them
# to all images in a directory, creating YOLO .txt label files for a single class.

import os

def generate_generic_labels():
    # --- Configuration ---
    base_path = '/mnt/d/ml_poke/'
    image_dir = os.path.join(base_path, 'data/pics_from_discord')
    output_dir = os.path.join(base_path, 'data/labels')
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Image dimensions from the master screenshot, needed for normalization.
    image_width = 244
    image_height = 227

    # The single class index for "card"
    class_index = 0

    # The 5 bounding boxes extracted from your JSON file (in pixel coordinates)
    # The JSON data had negative width/height, so we use abs() to make them positive.
    # The coordinates are [x_min, y_min, width, height]
    master_boxes = [
        [0.58, 6.37, 74.91, 105.12],   # Top-left
        [82.24, 4.69, 76.92, 106.89],  # Top-middle
        [165.22, 4.11, 74.33, 107.47], # Top-right
        [38.97, 116.74, 77.50, 108.98],# Bottom-left
        [121.31, 116.74, 80.27, 110.25] # Bottom-right
    ]

    print('Starting generic label generation...')

    # --- Main Loop ---
    # Get all image files from the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        label_filename = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        yolo_annotations = []
        for box in master_boxes:
            # Convert pixel coordinates [x_min, y_min, width, height] to YOLO format
            x_center = box[0] + box[2] / 2
            y_center = box[1] + box[3] / 2
            
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = box[2] / image_width
            height_norm = box[3] / image_height
            
            yolo_annotations.append(f'{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}')
            
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
            
    print(f'Successfully created {len(image_files)} generic label files in {output_dir}')

if __name__ == '__main__':
    generate_generic_labels()