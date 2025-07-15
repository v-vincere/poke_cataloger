# create_hash_database.py (with Standardized Resizing and Debug Image Saving)
# MODIFICATION: This script now saves a copy of each downscaled reference
# image into a 'downscaled_images' folder. This allows for visual
# verification of the exact images used to generate the hashes.

import os
import json
from PIL import Image
import imagehash

def create_hash_database():
    # --- Configuration ---
    base_path = '/mnt/d/ml_poke/data/'
    image_dir = os.path.join(base_path, 'card_images')
    output_file = os.path.join(base_path, 'card_hashes.json')
    
    # --- NEW: Configuration for saving downscaled images for verification ---
    downscaled_dir = os.path.join(base_path, 'downscaled_images')
    
    # Define the standard size for hashing.
    HASH_IMAGE_SIZE = (72, 108)
    
    # This will hold all hashes for all cards
    hash_database = {}
    
    # --- NEW: Create the output directory for downscaled images if it doesn't exist ---
    os.makedirs(downscaled_dir, exist_ok=True)
    
    print(f'Scanning images in: {image_dir}')
    print(f'All reference images will be resized to {HASH_IMAGE_SIZE}.')
    print(f"Downscaled verification images will be saved to: {downscaled_dir}")
    # create_hash_database.py (Original Full-Size Version)
# MODIFICATION: This script has been reverted to its original logic. It calculates
# hashes on the full-sized, original reference images. This is required for the
# bot's new upscaling strategy to work correctly.

import os
import json
from PIL import Image
import imagehash

def create_hash_database():
    # --- Configuration ---
    base_path = '/mnt/d/ml_poke/data/'
    image_dir = os.path.join(base_path, 'card_images')
    output_file = os.path.join(base_path, 'card_hashes.json')
    
    hash_database = {}
    print(f'Scanning images in {image_dir} to build a full-resolution hash database...')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        try:
            image_path = os.path.join(image_dir, filename)
            card_id = os.path.splitext(filename)[0]
            
            with Image.open(image_path) as img:
                # Hashes are calculated on the original, full-size image
                perceptual_hash = imagehash.phash(img)
                difference_hash = imagehash.dhash(img)
                wavelet_hash = imagehash.whash(img)
                
                hash_database[card_id] = {
                    'phash': str(perceptual_hash),
                    'dhash': str(difference_hash),
                    'whash': str(wavelet_hash)
                }
                
        except Exception as e:
            print(f'Could not process {filename}: {e}')
            
    with open(output_file, 'w') as f:
        json.dump(hash_database, f, indent=4)
        
    print(f'\nSuccessfully created full-resolution hash database with {len(hash_database)} entries.')
    print(f'Database saved to {output_file}')

if __name__ == '__main__':
    create_hash_database()
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Counter for user feedback
    processed_count = 0
    
    for filename in image_files:
        try:
            image_path = os.path.join(image_dir, filename)
            card_id = os.path.splitext(filename)[0]
            
            with Image.open(image_path) as img:
                # --- STEP 1: RESIZE THE IMAGE ---
                # Resize the image to the standard size using a high-quality filter.
                resized_img = img.resize(HASH_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # --- STEP 2 (NEW): SAVE THE RESIZED IMAGE FOR VERIFICATION ---
                # The resized image is converted to RGB to ensure compatibility with JPEG saving.
                save_path = os.path.join(downscaled_dir, filename)
                resized_img.convert('RGB').save(save_path)
                
                # --- STEP 3: CALCULATE HASHES ---
                # Calculate all hash types on the RESIZED image
                perceptual_hash = imagehash.phash(resized_img)
                difference_hash = imagehash.dhash(resized_img)
                wavelet_hash = imagehash.whash(resized_img)
                
                # Store the hashes in the database
                hash_database[card_id] = {
                    'phash': str(perceptual_hash),
                    'dhash': str(difference_hash),
                    'whash': str(wavelet_hash)
                }
                
                processed_count += 1
                
        except Exception as e:
            print(f'Could not process {filename}: {e}')
            
    # Save the comprehensive hash database to a JSON file
    with open(output_file, 'w') as f:
        json.dump(hash_database, f, indent=4)
        
    print(f'\nSuccessfully processed {processed_count} images.')
    print(f'Created comprehensive hash database with {len(hash_database)} entries.')
    print(f'Database saved to: {output_file}')

if __name__ == '__main__':
    create_hash_database()