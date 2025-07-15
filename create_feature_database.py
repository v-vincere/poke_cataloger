# create_master_database.py
# MODIFICATION: This is a unified script that generates ALL required reference data.
# 1. Calculates pHash, dHash, and wHash and saves them to card_hashes.json.
# 2. Calculates SIFT features and saves them to data/features_sift/.
# 3. Calculates AKAZE features and saves them to data/features_akaze/.

import os
import cv2
import numpy as np
import json
from PIL import Image
import imagehash

def create_master_database():
    # --- Configuration ---
    BASE_PATH = '/mnt/d/ml_poke/data/'
    IMAGE_DIR = os.path.join(BASE_PATH, 'card_images')
    
    # Hash configuration
    HASH_OUTPUT_FILE = os.path.join(BASE_PATH, 'card_hashes.json')
    
    # Feature matching configuration
    SIFT_FEATURES_DIR = os.path.join(BASE_PATH, 'features_sift')
    AKAZE_FEATURES_DIR = os.path``.join(BASE_PATH, 'features_akaze')
    FEATURE_IMAGE_SIZE = (300, 418) # Standard size for feature detection pre-processing

    # --- Initialization ---
    os.makedirs(SIFT_FEATURES_DIR, exist_ok=True)
    os.makedirs(AKAZE_FEATURES_DIR, exist_ok=True)
    
    try:
        sift = cv2.SIFT_create()
        akaze = cv2.AKAZE_create()
    except cv2.error as e:
        print("\n--- FATAL ERROR ---")
        print("Could not initialize SIFT or AKAZE. Please ensure you have 'opencv-contrib-python' installed.")
        print("Run: pip uninstall opencv-python && pip install opencv-contrib-python")
        print(f"Original OpenCV error: {e}\n")
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hash_database = {}
    
    print("--- Starting Master Database Creation ---")
    print(f"Scanning images in: {IMAGE_DIR}")
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    processed_count = 0

    for filename in image_files:
        try:
            card_id = os.path.splitext(filename)[0]
            image_path = os.path.join(IMAGE_DIR, filename)
            
            # --- Part 1: Perceptual Hashing ---
            with Image.open(image_path) as img:
                hash_database[card_id] = {
                    'phash': str(imagehash.phash(img)),
                    'dhash': str(imagehash.dhash(img)),
                    'whash': str(imagehash.whash(img)),
                }

            # --- Part 2: Feature Descriptors (SIFT & AKAZE) ---
            img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                print(f"Warning: Could not read {filename} with OpenCV. Skipping feature generation.")
                continue

            # Pre-processing pipeline for feature detection
            img_resized = cv2.resize(img_cv, FEATURE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            img_enhanced = clahe.apply(img_resized)

            # SIFT
            kp_sift, des_sift = sift.detectAndCompute(img_enhanced, None)
            if des_sift is not None:
                np.save(os.path.join(SIFT_FEATURES_DIR, f"{card_id}.npy"), des_sift)

            # AKAZE
            kp_akaze, des_akaze = akaze.detectAndCompute(img_enhanced, None)
            if des_akaze is not None:
                np.save(os.path.join(AKAZE_FEATURES_DIR, f"{card_id}.npy"), des_akaze)

            processed_count += 1
            print(f"Processed {card_id}...")

        except Exception as e:
            print(f'ERROR: Could not process {filename}: {e}')
    
    # Save the hash database to JSON
    with open(HASH_OUTPUT_FILE, 'w') as f:
        json.dump(hash_database, f, indent=4)

    print("\n--- Master Database Creation Complete! ---")
    print(f"Successfully processed {processed_count} cards.")
    print(f"Hash database saved to: {HASH_OUTPUT_FILE}")
    print(f"SIFT features saved in: {SIFT_FEATURES_DIR}")
    print(f"AKAZE features saved in: {AKAZE_FEATURES_DIR}")

if __name__ == '__main__':
    create_master_database()