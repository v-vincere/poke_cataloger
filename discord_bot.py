# discord_bot.py (Fully configured by config.ini)

import discord
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import imagehash
from PIL import Image
import asyncio
import torch
import configparser
import logging
from setup_logging import setup_logs
import csv
from datetime import datetime
import xml.etree.ElementTree as ET
import time
import threading
import concurrent.futures
import sys

# --- G-SHEETS: GOOGLE API IMPORTS ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# --- CONFIGURATION & LOGGING SETUP ---
config = configparser.ConfigParser()
config.read('data/config.ini')
setup_logs(config.get('Logging', 'log_directory', fallback='logs'))

class LoggerStream:
    def __init__(self, logger, level):
        self.logger, self.level = logger, level
    def write(self, buf):
        for line in buf.rstrip().splitlines(): self.logger.log(self.level, line.rstrip())
    def flush(self): pass

sys.stdout = LoggerStream(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = LoggerStream(logging.getLogger('STDERR'), logging.ERROR)

# --- CONFIGURATION VARIABLES ---
try:
    BOT_TOKEN = config.get('Discord', 'bot_token')
    TARGET_CHANNEL_ID = config.getint('Discord', 'target_channel_id')
    BASE_PATH = config.get('Paths', 'base_path')
    MODEL_PATH = os.path.join(BASE_PATH, config.get('Paths', 'model_path'))
    CSV_PATH = os.path.join(BASE_PATH, config.get('Paths', 'csv_path'))
    HASH_DB_PATH = os.path.join(BASE_PATH, config.get('Paths', 'hash_db_path'))
    SIFT_FEATURES_DIR = os.path.join(BASE_PATH, config.get('Paths', 'sift_features_dir'))
    AKAZE_FEATURES_DIR = os.path.join(BASE_PATH, config.get('Paths', 'akaze_features_dir'))
    TEMP_DIR = os.path.join(BASE_PATH, 'data/temp')
    CSV_LOG_ENABLED = config.getboolean('CSVLogging', 'enable_csv_log')
    CSV_LOG_PATH = config.get('CSVLogging', 'csv_log_path')
    CSV_HEADERS = ['timestamp', 'deviceAccount', 'card_name', 'rarity', 'uploaded_to_sheets']
    ALGO_ENABLED = {
        'phash': config.getboolean('Algorithms', 'phash_enabled'), 'dhash': config.getboolean('Algorithms', 'dhash_enabled'),
        'whash': config.getboolean('Algorithms', 'whash_enabled'), 'sift': config.getboolean('Algorithms', 'sift_enabled'),
        'akaze': config.getboolean('Algorithms', 'akaze_enabled'),
    }
    ALGO_PRIORITY = [algo.strip() for algo in config.get('Algorithms', 'priority').split(',')]
    FLANN_ENABLED = config.getboolean('Algorithms', 'flann_enabled', fallback=False)
    PARSE_HISTORY = config.getboolean('Historical', 'parse_history', fallback=False)
    LAST_MESSAGE_ID = config.get('Historical', 'last_message_id', fallback=None)
    GSHEETS_ENABLED = config.getboolean('GoogleSheets', 'enabled', fallback=False)
    GSHEETS_CRED_PATH = os.path.join(BASE_PATH, config.get('GoogleSheets', 'credentials_path'))
    GSHEETS_SPREADSHEET_NAME = config.get('GoogleSheets', 'spreadsheet_name')
    GSHEETS_SYNC_INTERVAL = config.getint('GoogleSheets', 'sync_interval_seconds', fallback=300)

except (configparser.NoSectionError, configparser.NoOptionError) as e:
    logging.critical("FATAL: Configuration error in config.ini: %s", e); exit()

# --- BOT, MODEL & MATCHER SETUP ---
intents = discord.Intents.default(); intents.message_content = True
client = discord.Client(intents=intents)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; logging.info("--- Using device: %s ---", DEVICE)
YOLO_MODEL = YOLO(MODEL_PATH); YOLO_MODEL.to(DEVICE)
CARD_INFO_DF = pd.read_csv(CSV_PATH)
CARD_INFO_DF['card_id_str'] = CARD_INFO_DF['image_filename'].str.replace('.jpg', '', regex=False)
CARD_LOOKUP = CARD_INFO_DF.set_index('card_id_str')[['card_name', 'rarity', 'image_filename']].to_dict(orient='index')
with open(HASH_DB_PATH, 'r') as f: HASH_DATABASE = json.load(f)
SIFT = cv2.SIFT_create() if ALGO_ENABLED.get('sift') else None
AKAZE = cv2.AKAZE_create() if ALGO_ENABLED.get('akaze') else None
BF_MATCHER = cv2.BFMatcher()
FLANN_INDEX_KDTREE = 1; index_params_sift = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50); FLANN_MATCHER_SIFT = cv2.FlannBasedMatcher(index_params_sift, search_params)
FLANN_INDEX_LSH = 6; index_params_akaze = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
FLANN_MATCHER_AKAZE = cv2.FlannBasedMatcher(index_params_akaze, search_params)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
SIFT_FEATURES_DB = {}; AKAZE_FEATURES_DB = {}
is_parsing_history = False
BOT_IS_FULLY_READY = False # Flag to prevent message processing during startup

def load_features_to_memory(features_dir, feature_type):
    start_time = time.time(); logging.info("Pre-loading %s features from %s...", feature_type, features_dir); db = {}
    if not os.path.isdir(features_dir): return db
    for filename in os.listdir(features_dir):
        if filename.endswith('.npy'):
            card_id, ref_path = os.path.splitext(filename)[0], os.path.join(features_dir, filename)
            try:
                descriptors = np.load(ref_path)
                if descriptors.size > 0:
                    if feature_type == 'SIFT' and descriptors.dtype != np.float32: descriptors = descriptors.astype(np.float32)
                    db[card_id] = descriptors
            except Exception as e: logging.warning("Could not load feature file %s: %s", filename, e)
    logging.info("Loaded %d %s feature sets in %.2f seconds.", len(db), feature_type, time.time() - start_time)
    return db

if ALGO_ENABLED.get('sift'): SIFT_FEATURES_DB = load_features_to_memory(SIFT_FEATURES_DIR, 'SIFT')
if ALGO_ENABLED.get('akaze'): AKAZE_FEATURES_DB = load_features_to_memory(AKAZE_FEATURES_DIR, 'AKAZE')
logging.info("--- Models and data loaded successfully. ---")
logging.info("--- Feature matching mode: %s ---", "FLANN" if FLANN_ENABLED else "Brute-Force (BF)")

# --- G-SHEETS FUNCTIONS ---
def authorize_gspread():
    if not GSPREAD_AVAILABLE: return None
    try:
        creds = Credentials.from_service_account_file(GSHEETS_CRED_PATH, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        return gspread.authorize(creds)
    except Exception as e: logging.error("G-SHEETS: Authorization failed: %s", e); return None

def sync_to_google_sheets():
    if not os.path.exists(CSV_LOG_PATH):
        return  # No file to sync, exit gracefully

    try:
        # If the file is empty, pandas might throw an error or create an empty dataframe
        if os.path.getsize(CSV_LOG_PATH) == 0:
            logging.info("G-SHEETS: CSV log is empty. Nothing to sync.")
            return
            
        df = pd.read_csv(CSV_LOG_PATH)

        # --- Start of new validation logic ---
        REQUIRED_COLS = ['timestamp', 'deviceAccount', 'card_name', 'rarity']
        
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

        if missing_cols:
            logging.error(
                "G-SHEETS: Sync failed. The CSV log at '%s' is missing required columns: %s. "
                "To fix this, please delete the file and let the bot regenerate it.",
                CSV_LOG_PATH, ', '.join(missing_cols)
            )
            return # Abort the sync to prevent a crash
        # --- End of new validation logic ---

        if 'uploaded_to_sheets' not in df.columns:
            df['uploaded_to_sheets'] = False
        
        df['uploaded_to_sheets'] = df['uploaded_to_sheets'].fillna(False).astype(bool)
        
        unsynced_df = df[df['uploaded_to_sheets'] == False].copy()
        
        if unsynced_df.empty:
            return # Nothing new to sync

        unsynced_df.fillna('', inplace=True)

        logging.info("G-SHEETS: Found %d new rows to sync.", len(unsynced_df))
        gc = authorize_gspread()
        if not gc:
            return

        try:
            spreadsheet = gc.open(GSHEETS_SPREADSHEET_NAME)
        except gspread.SpreadsheetNotFound:
            spreadsheet = gc.create(GSHEETS_SPREADSHEET_NAME)
            spreadsheet.share(gc.auth.service_account_email, perm_type='user', role='writer')

        try:
            worksheet = spreadsheet.worksheet("Raw Data")
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="Raw Data", rows="1000", cols="20")
            worksheet.append_row(REQUIRED_COLS)

        rows_to_append = unsynced_df[REQUIRED_COLS].values.tolist()
        worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        
        df.loc[unsynced_df.index, 'uploaded_to_sheets'] = True
        df.to_csv(CSV_LOG_PATH, index=False)
        logging.info("G-SHEETS: Successfully synced %d new rows.", len(rows_to_append))

    except pd.errors.EmptyDataError:
        logging.info("G-SHEETS: CSV log file is empty. Nothing to sync.")
    except Exception as e:
        logging.error("G-SHEETS: An error occurred during sync: %s", e, exc_info=True)

def periodic_sync_task():
    logging.info("G-SHEETS: Background sync thread started.");
    while True:
        try: sync_to_google_sheets()
        except Exception as e: logging.error("G-SHEETS: Unhandled exception in sync thread: %s", e, exc_info=True)
        time.sleep(GSHEETS_SYNC_INTERVAL)

# --- CORE FUNCTIONS ---
def find_best_match_single_matcher(query_descriptors, feature_db, matcher, feature_type):
    best_match = {'card_id': None, 'matches': -1}
    if query_descriptors is None or query_descriptors.size == 0: return best_match
    if feature_type == 'SIFT' and query_descriptors.dtype != np.float32: query_descriptors = query_descriptors.astype(np.float32)
    for card_id, ref_descriptors in feature_db.items():
        if ref_descriptors is None or ref_descriptors.size == 0: continue
        try:
            matches = matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
            good_matches = [m for pair in matches if len(pair) == 2 and (m := pair[0]).distance < 0.75 * pair[1].distance]
            if len(good_matches) > best_match['matches']:
                best_match['matches'] = len(good_matches); best_match['card_id'] = card_id
        except (cv2.error, ValueError): continue
    return best_match

def find_best_hash_match(query_hashes):
    results = {algo: {'card_id': None, 'distance': float('inf')} for algo in ['phash', 'dhash', 'whash']}
    for card_id, ref_hashes in HASH_DATABASE.items():
        for algo in results.keys():
            if ALGO_ENABLED.get(algo) and query_hashes.get(algo):
                try:
                    distance = imagehash.hex_to_hash(query_hashes[algo]) - imagehash.hex_to_hash(ref_hashes[algo])
                    if distance < results[algo]['distance']: results[algo]['distance'] = distance; results[algo]['card_id'] = card_id
                except (TypeError, KeyError): continue
    return results

def get_best_identification(analysis_data):
    for algo in ALGO_PRIORITY:
        if not ALGO_ENABLED.get(algo): continue
        result, winning_algo_name = None, algo
        if 'hash' in algo:
            result = analysis_data.get('hash_results', {}).get(algo)
        else: # Covers 'sift' and 'akaze'
            result = analysis_data.get(f'{algo}_result')
            winning_algo_name = f"{algo}_{'flann' if FLANN_ENABLED else 'bf'}"
        if result and result.get('card_id'):
            return winning_algo_name, result
    return None, None

def log_to_csv(log_data):
    if not CSV_LOG_ENABLED: return
    log_data['uploaded_to_sheets'] = False
    try:
        # Get the directory part of the path
        log_dir = os.path.dirname(CSV_LOG_PATH)
        
        # Only try to create directories if a directory path is actually specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(CSV_LOG_PATH)
        
        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_data)
            
    except (IOError, OSError) as e:
        logging.error("Could not write to CSV log file at %s: %s", CSV_LOG_PATH, e)

def run_vision_pipeline(image_path, original_filename):
    pipeline_start_time = time.time(); logging.info("PIPELINE START: %s", original_filename)
    source_image_bgr = cv2.imread(image_path)
    if source_image_bgr is None: return []
    yolo_start_time = time.time()
    results = YOLO_MODEL(source_image_bgr, verbose=False); yolo_end_time = time.time()
    detected_boxes = results[0].boxes.xyxy
    logging.info("PERF: YOLO detection found %d cards in %.4f seconds.", len(detected_boxes), yolo_end_time - yolo_start_time)

    def _process_single_crop(box, crop_idx):
        crop_start_time = time.time()
        x1, y1, x2, y2 = [int(coord) for coord in box]
        crop_bgr = source_image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0: return None
        analysis_data = {}
        
        if any(ALGO_ENABLED.get(h) for h in ['phash', 'dhash', 'whash']):
            hash_start_time = time.time()
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            upscaled_crop = crop_pil.resize((600, 824), Image.Resampling.BILINEAR)
            query_hashes = {'phash': str(imagehash.phash(upscaled_crop)) if ALGO_ENABLED.get('phash') else None, 'dhash': str(imagehash.dhash(upscaled_crop)) if ALGO_ENABLED.get('dhash') else None, 'whash': str(imagehash.whash(upscaled_crop)) if ALGO_ENABLED.get('whash') else None}
            analysis_data['hash_results'] = find_best_hash_match(query_hashes)
            logging.info("PERF: Crop #%d hashing took %.4f seconds.", crop_idx, time.time() - hash_start_time)

        if ALGO_ENABLED.get('sift') or ALGO_ENABLED.get('akaze'):
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            feature_crop_resized = cv2.resize(crop_gray, (300, 418), interpolation=cv2.INTER_AREA)
            feature_crop_enhanced = CLAHE.apply(feature_crop_resized)
            
            for feature_type, detector, feature_db in [('SIFT', SIFT, SIFT_FEATURES_DB), ('AKAZE', AKAZE, AKAZE_FEATURES_DB)]:
                if ALGO_ENABLED.get(feature_type.lower()):
                    compute_start = time.time()
                    _, descriptors = detector.detectAndCompute(feature_crop_enhanced, None)
                    logging.info("PERF: Crop #%d %s compute took %.4fs.", crop_idx, feature_type.upper(), time.time() - compute_start)

                    if descriptors is not None:
                        matcher, matcher_name = (FLANN_MATCHER_SIFT, "FLANN") if FLANN_ENABLED and feature_type == 'SIFT' else \
                                              (FLANN_MATCHER_AKAZE, "FLANN") if FLANN_ENABLED else \
                                              (BF_MATCHER, "BF")
                        match_start = time.time()
                        result = find_best_match_single_matcher(descriptors, feature_db, matcher, feature_type)
                        logging.info("PERF: Crop #%d %s %s Match took %.4fs.", crop_idx, feature_type.upper(), matcher_name, time.time() - match_start)
                        
                        if card_id := result.get('card_id'):
                            card_name = CARD_LOOKUP.get(card_id, {}).get('card_name', 'Unknown')
                            logging.info("MATCH_RESULT: Crop #%d %s (%s) -> %s (Score: %d)", crop_idx, feature_type.upper(), matcher_name, card_name, result['matches'])
                        
                        analysis_data[f'{feature_type.lower()}_result'] = result

        logging.info("PERF: Total processing for Crop #%d took %.4f seconds.", crop_idx, time.time() - crop_start_time)
        return analysis_data

    analysis_reports = []
    if detected_boxes is not None and len(detected_boxes) > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_crop = {executor.submit(_process_single_crop, box, i + 1): box for i, box in enumerate(detected_boxes)}
            for future in concurrent.futures.as_completed(future_to_crop):
                try:
                    if data := future.result(): analysis_reports.append(data)
                except Exception as exc: logging.error("A crop analysis generated an exception: %s", exc, exc_info=True)

    logging.info("PIPELINE END: %s. Total time: %.4f seconds.", original_filename, time.time() - pipeline_start_time)
    return analysis_reports

def parse_device_account_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for string_tag in root.findall('string'):
            if string_tag.get('name') == 'deviceAccount':
                logging.info("Found deviceAccount in XML: %s", string_tag.text); return string_tag.text
    except (ET.ParseError, FileNotFoundError): return "N/A"
    return "N/A"

async def process_message(message):
    """
    Downloads attachments from a message, runs the vision pipeline,
    and logs the results. Returns a string report.
    """
    image_attachments = [att for att in message.attachments if att.filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_attachments:
        return None

    image_att = image_attachments[0]
    if "FRIENDCODE" in image_att.filename:
        return f"Skipping friend code image: `{image_att.filename}`"

    image_path = os.path.join(TEMP_DIR, image_att.filename)
    temp_files_to_delete = [image_path]
    device_account = "N/A"
    
    try:
        await image_att.save(image_path)
        xml_att = next((att for att in message.attachments if att.filename.lower().endswith('.xml')), None)
        if xml_att:
            xml_path = os.path.join(TEMP_DIR, xml_att.filename)
            await xml_att.save(xml_path)
            temp_files_to_delete.append(xml_path)
            device_account = parse_device_account_from_xml(xml_path)

        loop = asyncio.get_running_loop()
        analysis_reports = await loop.run_in_executor(None, run_vision_pipeline, image_path, image_att.filename)
        
        report = f"**Analysis for `{image_att.filename}` (MSG ID: {message.id}):**\n"
        if not analysis_reports:
            report += "YOLO did not detect any cards."
        else:
            for i, data in enumerate(analysis_reports):
                report += f"\n--- **Detected Card #{i + 1}** ---\n"
                winning_algo, final_result = get_best_identification(data)
                if winning_algo and final_result:
                    card_id = final_result.get('card_id')
                    card_name = CARD_LOOKUP.get(card_id, {}).get('card_name', 'Unknown')
                    rarity = CARD_LOOKUP.get(card_id, {}).get('rarity', 'Unknown')
                    
                    report += f"Identified as **{card_name}** (Rarity: {rarity}) via `{winning_algo.upper()}`"
                    
                    log_entry = {
                        "timestamp": message.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "deviceAccount": device_account,
                        "card_name": card_name,
                        "rarity": rarity
                    }
                    logging.info("CSV_LOG: Writing -> Time: %s, Card: %s, Rarity: %s", log_entry['timestamp'], card_name, rarity)
                    log_to_csv(log_entry)
                else:
                    report += "Could not identify this card."
        
        return report

    except Exception as e:
        logging.error("ERROR processing %s: %s", image_att.filename, e, exc_info=True)
        return f"An error occurred while processing `{image_att.filename}`: {type(e).__name__}"
    finally:
        for path in temp_files_to_delete:
            if os.path.exists(path):
                os.remove(path)

def _update_last_message_id_in_config(message_id):
    """Internal helper to write the last processed message ID to config.ini."""
    if not message_id: return
    try:
        config.set('Historical', 'last_message_id', str(message_id))
        with open('data/config.ini', 'w') as configfile:
            config.write(configfile)
    except IOError as e:
        logging.error("Could not write progress to config.ini: %s", e)

async def parse_historical_messages():
    """
    Parses all messages in a channel since the last recorded message ID,
    but only processes messages that contain an XML file.
    """
    global is_parsing_history
    if not PARSE_HISTORY:
        return

    is_parsing_history = True
    logging.info("--- Starting historical message parsing. ---")
    target_channel = client.get_channel(TARGET_CHANNEL_ID)
    if not target_channel:
        logging.error("Could not find target channel with ID %s.", TARGET_CHANNEL_ID)
        is_parsing_history = False
        return

    start_point = int(LAST_MESSAGE_ID) if LAST_MESSAGE_ID and LAST_MESSAGE_ID.isdigit() else None
    
    try:
        await target_channel.send(f"`Starting historical data population. Will only process messages with XML files. Resuming from message ID: {start_point or 'Beginning'}`")
        messages_processed = 0
        final_message_id = start_point

        async for message in target_channel.history(limit=None, after=discord.Object(id=start_point) if start_point else None, oldest_first=True):
            # Always update the last message ID to ensure we make progress
            final_message_id = message.id

            # --- Start of new logic ---
            # Check for attachments and XML file presence first.
            has_xml = any(att.filename.lower().endswith('.xml') for att in message.attachments)

            if message.author == client.user or not message.attachments or not has_xml:
                if not has_xml and message.attachments:
                     logging.info("HISTORICAL: Skipping message %s (no XML file found).", message.id)
                # Save progress and skip to the next message
                _update_last_message_id_in_config(final_message_id)
                continue
            # --- End of new logic ---

            # If we reach here, the message has an XML file and is valid for processing.
            try:
                logging.info("HISTORICAL: Processing message %s (XML found)...", message.id)
                await process_message(message)
                messages_processed += 1
                
            except Exception as e:
                logging.error("Failed to process historical message %s: %s", message.id, e, exc_info=True)
            finally:
                # Update progress in the config file after every message
                _update_last_message_id_in_config(final_message_id)

        logging.info("Historical parse complete. Disabling for next bot start.")
        config.set('Historical', 'parse_history', 'false')
        if final_message_id: 
            config.set('Historical', 'last_message_id', str(final_message_id))
        with open('data/config.ini', 'w') as configfile:
            config.write(configfile)
        
        await target_channel.send(f"`Historical data population complete. Processed {messages_processed} new messages containing XML files.`")

    except Exception as e:
        logging.error("An error occurred during historical message parsing: %s", e, exc_info=True)
        await target_channel.send(f"`An error occurred during historical data population. Check logs. Progress has been saved.`")
    finally:
        is_parsing_history = False
        logging.info("--- Finished historical message parsing. ---")


# --- DISCORD EVENTS ---
@client.event
async def on_ready():
    global BOT_IS_FULLY_READY
    logging.info('Bot is ready and logged in as %s', client.user)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    if GSHEETS_ENABLED and GSPREAD_AVAILABLE:
        threading.Thread(target=periodic_sync_task, daemon=True).start()
        
    # Process all old messages first
    await parse_historical_messages()
    
    # Now, signal that the bot is ready for new messages
    BOT_IS_FULLY_READY = True
    logging.info("--- Bot is fully initialized and ready for new messages. ---")

@client.event
async def on_message(message):
    # Ignore ALL messages until the historical parse in on_ready is complete
    if not BOT_IS_FULLY_READY:
        return

    if message.author == client.user or message.channel.id != TARGET_CHANNEL_ID or not message.attachments: 
        return

    processing_msg = await message.channel.send(f"🔬 Analyzing attachments in message `{message.id}`...")
    report = await process_message(message)

    if report:
        await processing_msg.edit(content=report)
    else:
        await processing_msg.delete()


def main():
    if not BOT_TOKEN or "YOUR_DISCORD_BOT_TOKEN_HERE" in BOT_TOKEN:
        logging.critical('FATAL ERROR: Please set your bot_token in data/config.ini'); return
    try:
        client.run(BOT_TOKEN)
    except discord.errors.LoginFailure:
        logging.critical("FATAL ERROR: Login failed. The bot token in data/config.ini is likely invalid.")
    except Exception as e:
        logging.critical("An unexpected error occurred at the top level: %s", e, exc_info=True)

if __name__ == '__main__':
    main()