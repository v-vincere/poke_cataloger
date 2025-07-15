# Pokémon TCG Card Identification Bot

This project uses a combination of computer vision techniques to automatically identify Pokémon TCG cards from images. It's designed to be used with a Discord bot that can analyze images posted in a specific channel.

## Features

-   **Card Detection:** Uses a YOLOv11 model to detect the location of cards in an image.
-   **Multi-faceted Identification:** Employs a prioritized pipeline of identification algorithms:
    -   **Perceptual Hashing (pHash, dHash, wHash):** Quickly finds similar-looking cards.
    -   **Feature Matching (SIFT, AKAZE):** More robustly identifies cards based on key points, even with variations in angle or lighting.
-   **Discord Bot Integration:** A Python-based Discord bot that monitors a channel for new images, runs the identification pipeline, and posts the results.
-   **Google Sheets Logging:** Can automatically log identified cards to a Google Sheet for easy tracking and cataloging.
-   **Historical Data Parsing:** Can parse through the history of a Discord channel to identify cards in past messages.

## How It Works

1.  **Image Submission:** A user posts an image containing one or more Pokémon cards to a designated Discord channel.
2.  **Card Detection:** The Discord bot downloads the image. A YOLO (You Only Look Once) object detection model scans the image to find the bounding boxes of each card.
3.  **Image Cropping & Preprocessing:** Each detected card is cropped from the main image. The cropped image is then preprocessed (resized, enhanced) to standardize it for the identification algorithms.
4.  **Identification Pipeline:** The bot runs a series of algorithms in a specific order of priority (configurable in `data/config.ini`):
    *   It first tries fast perceptual hashing algorithms (like AKAZE, SIFT, pHash) to find a match against a pre-computed database of reference card images.
    *   If hashing doesn't yield a confident result, it moves to more computationally intensive feature matching algorithms (SIFT or AKAZE). These algorithms extract unique feature descriptors from the card and match them against a database of descriptors from all known cards.
5.  **Result Reporting:** The bot reports the name and rarity of the identified card back to the Discord channel.
6.  **Logging:** The identification result (timestamp, card name, rarity, etc.) is logged to a local CSV file and, if enabled, synced to a Google Sheet.

## Project Structure

```
for_release/
│
├── .gitignore
├── create_feature_database.py
├── create_hash_database.py
├── data/
│   ├── config.ini              # Main configuration file
│   ├── card_hashes.json        # Pre-computed hash database
│   ├── features_akaze/         # Pre-computed AKAZE features
│   └── features_sift/          # Pre-computed SIFT features
├── dataset.yaml
├── discord_bot.py
├── duplicate_labels.py
├── identification_log.csv
├── README.md
├── setup_logging.py
├── train.py
└── yolo11n.pt
```

---

## Setup and Installation

This guide assumes you are using Windows and will be setting up the project within the Windows Subsystem for Linux (WSL).

### 1. Setting up WSL (Windows Subsystem for Linux)

If you don't have WSL installed, follow these steps:

1.  **Open PowerShell as Administrator:** Search for "PowerShell" in the Start Menu, right-click it, and select "Run as administrator."
2.  **Install WSL:** Run the following command. This will install the necessary WSL components and a default Ubuntu distribution.
    ```powershell
    wsl --install
    ```
3.  **Reboot:** Restart your computer when prompted.
4.  **Initial Ubuntu Setup:** After rebooting, Ubuntu will start and ask you to create a username and password. This will be your user account within the Linux environment.

### 2. Installing Python and Dependencies

1.  **Open your WSL terminal:** You can do this by searching for "Ubuntu" in the Start Menu.
2.  **Update Package Lists:**
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
3.  **Install Python and Pip:**
    ```bash
    sudo apt install python3 python3-pip -y
    ```
4.  **Install `opencv-contrib-python`:** The feature matching algorithms (SIFT/AKAZE) require the "contrib" version of OpenCV.
    ```bash
    pip install opencv-contrib-python
    ```
5.  **Install Other Python Libraries:**
    ```bash
    pip install discord pandas ultralytics imagehash torch torchvision torchaudio configparser gspread google-auth-oauthlib
    ```
    *Note: The `torch` installation might take some time. If you have a powerful NVIDIA GPU, you can follow the official PyTorch instructions to install a CUDA-enabled version for better performance.*

### 3. Project Configuration

1.  **Navigate to the Project Directory:** In your WSL terminal, go to the `for_release` directory.
2.  **Edit `config.ini`:** Open the `data/config.ini` file in a text editor. You'll need to fill in your specific API keys and settings.

    ```ini
    [Discord]
    bot_token = YOUR_DISCORD_BOT_TOKEN_HERE
    target_channel_id = YOUR_TARGET_DISCORD_CHANNEL_ID_HERE

    [Paths]
    ; IMPORTANT: Update this to the absolute path of your project directory in WSL.
    ; Example: /home/your_username/ml_poke/for_release/
    base_path = /path/to/your/project/
    ...

    [GoogleSheets]
    enabled = false ; Set to true to enable Google Sheets logging
    spreadsheet_name = your-spreadsheet-name
    ; Path to your Google Cloud service account JSON file.
    credentials_path = data/your-credentials.json
    ```

#### **How to get API Keys and IDs:**

*   **`bot_token` (Discord):**
    1.  Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    2.  Click "New Application."
    3.  Give it a name and click "Create."
    4.  Go to the "Bot" tab on the left.
    5.  Click "Add Bot," then "Yes, do it!"
    6.  Under the bot's username, click "Reset Token" to view and copy your token. **Treat this like a password!**
    7.  You also need to enable "Message Content Intent" under the "Privileged Gateway Intents" section on this page.

*   **`target_channel_id` (Discord):**
    1.  In your Discord client, go to User Settings > Advanced.
    2.  Enable "Developer Mode."
    3.  Go to the channel you want the bot to monitor, right-click its name in the channel list, and click "Copy Channel ID."

*   **`credentials_path` (Google Sheets):**
    1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
    2.  Create a new project.
    3.  Enable the "Google Drive API" and "Google Sheets API" for your project.
    4.  Go to "Credentials" > "Create Credentials" > "Service Account."
    5.  Give the service account a name, grant it the "Editor" role, and finish.
    6.  Go to the "Keys" tab for your new service account, click "Add Key" > "Create new key," choose "JSON," and create it. A JSON file will be downloaded.
    7.  Place this JSON file in the `data/` subdirectory and update the `credentials_path` in `config.ini`.
    8.  Finally, open the downloaded JSON file, find the `client_email` address, and share your Google Sheet with that email address, giving it "Editor" permissions.

---

## How to Run the Bot

Once everything is set up and configured, you can start the bot.

1.  **Navigate to the Project Directory:** Make sure you are in the `for_release` directory in your WSL terminal.
2.  **Run the Bot:**
    ```bash
    python3 discord_bot.py
    ```

The bot will log in to Discord, and you should see output in your terminal indicating it's ready. You can now post images in your target channel to have them identified.

## Advanced Usage

### Building Your Own Reference Database (Optional)

This project comes with pre-built databases for hashes and features. However, if you want to add your own card images to the identification system, you will need to rebuild the databases.

1.  **Add Images:** Place your new card images inside the `data/card_images/` directory. The filename (without the extension) will be the card's ID (e.g., `swsh4-55.jpg`).
2.  **Run the Database Script:** Execute the following command from the project's root directory:
    ```bash
    python3 create_feature_database.py
    ```
    This will update `data/card_hashes.json` and the `data/features_sift/` and `data/features_akaze/` directories with the new card information.

### Training Your Own Model (Optional)

The project includes `yolo11n.pt`, a pre-trained model for detecting cards. If you want to train your own model on a custom dataset of card images, you can use the `train.py` script.

1.  **Prepare Your Dataset:** You'll need a labeled dataset of images in the YOLO format. The `dataset.yaml` file points to the expected locations for training and validation images and labels.
2.  **Run Training:**
    ```bash
    python3 train.py
    ```
    The training script will create a new directory in the `runs/` folder containing your trained model weights (`best.pt`), graphs, and metrics.
3.  **Update Config:** To use your newly trained model, update the `model_path` in `data/config.ini` to point to your new `best.pt` file.