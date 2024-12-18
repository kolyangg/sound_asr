#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define URLs of the datasets
URLS=(
    # "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    "https://www.openslr.org/resources/12/dev-other.tar.gz"
    # "https://www.openslr.org/resources/12/test-clean.tar.gz"
    "https://www.openslr.org/resources/12/test-other.tar.gz"
    # "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
    "https://www.openslr.org/resources/12/train-other-500.tar.gz"
)

# Define the target directory
TARGET_DIR="data/datasets/librispeech"

# Create the target directory if it doesn't exist
echo "Creating target directory at '$TARGET_DIR'..."
mkdir -p "$TARGET_DIR"

# Navigate to the target directory
cd "$TARGET_DIR"

# Download each file
for URL in "${URLS[@]}"; do
    # Extract the filename from the URL
    FILE_NAME=$(basename "$URL")
    
    # Check if the file already exists to avoid re-downloading
    if [ -f "$FILE_NAME" ]; then
        echo "File '$FILE_NAME' already exists. Skipping download."
    else
        echo "Downloading '$FILE_NAME'..."
        wget "$URL"
    fi
done

# Extract the downloaded tar.gz files
echo "Extracting downloaded files..."
for FILE in *.tar.gz; do
    echo "Extracting '$FILE'..."
    tar -xzf "$FILE"
done

# Remove the original tar.gz files after extraction
echo "Removing compressed files..."
rm -f *.tar.gz

echo "All tasks completed successfully!"
