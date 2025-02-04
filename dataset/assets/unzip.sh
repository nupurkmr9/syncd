#!/bin/bash

ROOT_DIR="$1"

# Loop through all .zip files in the current directory
for zip_file in "$ROOT_DIR"/*.zip; do
    # Extract the stem (filename without .zip extension)
    dir_name="${zip_file%.zip}"
    dir_name="${dir_name##*/}"

    target_dir="$ROOT_DIR/$dir_name"

    # Create the destination directory
    mkdir -p "$target_dir"

    # Unzip the file into the destination directory
    unzip -q "$zip_file" -d "$target_dir"

    echo "Unzipped: $zip_file -> $target_dir"
done