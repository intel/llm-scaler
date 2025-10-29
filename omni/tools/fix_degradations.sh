#!/bin/bash

# ==============================================================================
# Script Name: patch_degradations.sh
# Description: Replaces a specific import statement in basicsr's degradations.py.
#              This is often needed to fix compatibility issues with newer
#              PyTorch/torchvision versions where rgb_to_grayscale was moved.
# Requirements: sudo access
# ==============================================================================

# --- Configuration Variables ---
FILE_PATH="/usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py"
OLD_STRING="from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW_STRING="from torchvision.transforms.functional import rgb_to_grayscale"
BACKUP_EXTENSION=".bak" # Extension for the backup file

# --- Script Start ---
echo "--------------------------------------------------------"
echo "Starting patch script for basicsr/data/degradations.py"
echo "--------------------------------------------------------"
echo "Target file: $FILE_PATH"

# 1. Check if the target file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: Target file not found at '$FILE_PATH'."
    echo "Please ensure basicsr is installed for Python 3.10."
    echo "Exiting."
    exit 1
fi

# 2. Check if the original string exists in the file
# Using grep -q for quiet mode, -F for fixed string match
if ! grep -qF "$OLD_STRING" "$FILE_PATH"; then
    echo "The original string to be replaced was not found in '$FILE_PATH'."
    echo "It's possible the file has already been patched or has a different version."
    echo "No changes made. Exiting gracefully."
    exit 0
fi

# 3. Create a backup of the original file
BACKUP_FILE="${FILE_PATH}${BACKUP_EXTENSION}"
echo "Creating a backup of the original file to: $BACKUP_FILE"
sudo cp "$FILE_PATH" "$BACKUP_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create backup file. Please check permissions."
    echo "Exiting without making changes."
    exit 1
fi
echo "Backup created successfully."

# 4. Perform the replacement using sed
echo "Replacing string in '$FILE_PATH'..."
echo "  Old: '$OLD_STRING'"
echo "  New: '$NEW_STRING'"
echo "(Requires sudo password)"

# Using '|' as a delimiter for sed's substitute command (s///)
# because the strings contain '/' characters.
# -i flag for in-place editing.
sudo sed -i "s|$OLD_STRING|$NEW_STRING|g" "$FILE_PATH"

if [ $? -ne 0 ]; then
    echo "Error: The 'sed' command failed. Please check permissions or script syntax."
    echo "Original file might be corrupted. Check '$BACKUP_FILE'."
    echo "Exiting."
    exit 1
fi

# 5. Verify the replacement
echo "Verification:"
if grep -qF "$NEW_STRING" "$FILE_PATH" && ! grep -qF "$OLD_STRING" "$FILE_PATH"; then
    echo "  Success! The string has been replaced correctly."
    echo "  Old string no longer found."
    echo "  New string found."
else
    echo "  Warning: Verification failed. The replacement might not be complete or correct."
    echo "  Please check the file '$FILE_PATH' manually."
    echo "  You can restore the original file from '$BACKUP_FILE' if needed."
    exit 1
fi

echo "--------------------------------------------------------"
echo "Patch completed successfully!"
echo "If you encounter any issues, you can restore the file:"
echo "sudo mv \"$BACKUP_FILE\" \"$FILE_PATH\""
echo "--------------------------------------------------------"

exit 0