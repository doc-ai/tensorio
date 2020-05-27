#! /bin/sh

##
## Downloads models from our public tensorio-examples bucket
## 

MODELS_URL=https://storage.googleapis.com/tensorio-examples/r1.13/models.tar.gz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR=`mktemp -d`

# Check if tmp dir was created

if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

# Deletes the temp directory

function cleanup {      
  rm -rf "$WORK_DIR"
}

# Register the cleanup function to be called on the EXIT signal

trap cleanup EXIT

# Remove the current models directory

echo "Deleting current models directory"

if [[ -d models ]]; then
  rm -r models
fi

# Download models into work directory and unzip

echo "Downloading models"

cd $WORK_DIR
curl -o models.tar.gz $MODELS_URL

echo "Unarchiving models"

tar -xzvf models.tar.gz

# Copy models back into directory from which this script was run

cp -r $WORK_DIR/models $SCRIPT_DIR/models

echo "Models downloaded, cleaning up"