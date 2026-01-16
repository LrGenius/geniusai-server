#!/bin/bash
#set -e

echo "Building geniusai-server..."

# Check for conda and activate environment
if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please install it first."
    exit 1
fi

# Check if the environment exists
if ! conda env list | grep -q "geniusai_server"; then
    echo "Conda environment 'geniusai_server' not found, creating it..."
    conda env create -f environment.yml
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate geniusai_server
# python post_install.py

# Check if pyinstaller is installed
if ! command -v pyinstaller &> /dev/null
then
    echo "pyinstaller could not be found, please install it in the geniusai_server environment."
    exit 1
fi

# Set environment variable to fix OpenMP library conflict during build
export KMP_DUPLICATE_LIB_OK=TRUE

pyinstaller geniusai_server.spec --noconfirm

conda deactivate

echo "Build complete. The executable can be found in the dist/ directory."

