#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_arpa_file> <output_bin_file>"
  exit 1
fi

# Record the current working directory
CURRENT_DIR=$(pwd)

INPUT_ARPA=$1
OUTPUT_BIN=$2

# Clone the kenlm repository
git clone https://github.com/kpu/kenlm.git

# Build kenlm
cd kenlm || exit
mkdir -p build
cd build || exit
cmake ..
make -j4

# Run build_binary
cd bin || exit
./build_binary "$CURRENT_DIR/$INPUT_ARPA" "$CURRENT_DIR/$OUTPUT_BIN"

cd "$CURRENT_DIR" || exit
echo "Conversion completed: $INPUT_ARPA -> $OUTPUT_BIN"
