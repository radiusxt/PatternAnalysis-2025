#!/bin/bash

# Exit if any command fails
set -e

echo "Running train.py"
python3 ./recognition/s4696725_siamese/train.py

sleep 2

echo "Running predict.py"
python3 ./recognition/s4696725_siamese/predict.py

echo "Done!"