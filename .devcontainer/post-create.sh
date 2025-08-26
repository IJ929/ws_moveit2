#!/bin/bash
set -e

echo "📦 Running post-create commands..."

source /root/.bashrc
python3 get-pip.py
pip3 install pandas