#!/bin/bash
# Rebuild the dataset_collector package
cd /ws_lucas
rm -rf build/ install/ log/
colcon build
source install/setup.bash