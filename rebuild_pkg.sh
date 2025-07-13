#!/bin/bash
# Rebuild the dataset_collector package
cd /ws_lucas
rm -rf build/ install/ log/
colcon build --packages-select dataset_collector
source install/setup.bash