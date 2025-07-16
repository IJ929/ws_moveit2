#!/bin/bash
# Rebuild the dataset_collector package
# cd /ws_lucas
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select my_moveit_demo
