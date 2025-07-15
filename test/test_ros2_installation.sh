# Test ROS 2 environment
ros2 --help

# Check available ROS 2 packages
ros2 pkg list | grep -E "(moveit|ros2)"

# Test ROS 2 communication
ros2 topic list

# Check ROS 2 nodes
ros2 node list