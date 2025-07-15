# Check if MoveIt 2 packages are available
ros2 pkg list | grep moveit

# Test MoveIt 2 setup assistant (GUI test)
ros2 launch moveit_setup_assistant setup_assistant.launch.py

# Check MoveIt 2 planning capabilities
ros2 run moveit_ros_planning moveit_planning_execution

# List MoveIt 2 services
ros2 service list | grep moveit