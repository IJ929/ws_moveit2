# Test with demo robots (if available)
ros2 launch moveit2_tutorials demo.launch.py

# Test Panda robot demo
ros2 launch moveit_resources_panda_moveit_config demo.launch.py

# Test UR5 robot demo  
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e