import os
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("panda", package_name="moveit_resources_panda_moveit_config")
        .robot_description(file_path="config/panda.urdf.xacro")
        .trajectory_execution(file_path="config/gripper_controllers.yaml")
        .to_moveit_configs()
    )

    # Data collector node
    data_collector_node = Node(
        package="dataset_collector", # Replace with your package name
        executable="data_collector.py", # Your script's name
        name="moveit_py_data_collector",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([data_collector_node])