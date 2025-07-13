#!/usr/bin/env python3
"""
A script to collect an IK solutions dataset for a Franka Emika Panda robotic arm.
This dataset is a csv file with columns for joint configurations and end-effector poses (position + quaternion).
"""

import rclpy
from rclpy.logging import get_logger
import pandas as pd
import numpy as np
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped

def main():
    """
    Main function to collect the dataset.
    """
    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("data_collector_node")

    # Instantiate MoveItPy instance
    panda = MoveItPy(node_name="moveit_py_data_collector")
    panda_arm = panda.get_planning_component("panda_arm")
    logger.info("MoveItPy instance created")

    # Get the robot model
    robot_model = panda.get_robot_model()

    # Create a RobotState instance
    robot_state = RobotState(robot_model)

    # Number of samples to collect
    num_samples = 1000
    logger.info(f"Collecting {num_samples} samples.")

    # Lists to store the data
    data = []

    for i in range(num_samples):
        # Set the robot state to a new random position
        robot_state.set_to_random_positions(panda_arm.get_joint_model_group())
        robot_state.update()

        # Check for collisions for the current random state
        if panda.get_planning_scene_monitor().get_planning_scene().is_state_colliding(robot_state, panda_arm.get_joint_model_group().name):
            logger.warning("Skipping colliding state.")
            continue

        # Get the joint positions
        joint_positions = robot_state.get_joint_group_positions(panda_arm.get_joint_model_group())

        # Get the end-effector pose
        ee_pose = panda_arm.get_end_effector_link_pose()

        # Append data
        row = {
            'joint_1': joint_positions[0],
            'joint_2': joint_positions[1],
            'joint_3': joint_positions[2],
            'joint_4': joint_positions[3],
            'joint_5': joint_positions[4],
            'joint_6': joint_positions[5],
            'joint_7': joint_positions[6],
            'pos_x': ee_pose.pose.position.x,
            'pos_y': ee_pose.pose.position.y,
            'pos_z': ee_pose.pose.position.z,
            'quat_x': ee_pose.pose.orientation.x,
            'quat_y': ee_pose.pose.orientation.y,
            'quat_z': ee_pose.pose.orientation.z,
            'quat_w': ee_pose.pose.orientation.w,
        }
        data.append(row)
        
        if (i+1) % 100 == 0:
            logger.info(f"Collected {i+1}/{num_samples} samples.")


    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = "ik_solutions_dataset.csv"
    df.to_csv(csv_filename, index=False)
    logger.info(f"Dataset saved to {csv_filename}")

    rclpy.shutdown()

if __name__ == "__main__":
    main()