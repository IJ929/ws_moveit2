#!/usr/bin/env python3
"""
This script collects joint configurations and end-effector poses of a robot
and saves them to a CSV file.
"""

import time
import numpy as np
import pandas as pd
import tqdm
# generic ros libraries
import rclpy
from rclpy.logging import get_logger

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy
)
from geometry_msgs.msg import PoseStamped
# from moveit.core.kinematic_constraints import construct_joint_constraint

import sys


def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.data_collection")

    try:
        panda = MoveItPy(node_name="moveit_py")
        panda_arm = panda.get_planning_component("panda_arm")
    except Exception as e:
        logger.error(f"Failed to initialize MoveItPy: {e}")
        rclpy.shutdown()
        sys.exit(1)
    

    ###########################################################################
    # Collecting Data
    ###########################################################################
    try:
        
        logger.info("Data collection has started.")
        # Get the robot model and create a RobotState object
        robot_model = panda.get_robot_model()
        robot_state = RobotState(robot_model)
        arm_joint_model_group = robot_model.get_joint_model_group("panda_arm")
        end_effector_link = "panda_link8"

        # Collect the data 
        num_data_points = 5_000_000
        num_dofs = len(arm_joint_model_group.joint_model_names)-1
        num_end_effector_pose_dimensions = 7  # x, y, z, qx, qy, qz, qw
        table_data = np.empty((num_data_points, num_dofs + num_end_effector_pose_dimensions), dtype=float)
        logger.info(f"Collecting {num_data_points} data points...")

        tr = tqdm.trange(num_data_points, desc="Collecting Data", unit="point")
        for i in tr:
            robot_state.set_to_random_positions(arm_joint_model_group)
            robot_state.update()
            
            row_data = []
            
            joint_positions = [robot_state.joint_positions[name] for name in arm_joint_model_group.joint_model_names[:-1]]
            # logger.info(f"Joint positions: {joint_positions}")
            
            pose = robot_state.get_pose(end_effector_link)
            ee_pose = [pose.position.x, pose.position.y, pose.position.z,
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            # logger.info(f"End-Effector Pose: {ee_pose}")
            row_data.extend(joint_positions)
            row_data.extend(ee_pose)    
            table_data[i] = row_data

        logger.info("Data collection complete.")

        # Define CSV file path and headers
        csv_file_path = "data/panda_arm_training_data.csv"
        # Corrected line: access .joint_model_names as an attribute
        joint_names = arm_joint_model_group.joint_model_names[:-1]
        pose_headers = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
        csv_headers = joint_names + pose_headers

        df = pd.DataFrame(table_data, columns=csv_headers)
        logger.info(f"Data collected: {df.shape[0]} rows, {df.shape[1]} columns")

        # Write the collected data to the CSV file
        df.to_csv(csv_file_path, index=False)        
        logger.info(f"Successfully saved data to {csv_file_path}")

    except IOError as e:
        logger.error(f"Error writing to file {csv_file_path}: {e}")

    # Properly clean up objects before shutdown
    finally:
        logger.info("Cleaning up resources...")
        if robot_state:
            del robot_state
        if arm_joint_model_group:
            del arm_joint_model_group
        if robot_model:
            del robot_model
        if panda_arm:
            del panda_arm
        if panda:
            del panda
        
        # Give some time for cleanup
        time.sleep(0.1)

    rclpy.shutdown()

if __name__ == "__main__":
    main()