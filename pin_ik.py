#%%
import numpy as np
import pandas as pd
import pinocchio
from tqdm import tqdm, trange
from numpy.linalg import norm, solve
from example_robot_data import load

def generate_dataset(model, ee_frame_id, num_samples=1000, csv_path=None):
    """
    Generate a dataset of joint configurations and corresponding end-effector poses.
    
    Args:
        model: Pinocchio model
        ee_frame_id: End-effector frame ID
        num_samples: Number of samples to generate
        csv_path: Optional path to save the dataset as CSV
    
    Returns:
        pd.DataFrame: DataFrame containing joint configurations and end-effector poses
    """
    data = model.createData()
    n_dofs = model.nq
    n_pose = 7  # 3 for position, 4 for quaternion
    rows = np.empty((num_samples, n_dofs + n_pose))
    
    for i in trange(num_samples, desc="Generating dataset", ncols=100, unit=" samples"):
        # Random joint configuration within joint limits
        q = np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        
        # Forward kinematics to get end-effector pose
        pinocchio.forwardKinematics(model, data, q)
        pinocchio.updateFramePlacement(model, data, ee_frame_id)
        ee_pose = data.oMf[ee_frame_id]
        
        # Extract position and quaternion
        pos = ee_pose.translation
        quat = pinocchio.Quaternion(ee_pose.rotation).coeffs()  # (x, y, z, w)
        
        row = list(q) + list(pos) + list(quat)
        rows[i] = row

    columns = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
               'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    df = pd.DataFrame(rows, columns=columns)
    if csv_path is not None:
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
    return df

def test_dataset_generation(csv_path, model, ee_frame_id):
    df = pd.read_csv(csv_path)
    expected_columns = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
                        'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    if list(df.columns) != expected_columns:
        raise ValueError(f"CSV columns do not match expected format. Found: {list(df.columns)}")
    
    # check any nan or inf values
    if df.isnull().values.any():
        raise ValueError("CSV contains NaN values.")
    if np.isinf(df.values).any():
        raise ValueError("CSV contains Inf values.")
    
    df = df.head(1000)  # Use a subset for quicker evaluation

    # compute the error of each row
    errors = []
    for index, row in df.iterrows():
        q_row = row[expected_columns[:7]].values
        target_pos = row[['pos_x', 'pos_y', 'pos_z']].values
        target_quat = row[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values
        target_rot = pinocchio.Quaternion(target_quat[3], target_quat[0], target_quat[1], target_quat[2]).toRotationMatrix()
        target_pose = pinocchio.SE3(target_rot, target_pos)

        data = model.createData()
        pinocchio.forwardKinematics(model, data, q_row)
        pinocchio.updateFramePlacement(model, data, ee_frame_id)
        err = pinocchio.log(target_pose * data.oMf[ee_frame_id].inverse()).vector
        err_norm = norm(err)
        errors.append(err_norm)
    df_err = pd.DataFrame(errors, columns=['pose_error_m'])
    print(df_err.describe())

def build_model(robot_name, urdf_path=None, joints_to_lock_names=None, q0_full=None):
    """
    Build a reduced Pinocchio model by locking specified joints.
    
    Args:
        robot_name: Name of the robot (for loading)
        urdf_path: Path to the URDF file
        joints_to_lock_names: List of joint names to lock
        q0_full: Reference configuration for locked joints (default: neutral)
    
    Returns:
        model: Pinocchio model
    """
    robot = load(robot_name) if urdf_path is None else pinocchio.buildModelFromUrdf(urdf_path)
    model = robot.model
    print(f"Full model nq (degrees of freedom): {model.nq}")
    print(f"Full model joint names: {[n for n in model.names]} \n")

    if q0_full is None:
        q0_full = pinocchio.neutral(model)

    if not joints_to_lock_names is None:
        joints_to_lock_ids = [model.getJointId(name) for name in joints_to_lock_names]
        print(f"Joint IDs to lock: {joints_to_lock_ids}")
        model = pinocchio.buildReducedModel(model, joints_to_lock_ids, q0_full)
    
    print(f"\nModel nq (degrees of freedom): {model.nq}")
    print(f"Model joint names: {[n for n in model.names]}")
    
    return model

def forward_kinematics(model, q, ee_frame_id):
    """
    Compute the forward kinematics for a given joint configuration.
    Args:
        model: Pinocchio model
        q: Joint configuration
        ee_frame_id: End-effector frame ID
    Returns:
        SE3: End-effector pose
    """
    data = model.createData()
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.updateFramePlacement(model, data, ee_frame_id)
    return data.oMf[ee_frame_id]

def compute_se3_error(current_pose, target_pose):
    """
    Compute the SE3 error between current and target poses.
    Args:
        current_pose: Current SE3 pose
        target_pose: Target SE3 pose
    Returns:
        err: Pose error vector
    """
    err = pinocchio.log(target_pose * current_pose.inverse()).vector
    return err

def inverse_kinematics(model, ee_frame_id, target_pose, q_init=None, 
                      eps=1e-4, max_iter=1000, dt=1e-1, damp=1e-6, verbose=False):
    """
    Solve inverse kinematics for a given target pose.
    
    Args:
        model: Pinocchio model
        ee_frame_id: Target frame ID
        target_pose: Desired SE3 pose
        q_init: Initial joint configuration (default: neutral)
        eps: Convergence tolerance
        max_iter: Maximum iterations
        dt: Integration time step
        damp: Damping factor for pseudo-inverse
        verbose: Print iteration progress
    
    Returns:
        tuple: (success, q_final, error_norm, iterations)
    """
    data = model.createData()
    q = q_init if q_init is not None else pinocchio.neutral(model)
    
    for i in range(max_iter):
        # Forward kinematics
        forward_kinematics(model, q, ee_frame_id)
        
        # Compute pose error
        err = compute_se3_error(data.oMf[ee_frame_id], target_pose)
        err_norm = norm(err)
        
        # Check convergence
        if err_norm < eps:
            return True, q, err_norm, i + 1
        
        # Compute Jacobian and update
        J = pinocchio.computeFrameJacobian(
            model, data, q, ee_frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        v = J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * dt)
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: error = {err_norm:.4f}")
    
    return False, q, err_norm, max_iter

def test_inverse_kinematics(model, ee_frame_id):
    # Define target pose
    R_des = pinocchio.rpy.rpyToMatrix(np.pi, 0, 0)
    p_des = np.array([0.5, 0.2, 0.5])
    target_pose = pinocchio.SE3(R_des, p_des)

    # 3. --- Solve inverse kinematics
    success, q_final, final_error, iterations = inverse_kinematics(
        model, ee_frame_id, target_pose, verbose=True
    )

    # 4. --- Print results
    if success:
        print(f"\nConvergence achieved in {iterations} iterations! 🤖")
    else:
        print(f"\nWarning: Did not converge after {iterations} iterations.")

    print(f"Final joint configuration: {q_final.flatten().tolist()}")
    print(f"Final pose error: {final_error:.1e} m")

    # Verify final pose
    achieved_pose = forward_kinematics(model, q_final, ee_frame_id)

    print(f"Desired position: {target_pose.translation.T}")
    print(f"Achieved position: {achieved_pose.translation.T}")

class RobotModel:
    def __init__(self, robot_name, urdf_path=None, joints_to_lock_names=None, q0_full=None):
        self.model = build_model(robot_name, urdf_path, joints_to_lock_names, q0_full)
        self.ee_frame_id = self.model.getFrameId('panda_finger_joint1')
        
    def sample_random_configurations(self, num_samples=1):
        return np.random.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit, (num_samples, self.model.nq))
    
    def convert_pose_to_se3(self, pose):
        pos = pose[:3]
        quat = pose[3:]
        R = pinocchio.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()
        return pinocchio.SE3(R, pos)
    
    def convert_se3_to_pose(self, se3):
        pos = se3.translation
        quat = pinocchio.Quaternion(se3.rotation).coeffs()
        pose = np.array(list(pos) + list(quat))
        return pose

    def forward_kinematics(self, q):
        """
        Compute the forward kinematics for a given joint configuration.
        Args:
            q: Joint configuration
        Returns:
            SE3: End-effector pose
        """
        return forward_kinematics(self.model, q, self.ee_frame_id)
    
    def inverse_kinematics(self, target_pose, q_init=None, 
                           eps=1e-4, max_iter=1000, dt=1e-1, damp=1e-6, verbose=False):
        """
        Solve inverse kinematics for a given target pose.
        Args:
            target_pose: Desired SE3 pose
            q_init: Initial joint configuration (default: neutral)
            eps: Convergence tolerance
            max_iter: Maximum iterations
            dt: Integration time step
            damp: Damping factor for pseudo-inverse
            verbose: Print iteration progress
        Returns:
            tuple: (success, q_final, error_norm, iterations)"""
        return inverse_kinematics(self.model, self.ee_frame_id, target_pose, q_init, eps, max_iter, dt, damp, verbose)
    
    def compute_pose_error(self, current_pose, target_pose):
        """
        Compute the pose error between current and target poses.
        Args:
            current_pose: Current translation, quaternion pose
            target_pose: Target SE3 pose
        Returns:
            err: Pose error vector
        """
        current_se3 = self.convert_pose_to_se3(current_pose)
        target_se3 = self.convert_pose_to_se3(target_pose)
        return self.compute_se3_error(current_se3, target_se3)

    def compute_se3_error(self, current_pose, target_pose):
        """
        Compute the pose error between current and target poses.
        Args:
            current_pose: Current SE3 pose
            target_pose: Target SE3 pose
        Returns:
            err: Pose error vector
        """
        return compute_se3_error(current_pose, target_pose)

#%%
if __name__ == "__main__":
    # 1. --- Load the full robot model
    robot_name = 'panda'
    joints_to_lock = ['panda_finger_joint1', 'panda_finger_joint2']
    model = build_model(robot_name, joints_to_lock_names=joints_to_lock)
    ee_frame_id = model.getFrameId('panda_finger_joint1')

    #%%
    # 2. --- Define the task
    test_inverse_kinematics(model, ee_frame_id)

    #%%
    csv_path = 'data/panda_arm_training_data_pinocchio.csv'
    # df = generate_dataset(model, ee_frame_id, num_samples=5_000_000, csv_path=csv_path)

    #%%
    test_dataset_generation(csv_path, model, ee_frame_id)
    #%%
