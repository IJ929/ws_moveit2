#%%
import numpy as np
import pandas as pd
import pinocchio
from tqdm import tqdm, trange
from numpy.linalg import norm, solve
from example_robot_data import load

def verify_moveit2_csv(csv_file, model, ee_frame_id):
    df = pd.read_csv(csv_file)
    expected_columns = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
                        'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    if list(df.columns) != expected_columns:
        raise ValueError(f"CSV columns do not match expected format. Found: {list(df.columns)}")
    df = df.head(1000)  # Use a subset for quicker evaluation

    # compute the error of each row
    errors = []
    for index, row in df.iterrows():
        q_row = row[expected_columns[:model.nq]].values
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

def generate_dataset(model, ee_frame_id, num_samples=1000):
    """
    Generate a dataset of joint configurations and corresponding end-effector poses.
    
    Args:
        model: Pinocchio model
        ee_frame_id: End-effector frame ID
        num_samples: Number of samples to generate
    
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
    return df

def inverse_kinematics(model, frame_id, target_pose, q_init=None, 
                      eps=1e-4, max_iter=1000, dt=1e-1, damp=1e-6, verbose=False):
    """
    Solve inverse kinematics for a given target pose.
    
    Args:
        model: Pinocchio model
        frame_id: Target frame ID
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
        pinocchio.forwardKinematics(model, data, q)
        pinocchio.updateFramePlacement(model, data, frame_id)
        
        # Compute pose error
        current_pose = data.oMf[frame_id]
        err = pinocchio.log(target_pose * current_pose.inverse()).vector
        err_norm = norm(err)
        
        # Check convergence
        if err_norm < eps:
            return True, q, err_norm, i + 1
        
        # Compute Jacobian and update
        J = pinocchio.computeFrameJacobian(
            model, data, q, frame_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        v = J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * dt)
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: error = {err_norm:.4f}")
    
    return False, q, norm(err), max_iter

# 1. --- Load the full robot model
robot = load('panda')
model = robot.model
# urdf_path = "/root/ws_moveit/src/moveit_resources/panda_description/urdf/panda.urdf"
# model = pinocchio.buildModelFromUrdf(urdf_path)
print(f"Full model nq (degrees of freedom): {model.nq}")
print(f"Full model joint names: {[n for n in model.names]} \n")

# 2. --- Identify the joints to lock (the gripper fingers)
# We want to lock 'panda_finger_joint1' and 'panda_finger_joint2'
joints_to_lock_names = ['panda_finger_joint1', 'panda_finger_joint2']
joints_to_lock_ids = [model.getJointId(name) for name in joints_to_lock_names]
print(f"Joint IDs to lock: {joints_to_lock_ids}")

# 3. --- Create the reduced model (7-DOF arm)
# We provide a reference configuration where the joints will be locked.
# Using the neutral configuration is a good default.
q0_full = pinocchio.neutral(model) 
model = pinocchio.buildReducedModel(model, joints_to_lock_ids, q0_full)

print(f"\nReduced model nq (degrees of freedom): {model.nq}")
print(f"Reduced model joint names: {[n for n in model.names]}")


#%%
# 2. --- Define the task
EE_FRAME_ID = model.getFrameId('panda_finger_joint1')

# Define target pose
R_des = pinocchio.rpy.rpyToMatrix(np.pi, 0, 0)
p_des = np.array([0.5, 0.2, 0.5])
target_pose = pinocchio.SE3(R_des, p_des)

# 3. --- Solve inverse kinematics
success, q_final, final_error, iterations = inverse_kinematics(
    model, EE_FRAME_ID, target_pose, verbose=True
)

# 4. --- Print results
if success:
    print(f"\nConvergence achieved in {iterations} iterations! 🤖")
else:
    print(f"\nWarning: Did not converge after {iterations} iterations.")

print(f"Final joint configuration: {q_final.flatten().tolist()}")
print(f"Final pose error: {final_error:.1e} m")

# Verify final pose
data = model.createData()
pinocchio.forwardKinematics(model, data, q_final)
pinocchio.updateFramePlacement(model, data, EE_FRAME_ID)

print(f"Desired position: {target_pose.translation.T}")
print(f"Achieved position: {data.oMf[EE_FRAME_ID].translation.T}")

#%%
# df = generate_dataset(model, EE_FRAME_ID, num_samples=5_000_000)
# df.to_csv('data/panda_arm_training_data_pinocchio.csv', index=False)


#%%
verify_moveit2_csv('data/panda_arm_training_data_pinocchio.csv', model, EE_FRAME_ID)
#%%
