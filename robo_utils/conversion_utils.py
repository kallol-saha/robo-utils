from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import open3d as o3d

def quaternion_to_matrix(quaternion, format='xyzw'):
    """
    Convert a quaternion to a rotation matrix.
    """
    if format == 'wxyz':
        return R.from_quat(quaternion, scalar_first=True).as_matrix()
    elif format == 'xyzw':
        return R.from_quat(quaternion, scalar_first=False).as_matrix()
    else:
        raise ValueError(f"Invalid quaternion format: {format}")
    

def matrix_to_quaternion(matrix, format='xyzw'):
    """
    Convert a rotation matrix to a quaternion.
    """
    if format == 'wxyz':
        return R.from_matrix(matrix).as_quat(scalar_first=True)
    elif format == 'xyzw':
        return R.from_matrix(matrix).as_quat(scalar_first=False)
    else:
        raise ValueError(f"Invalid matrix format: {format}")

def pose_to_transformation(pose: np.ndarray, format='xyzw'):
    """
    Convert a pose to a transformation matrix.
    pose is a numpy array of shape (7,) or (N, 7)
    Returns a numpy array of shape (4, 4) or (N, 4, 4)
    """

    # Make dimensions consistent:
    if not isinstance(pose, np.ndarray):
        raise ValueError(f"Pose must be a numpy array")

    reshape_pose = False
    if len(pose.shape) == 1:
        reshape_pose = True
        pose = pose[np.newaxis, :]
    transformation_matrix = np.eye(4, 4)[np.newaxis, :, :].repeat(pose.shape[0], axis=0)

    # Convert to transformation matrix:
    transformation_matrix[..., :3, :3] = quaternion_to_matrix(pose[..., 3:], format=format)
    transformation_matrix[..., :3, 3] = pose[..., :3]

    if reshape_pose:
        transformation_matrix = transformation_matrix[0, :, :]
    return transformation_matrix

def transformation_to_pose(transformation_matrix, format='xyzw'):
    """
    Convert a transformation matrix to a pose.
    Accepts numpy array of shape (4, 4) or (N, 4, 4).
    Returns numpy array of shape (7,) or (N, 7) correspondingly.
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    reshape_out = False
    if transformation_matrix.ndim == 2:
        reshape_out = True
        transformation_matrix = transformation_matrix[np.newaxis, ...]  # (1,4,4)

    # Positions
    pos = transformation_matrix[:, :3, 3]  # (N,3)
    # Rotations to quaternions (vectorized)
    quat = matrix_to_quaternion(transformation_matrix[:, :3, :3], format=format)  # (N,4)

    poses = np.concatenate([pos, quat], axis=-1)  # (N,7)
    if reshape_out:
        poses = poses[0]
    return poses

def invert_transformation(transformation_matrix):
    """Inverts the given transformation matrix.
    
    Args:
        transformation_matrix: (..., 4, 4) transformation matrix (numpy ndarray only)
        
    Returns:
        inverse_transformation_matrix: (..., 4, 4) inverted transformation matrix (numpy ndarray)
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    if transformation_matrix.ndim == 2:
        tm = transformation_matrix[np.newaxis, ...]
        squeeze_out = True
    else:
        tm = transformation_matrix
        squeeze_out = False

    R = tm[..., :3, :3]
    t = tm[..., :3, 3]
    R_inv = np.linalg.inv(R)
    t_inv = -np.matmul(R_inv, t[..., None])[..., 0]

    batch_shape = tm.shape[:-2]
    I = np.eye(4, dtype=tm.dtype).reshape((1, 4, 4)).repeat(np.prod(batch_shape, dtype=int) if len(batch_shape) > 0 else 1, axis=0)
    I = I.reshape(*batch_shape, 4, 4)
    I[..., :3, :3] = R_inv
    I[..., :3, 3] = t_inv

    if squeeze_out:
        return I[0]
    return I

def invert_pose(pose: np.ndarray, format='xyzw') -> np.ndarray:
    """Inverts the given pose.
    pose is a numpy array of shape (7,)
    Returns a numpy array of shape (7,)
    """
    inverse_transformation_matrix = invert_transformation(pose_to_transformation(pose, format=format))
    return transformation_to_pose(inverse_transformation_matrix, format=format)

def transform_pcd(pcd, transform):
    """Transforms the given point cloud by the given transformation matrix.

    Args:
    -----
        pcd: Nx3 point cloud (numpy array or torch tensor)
        transform: 4x4 transformation matrix (numpy array or torch tensor)

    Returns:
    --------
            pcd_new: Nx3 transformed point cloud (same type as input)
    """

    # Convert both to numpy arrays if types don't match
    if isinstance(pcd, torch.Tensor) != isinstance(transform, torch.Tensor):
        print("WARNING: Point cloud and transformation matrix must be same type (both numpy arrays or both torch tensors)")
        if isinstance(pcd, torch.Tensor):
            pcd = pcd.cpu().numpy()
        if isinstance(transform, torch.Tensor):
            transform = transform.cpu().numpy()
    
    if isinstance(pcd, torch.Tensor):
        if pcd.shape[1] != 4:
            ones = torch.ones((pcd.shape[0], 1), device=pcd.device)
            pcd = torch.cat((pcd, ones), dim=1)
        pcd_new = torch.matmul(transform, pcd.T)[:-1, :].T
    else:
        if pcd.shape[1] != 4:
            pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new

def furthest_point_sample(pcd: np.ndarray | torch.Tensor, num_points: int = 1024):
    """
    """
    if isinstance(pcd, torch.Tensor):
        original_type = "torch"
        tensor_device = pcd.device
        pcd = pcd.detach().cpu().numpy()
    elif not isinstance(pcd, np.ndarray):
        raise ValueError("pcd must be a numpy array or torch tensor")
    else:
        original_type = "numpy"
        tensor_device = None

    if pcd.shape[1] != 3 and len(pcd.shape) != 2:
        raise ValueError("pcd must be a numpy array or torch tensor of shape (N, 3)")

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    downsampled_pcd = pcd_o3d.farthest_point_down_sample(num_samples=num_points)
    downsampled_pcd = np.asarray(downsampled_pcd.points)

    if original_type == "torch":
        return torch.from_numpy(downsampled_pcd).to(tensor_device)
    else:
        return downsampled_pcd

def move_pose_along_local_z(pose: np.ndarray | torch.Tensor, distance: float, format: str = 'wxyz'):
    """Translate pose(s) along their local +Z axis by a given distance.

    Args:
        pose: Pose as (7,) or (N,7), numpy ndarray or torch tensor, ordered
              (x, y, z, qw, qx, qy, qz) if format=='wxyz', or (x, y, z, qx, qy, qz, qw) if 'xyzw'.
        distance: Scalar translation to apply along local Z (meters).
        format: Quaternion convention, 'wxyz' or 'xyzw'.

    Returns:
        Pose(s) with updated position, same shape/type/device as input.
    """

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    if isinstance(pose, np.ndarray):
        single = False
        p = pose
        if p.ndim == 1:
            p = p[np.newaxis, :]
            single = True
        pos = p[:, :3]
        if format == 'wxyz':
            quat = p[:, 3:7]
        elif format == 'xyzw':
            quat = p[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        else:
            raise ValueError(f"Invalid quaternion format: {format}")

        R = quaternion_to_matrix(quat, format='wxyz')  # (N,3,3)
        z_axis = R[:, :, 2]  # (N,3)
        pos_new = pos + (distance * z_axis)
        out = p.copy()
        out[:, :3] = pos_new
        if single:
            out = out[0]
        return out

    else:
        raise ValueError("pose must be a numpy array or torch tensor")

def move_transformation_along_local_z(transformation_matrix: np.ndarray, distance: float):
    """Transforms the given transformation matrix along the local +Z axis by a given distance.

    Args:
        transformation_matrix: (4, 4) transformation matrix (numpy array or torch tensor)
        distance: Scalar translation to apply along local Z (meters).

    Returns:
        transformation_matrix_new: (4, 4) transformed transformation matrix (same type as input)
    """
    if not isinstance(transformation_matrix, np.ndarray):
        raise ValueError("transformation_matrix must be a numpy array")

    single = False
    T = transformation_matrix
    if T.ndim == 2:
        if T.shape != (4, 4):
            raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")
        T = T[np.newaxis, ...]
        single = True
    elif T.ndim == 3:
        if T.shape[-2:] != (4, 4):
            raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")
    else:
        raise ValueError("Expected (4,4) or (N,4,4) transformation matrix")

    R = T[:, :3, :3]
    t = T[:, :3, 3]
    z_axis = R[:, :, 2]
    t_new = t + distance * z_axis

    out = T.copy()
    out[:, :3, 3] = t_new
    if single:
        return out[0]
    return out