from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

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

def pose_to_transformation(pose, format='xyzw'):
    """
    Convert a pose to a transformation matrix.
    pose is a numpy array of shape (7,)
    Returns a numpy array of shape (4, 4)
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = quaternion_to_matrix(pose[3:], format=format)
    transformation_matrix[:3, 3] = pose[:3]
    return transformation_matrix

def transformation_to_pose(transformation_matrix, format='xyzw'):
    """
    Convert a transformation matrix to a pose.
    transformation_matrix is a numpy array of shape (4, 4)
    Returns a numpy array of shape (7,)
    """
    pose = np.zeros(7)
    pose[:3] = transformation_matrix[:3, 3]
    pose[3:] = matrix_to_quaternion(transformation_matrix[:3, :3], format=format)
    return pose

def invert_transformation(transformation_matrix):
    """Inverts the given transformation matrix.
    
    Args:
        transformation_matrix: 4x4 transformation matrix (numpy array or torch tensor)
        
    Returns:
        inverse_transformation_matrix: 4x4 inverted transformation matrix (same type as input)
    """
    if isinstance(transformation_matrix, torch.Tensor):
        inverse_rotation_matrix = torch.linalg.inv(transformation_matrix[:3, :3])
        inverse_translation = -torch.matmul(inverse_rotation_matrix, transformation_matrix[:3, 3])
        inverse_transformation_matrix = torch.eye(4, device=transformation_matrix.device)
    else:
        inverse_rotation_matrix = np.linalg.inv(transformation_matrix[:3, :3])
        inverse_translation = -np.matmul(inverse_rotation_matrix, transformation_matrix[:3, 3])
        inverse_transformation_matrix = np.eye(4)

    inverse_transformation_matrix[:3, :3] = inverse_rotation_matrix
    inverse_transformation_matrix[:3, 3] = inverse_translation

    return inverse_transformation_matrix

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