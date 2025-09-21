from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    """
    return R.from_quat(quaternion).as_matrix()

def matrix_to_quaternion(matrix):
    """
    Convert a rotation matrix to a quaternion.
    """
    return R.from_matrix(matrix).as_quat()