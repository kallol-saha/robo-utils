import torch
import numpy as np
import matplotlib.pyplot as plt

def make_line(start_point, end_point, density, color):
    """
    Generate simple lines between corresponding 3D points.
    
    Args:
        start_point: Starting 3D coordinates of the lines, shape (n,3) or (3,)
        end_point: Ending 3D coordinates of the lines, shape (n,3) or (3,) 
        density: Number of points along each line
        color: RGB color values for the lines
        
    Raises:
        ValueError: If start and end points are the same
        ValueError: If start_point and end_point shapes don't match
    """
    # Convert points to numpy arrays and ensure 2D
    start = np.atleast_2d(np.array(start_point))
    end = np.atleast_2d(np.array(end_point))
    
    if start.shape != end.shape:
        raise ValueError("Start and end points must have same shape")
        
    # Calculate direction vectors
    directions = end - start
    if np.any(np.all(directions == 0, axis=1)):
        raise ValueError("Start and end points cannot be the same")
    
    # Generate evenly spaced points along each line
    t = np.linspace(0, 1, density)
    t = t.reshape(-1, 1)
    
    # Expand dimensions for broadcasting
    start = start[:, np.newaxis, :]  # Shape: (n, 1, 3)
    end = end[:, np.newaxis, :]      # Shape: (n, 1, 3)
    
    # Generate points for all lines
    points = (1-t) * start + t * end  # Shape: (n, density, 3)
    points = points.reshape(-1, 3)    # Flatten to (n*density, 3)

    # Colors for all points
    colors = np.tile(np.array(color), (points.shape[0], 1))

    return points, colors

def make_coordinate_frame(rotation, translation, length=1.0, density=50):
    """
    Generate coordinate frame points and features at a specific SE(3) transformation.
    
    Args:
        rotation: 3x3 rotation matrix (torch.Tensor or numpy array)
        translation: 3D translation vector (torch.Tensor or numpy array)
        length: Length of each axis (float)
        density: Number of points per axis (int)
    
    Returns:
        points: Nx3 array of point coordinates
        features: Nx3 array of RGB colors
    """
    # Convert to numpy if needed
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()
    
    # Ensure rotation is 3x3 and translation is 3D
    rotation = np.array(rotation).reshape(3, 3)
    translation = np.array(translation).reshape(3)
    
    # Generate base axes at origin using start/end points
    origin = np.zeros(3)
    x_pts, x_col = make_line(origin, np.array([length, 0.0, 0.0]), density=density, color=(1, 0, 0))
    y_pts, y_col = make_line(origin, np.array([0.0, length, 0.0]), density=density, color=(0, 1, 0))
    z_pts, z_col = make_line(origin, np.array([0.0, 0.0, length]), density=density, color=(0, 0, 1))
    
    # Combine all points and colors
    all_points = np.vstack([x_pts, y_pts, z_pts])
    all_colors = np.vstack([x_col, y_col, z_col])
    
    # Apply transformation: R * points + t
    transformed_points = np.dot(all_points, rotation.T) + translation
    
    return transformed_points, all_colors

def make_gripper_visualization(rotation, translation, length=1.0, density=50, color=(1, 0, 0)):
    """
    Generate coordinate frame points and features at a specific SE(3) transformation.
    
    Args:
        rotation: 3x3 rotation matrix (torch.Tensor or numpy array)
        translation: 3D translation vector (torch.Tensor or numpy array)
        length: Length of each axis (float)
        density: Number of points per axis (int)
    
    Returns:
        points: Nx3 array of point coordinates
        features: Nx3 array of RGB colors
    """
    # Convert to numpy if needed
    if torch.is_tensor(rotation):
        rotation = rotation.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()

    base_pts, base_col = make_line(np.array([0., -length, -length]),
                                   np.array([0., length, -length]),
                                   density=density, color=color)
    
    # Left and right are as if you are looking at the gripper from the manipulator base, at home position

    left_finger_pts, left_finger_col = make_line(np.array([0., -length, length]),
                                                 np.array([0., -length, -length]),
                                                 density=density, color=color)
    
    right_finger_pts, right_finger_col = make_line(np.array([0., length, length]),
                                                   np.array([0., length, -length]),
                                                   density=density, color=color)
    
    top_pts, top_col = make_line(np.array([0., 0., -length]),
                                 np.array([0., 0., -length - (2 * length)]),
                                 density=density, color=color)
    
    # Combine all points and colors
    all_points = np.vstack([base_pts, left_finger_pts, right_finger_pts, top_pts])
    all_colors = np.vstack([base_col, left_finger_col, right_finger_col, top_col])
    
    # Apply transformation: R * points + t
    transformed_points = np.dot(all_points, rotation.T) + translation
    
    return transformed_points, all_colors

def get_cube_point_cloud(side_length, center, color, points_per_face=100):
    """
    Generate points on the surface of a cube.
    
    Args:
        side_length (float): Length of cube sides
        center (np.ndarray): 3D coordinates of cube center
        color (np.ndarray): RGB color values in range [0,1]
        points_per_face (int): Number of points to generate per face
        
    Returns:
        points (np.ndarray): Nx3 array of point coordinates
        colors (np.ndarray): Nx3 array of RGB colors
    """
    # Convert inputs to numpy arrays
    center = np.asarray(center)
    color = np.asarray(color)
    
    # Generate grid of points for one face
    side_points = int(np.sqrt(points_per_face))
    x = np.linspace(-side_length/2, side_length/2, side_points)
    y = np.linspace(-side_length/2, side_length/2, side_points)
    xx, yy = np.meshgrid(x, y)
    
    points = []
    # Front and back faces (fixed z)
    for z in [-side_length/2, side_length/2]:
        points.append(np.column_stack((xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), z))))
    
    # Left and right faces (fixed x)
    for x in [-side_length/2, side_length/2]:
        points.append(np.column_stack((np.full_like(xx.flatten(), x), yy.flatten(), xx.flatten())))
    
    # Top and bottom faces (fixed y)
    for y in [-side_length/2, side_length/2]:
        points.append(np.column_stack((xx.flatten(), np.full_like(xx.flatten(), y), yy.flatten())))
        
    # Combine all points and shift to center
    points = np.vstack(points) + center
    
    # Create color array
    colors = np.tile(color, (points.shape[0], 1))
    
    return points, colors

def gaussian_3d_pcd(mean, std, num_points, color = None):
    """
    Generate a point cloud sampled uniformly from inside a 3D ellipsoid
    (centered at mean, axes lengths given by std),
    and color the points using the plasma colormap based on Mahalanobis distance.
    """
    # Uniformly sample inside a unit sphere
    points = []
    while len(points) < num_points:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) <= 1:
            points.append(p)
    points = np.array(points)
    # Scale by std and shift by mean
    points = points * std + mean

    if color is None:
        # Color by Mahalanobis distance (for visualization)
        mahal = np.sqrt(np.sum(((points - mean) / std) ** 2, axis=1))
        mahal = mahal / mahal.max()  # Normalize to [0, 1]
        colors = plt.cm.plasma(1 - mahal)[:, :3]
    else:
        colors = color

    return points, (colors * 255).astype(np.uint8)

def make_cylinder_line(start_point, end_point, radius=0.01, axial_density=50, radial_density=20, color=(1, 0, 0)):
    """
    Generate a cylindrical surface of points connecting two 3D points.

    Args:
        start_point: Starting 3D coordinates of the cylinder axis (array-like)
        end_point: Ending 3D coordinates of the cylinder axis (array-like)
        radius: Cylinder radius (float)
        axial_density: Number of samples along the cylinder axis (int)
        radial_density: Number of samples around the circumference (int)
        color: RGB color values for all points (tuple of floats in [0, 1])

    Returns:
        points: (axial_density * radial_density, 3) array of point coordinates
        colors: (axial_density * radial_density, 3) array of RGB colors
    """
    # Convert inputs to numpy arrays
    if torch.is_tensor(start_point):
        start_point = start_point.detach().cpu().numpy()
    if torch.is_tensor(end_point):
        end_point = end_point.detach().cpu().numpy()

    start = np.array(start_point, dtype=np.float32).reshape(3)
    end = np.array(end_point, dtype=np.float32).reshape(3)

    axis_vector = end - start
    axis_length = np.linalg.norm(axis_vector)
    if axis_length <= 0:
        raise ValueError("Start and end points cannot be the same")
    if radius <= 0:
        raise ValueError("Radius must be positive")
    if axial_density < 2 or radial_density < 3:
        raise ValueError("axial_density must be >= 2 and radial_density must be >= 3")

    axis_unit = axis_vector / axis_length

    # Build an orthonormal basis {v1, v2} orthogonal to axis_unit
    arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(arbitrary, axis_unit)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v1 = np.cross(axis_unit, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis_unit, v1)

    # Sample along axis and around circumference
    t = np.linspace(0.0, 1.0, axial_density, dtype=np.float32)
    thetas = np.linspace(0.0, 2.0 * np.pi, radial_density, endpoint=False, dtype=np.float32)

    base_points = start + np.outer(t, axis_unit * axis_length)  # (A, 3)
    circle_dirs = np.outer(np.cos(thetas), v1) + np.outer(np.sin(thetas), v2)  # (R, 3)
    circle_dirs *= radius

    # Broadcast to create cylinder surface points
    points = base_points[:, None, :] + circle_dirs[None, :, :]
    points = points.reshape(-1, 3)

    colors = np.tile(np.array(color, dtype=np.float32), (points.shape[0], 1))

    return points, colors