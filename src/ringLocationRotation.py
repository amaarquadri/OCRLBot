import numpy as np

def compute_forward_vector(roll, pitch, yaw):
    # Convert angles from degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Rotation matrices around x, y, and z axes
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Start with a forward vector pointing along the X-axis
    forward = np.array([0, 0, 1])

    # Apply the rotations
    forward = np.dot(R_z, np.dot(R_y, np.dot(R_x, forward)))
    return forward

def get_ring_data():
    locations = [
        (49.999023, -2318.999756, 891.999939),
        (789.744812, -4441.580078, 1146.999878),
        (1513.665405, -6436.097656, 1827.999756),
        (1087.512695,-8471.915039,  2325.999756),
        (1215.035522, -10457.389648, 2667.999756),
        (2471.902832, -12348.450195, 2860.999756),
        (4225.242676, -14245.586914, 2701.999756)
    ]
    rotations = [
        (-236.32-3, 90.00, -146.31-3),
        (-98.48-1, 90.00, -345.96-2),
        (-261.61-2, 90.00, -165.96-3),
        (-351.56-2, 90.00, -270.00-3),
        (-133.44-1, 90.00, -26.56-3),
        (-332.75, 90.00, -200.56-4),
        (-161.56, 90.00, -26.56-3)
    ]
    rings_data = []

    for loc, (roll, pitch, yaw) in zip(locations, rotations):
        position = np.array(loc)
        forward_vector = -1 * compute_forward_vector(roll, pitch, yaw)
        rings_data.append((position, forward_vector))
    
    return rings_data

if __name__ == "__main__":
    print(get_ring_data())