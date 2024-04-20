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

# Example usage
roll = 0  # Replace these with your actual values
pitch = 0
yaw = 90

roll1 = -236.32 -3
pitch1 = 90.00
yaw1 = -146.31 - 3

roll2 = -98.48 - 1
pitch2 = 90.00
yaw2 = -345.96 - 2

roll3 = -261.61 - 2
pitch3 = 90.00
yaw3 = -165.96 - 3

roll4 = -351.56 - 2
pitch4 = 90.00
yaw4 = -270.00 - 3

roll5 = -133.44 - 1
pitch5 = 90.00
yaw5 = -26.56 - 3

roll6 = -332.75
pitch6 = 90.00
yaw6 = -200.56 - 4

roll7 = -161.56
pitch7 = 90.00
yaw7 = -26.56 - 3

print("Ring 1 Location: ", 49.999023, -2318.999756, 891.999939)
forward_vector1 = compute_forward_vector(roll1, pitch1, yaw1)
print("Forward Vector 1:", -1 * forward_vector1)

print("Ring 2 Location: ", 789.744812, -4441.580078, 1146.999878)
forward_vector2 = compute_forward_vector(roll2, pitch2, yaw2)
print("Forward Vector 2:", -1 * forward_vector2)

print("Ring 3 Location: ", 1513.665405, -6436.097656, 1827.999756)
forward_vector3 = compute_forward_vector(roll3, pitch3, yaw3)
print("Forward Vector 3:", -1 * forward_vector3)

print("Ring 4 Location: ", 1087.512695,-8471.915039,  2325.999756)
forward_vector4 = compute_forward_vector(roll4, pitch4, yaw4)
print("Forward Vector 4:", -1 * forward_vector4)

print("Ring 5 Location: ", 1215.035522, -10457.389648, 2667.999756)
forward_vector5 = compute_forward_vector(roll5, pitch5, yaw5)
print("Forward Vector 5:", -1 * forward_vector5)

print("Ring 6 Location: ", 2471.902832, -12348.450195, 2860.999756)
forward_vector6 = compute_forward_vector(roll6, pitch6, yaw6)
print("Forward Vector 6:", -1 * forward_vector6)

print("Ring 7 Location: ", 4225.242676, -14245.586914, 2701.999756)
forward_vector7 = compute_forward_vector(roll7, pitch7, yaw7)
print("Forward Vector 7:", -1 * forward_vector7)
