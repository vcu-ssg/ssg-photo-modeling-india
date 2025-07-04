#!/usr/bin/env python3
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def rotation_matrix_x(theta):
    rad = np.radians(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad),  np.cos(rad)]
    ])

def rotation_matrix_y(theta):
    rad = np.radians(theta)
    return np.array([
        [ np.cos(rad), 0, np.sin(rad)],
        [ 0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)]
    ])

def rotation_matrix_z(theta):
    rad = np.radians(theta)
    return np.array([
        [np.cos(rad), -np.sin(rad), 0],
        [np.sin(rad),  np.cos(rad), 0],
        [0, 0, 1]
    ])

# Parse input like: RX 30 RY -20 RZ 10 TX 5 TY 0 TZ -3 S 1.2
args = sys.argv[1:]
if len(args) % 2 != 0:
    print("Usage: generate_transform.py RX 30 RY -20 RZ 10 TX 5 TY 0 TZ -3 S 1.2")
    sys.exit(1)

# Defaults
R = np.eye(3)
T = np.array([0.0, 0.0, 0.0])
scale = 1.0

# Process args
i = 0
while i < len(args):
    key = args[i].upper()
    try:
        val = float(args[i+1])
    except ValueError:
        print(f"Invalid numeric value: {args[i+1]}")
        sys.exit(1)
    
    if key == 'RX':
        R = rotation_matrix_x(val) @ R
    elif key == 'RY':
        R = rotation_matrix_y(val) @ R
    elif key == 'RZ':
        R = rotation_matrix_z(val) @ R
    elif key == 'TX':
        T[0] += val
    elif key == 'TY':
        T[1] += val
    elif key == 'TZ':
        T[2] += val
    elif key == 'S':
        scale = val
    else:
        print(f"Unknown key: {key}")
        sys.exit(1)
    i += 2

# Convert rotation matrix to quaternion (returns x, y, z, w)
rot_obj = Rotation.from_matrix(R)
qx, qy, qz, qw = rot_obj.as_quat()

# Print in COLMAP format: scale qw qx qy qz tx ty tz
print(f"{scale:.8f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {T[0]:.8f} {T[1]:.8f} {T[2]:.8f}")
