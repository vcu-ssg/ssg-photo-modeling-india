#!/usr/bin/env python3
import sys
import numpy as np

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

# Parse input string like: "X -45 Y 90 Z 90"
args = sys.argv[1:]
if not args or len(args) % 2 != 0:
    print("Usage: generate_transform.py X -45 Y 90 Z 90")
    sys.exit(1)

# Build the composite rotation
R = np.eye(3)
i = 0
while i < len(args):
    axis = args[i].upper()
    angle = float(args[i+1])
    if axis == 'X':
        R = rotation_matrix_x(angle) @ R
    elif axis == 'Y':
        R = rotation_matrix_y(angle) @ R
    elif axis == 'Z':
        R = rotation_matrix_z(angle) @ R
    else:
        print(f"Unknown rotation axis: {axis}")
        sys.exit(1)
    i += 2

# Build final 4x4 matrix
transform = np.eye(4)
transform[:3,:3] = R

# Print in COLMAP's format
for row in transform:
    print(' '.join(f"{val:.8f}" for val in row))
