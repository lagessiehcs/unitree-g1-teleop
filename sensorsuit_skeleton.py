"""
Sensor suit skeleton data and forward kinematics.

This module provides skeleton data and FK computation without ROS2 dependencies,
suitable for use in both real-time teleoperation and offline retargeting pipelines.

Extracted from gmr_teleop.py to avoid ROS2 dependency for offline use.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class HumanBodyMeasurements():
    Hips = 0.24
    Shoulder = 0.4
    UpperArm = 0.3
    ForeArm = 0.18
    Thigh = 0.45
    Shank = 0.40
    Foot = 0.25
    Hand = 0.18
    Back = 0.5


def _get_sensor_frame_offsets(offset_lowerback_deg=10):
    """
    Get sensor frame orientation offsets for sensor suit.

    Args:
        offset_lowerback_deg: Lower back sensor tilt offset in degrees (default: 10)

    Returns:
        dict: Mapping joint name to scipy Rotation for sensor mounting orientation
    """
    offset_lowerback = offset_lowerback_deg
    return {
        "LowerBack":        R.from_euler('XYZ', [0.0, 0.0, 0.0], degrees=True),
        "UpperBack":        R.from_euler('XYZ', [offset_lowerback, 0.0, 0.0], degrees=True),
        "ThighLeft":        R.from_euler('XYZ', [offset_lowerback, -90.0, 0.0], degrees=True),
        "ThighRight":       R.from_euler('XYZ', [offset_lowerback, 90.0, 0.0], degrees=True),
        "ShankLeft":        R.from_euler('XYZ', [offset_lowerback, -90.0, 0.0], degrees=True),
        "ShankRight":        R.from_euler('XYZ', [offset_lowerback, 90.0, 0.0], degrees=True),
        "FootLeft":         R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, 180], degrees=True),
        "FootRight":        R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, 180], degrees=True),
        "UpperArmLeft":     R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, -90.0], degrees=True),
        "UpperArmRight":    R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, 90.0], degrees=True),
        "ForeArmLeft":      R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, -90.0], degrees=True),
        "ForeArmRight":      R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, 90.0], degrees=True),
        "HandLeft":         R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, -90.0], degrees=True),
        "HandRight":        R.from_euler('XYZ', [-(90.0-offset_lowerback), 0.0, 90.0], degrees=True),
    }


def _get_sensor_local_offsets():
    """
    Get local position offsets in parent frame for sensor suit skeleton.

    Returns:
        dict: Mapping joint name to [x, y, z] offset in parent frame
    """
    h = HumanBodyMeasurements()
    return {
        "LowerBack":        [0.0, 0.0, 0.0],
        "UpperBack":        [0.0, 0.0, 0.0],
        "ThighLeft":        [-h.Hips/2., 0.0, 0.0],
        "ThighRight":       [h.Hips/2., 0.0, 0.0],
        "ShankLeft":        [0.0, -h.Thigh, 0.0],
        "ShankRight":        [0.0, -h.Thigh, 0.0],
        "FootLeft":         [0.0, -h.Shank, 0.0],
        "FootRight":        [0.0, -h.Shank, 0.0],
        "UpperArmLeft":     [-h.Shoulder/2, h.Back, 0.0],
        "UpperArmRight":    [h.Shoulder/2, h.Back, 0.0],
        "ForeArmLeft":      [0.0, -h.UpperArm, 0.0],
        "ForeArmRight":      [0.0, -h.UpperArm, 0.0],
        "HandLeft":         [0.0, -h.ForeArm, 0.0],
        "HandRight":        [0.0, -h.ForeArm, 0.0],
    }


def _get_sensor_kinematic_chain():
    """
    Get kinematic chain (parent-child relationships) for sensor suit skeleton.

    Returns:
        dict: Mapping joint name to parent joint name (None for root)
    """
    return {
        "LowerBack":        None,
        "UpperBack":        "LowerBack",
        "ThighLeft":        "LowerBack",
        "ThighRight":       "LowerBack",
        "ShankLeft":        "ThighLeft",
        "ShankRight":       "ThighRight",
        "FootLeft":         "ShankLeft",
        "FootRight":        "ShankRight",
        "UpperArmLeft":     "UpperBack",
        "UpperArmRight":    "UpperBack",
        "ForeArmLeft":      "UpperArmLeft",
        "ForeArmRight":     "UpperArmRight",
        "HandLeft":         "ForeArmLeft",
        "HandRight":        "ForeArmRight",
    }


def get_sensorsuit_skeleton_data(offset_lowerback_deg=10):
    """
    Get sensor suit skeleton data for offline retargeting.

    Public API function that exposes skeleton initialization data.

    Args:
        offset_lowerback_deg: Lower back sensor tilt offset in degrees (default: 10)

    Returns:
        tuple: (body_measurements, kinematic_chain, local_offsets, frame_offsets)
    """
    return (
        HumanBodyMeasurements(),
        _get_sensor_kinematic_chain(),
        _get_sensor_local_offsets(),
        _get_sensor_frame_offsets(offset_lowerback_deg)
    )


def compute_sensor_forward_kinematics(sensor_orientations_global, root_position, kinematic_chain, local_offsets):
    """
    Compute forward kinematics for sensor suit skeleton.

    Tree-based FK propagation: iteratively compute child positions from parent
    positions and orientations using kinematic chain and local offsets.

    Args:
        sensor_orientations_global: dict mapping joint name to scipy Rotation (global frame)
        root_position: [x,y,z] position for root joint
        kinematic_chain: dict mapping joint name to parent joint name
        local_offsets: dict mapping joint name to [x,y,z] local offset in parent frame

    Returns:
        dict: Mapping joint name to [x,y,z] position

    Source: Extracted from G1TeleopNode.gmr() method in gmr_teleop.py
    """
    # Find root joint
    root = None
    for joint, parent in kinematic_chain.items():
        if parent is None:
            root = joint
            break

    # Initialize positions
    positions = {root: root_position}

    # Track which positions have been computed
    pos_set_flag = {joint: False for joint in kinematic_chain.keys()}
    pos_set_flag[root] = True

    # Tree-based FK propagation
    while any(f is False for f in pos_set_flag.values()):
        for link in kinematic_chain.keys():
            base = kinematic_chain[link]

            if link == root:
                continue

            if pos_set_flag[link] or not pos_set_flag[base]:
                continue

            positions[link] = positions[base] + sensor_orientations_global[base].apply(local_offsets[link])

            pos_set_flag[link] = True

    return positions
