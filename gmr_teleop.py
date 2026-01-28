import rclpy
from rclpy.node import Node

from sensorsuit_msgs.msg import ImuReadings
from scipy.spatial.transform import Rotation as R
from unitree_hg.msg import LowCmd, LowState

from utils.crc import CRC
import numpy as np
import threading
import time

import signal
import sys
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from tqdm import tqdm

import argparse

from sensorsuit_skeleton import (
    compute_sensor_forward_kinematics,
    _get_sensor_frame_offsets,
    _get_sensor_kinematic_chain,
    _get_sensor_local_offsets
)

class SensorID():
    Chest = 1
    LowerBack = 2
    UpperArmLeft = 3
    UpperArmRight = 4
    ForearmLeft = 5
    ForearmRight = 6
    HandLeft = 7
    HandRight = 8
    ThighLeft = 9
    ThighRight = 10
    ShankLeft = 11
    ShankRight = 12
    FootLeft = 13
    FootRight = 14
    PelvisLeft = 15
    PelvisRight = 16
    ShoulderLeft = 17
    ShoulderRight = 18
    ChestLeft = 19
    ChestRight = 20
    LowerPinkyLeft = 21
    UpperPinkyLeft = 22
    LowerRingLeft = 23
    UpperRingLeft = 24
    LowerMiddleLeft = 25
    UpperMiddleLeft = 26
    LowerIndexLeft = 27
    UpperIndexLeft = 28
    LowerThumbLeft = 29
    UpperThumbLeft = 30
    LowerThumbRight = 31
    UpperThumbRight = 32
    LowerIndexRight = 33
    UpperIndexRight = 34
    LowerMiddleRight = 35
    UpperMiddleRight = 36
    LowerRingRight = 37
    UpperRingRight = 38
    LowerPinkyRight = 39
    UpperPinkyRight = 40
    UpperBack = 47
    Head = 48
    Wristband = 50

class G1JointID():
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof
    NotUsedJoint = 29

G1_JOINTS_23DOF = [
    G1JointID.LeftHipPitch,
    G1JointID.LeftHipRoll,
    G1JointID.LeftHipYaw,
    G1JointID.LeftKnee,
    G1JointID.LeftAnklePitch,
    G1JointID.LeftAnkleRoll,
    G1JointID.RightHipPitch,
    G1JointID.RightHipRoll,
    G1JointID.RightHipYaw,
    G1JointID.RightKnee,
    G1JointID.RightAnklePitch,
    G1JointID.RightAnkleRoll,
    G1JointID.WaistYaw,
    G1JointID.LeftShoulderPitch,
    G1JointID.LeftShoulderRoll,
    G1JointID.LeftShoulderYaw,
    G1JointID.LeftElbow,
    G1JointID.LeftWristRoll,
    G1JointID.RightShoulderPitch,
    G1JointID.RightShoulderRoll,
    G1JointID.RightShoulderYaw,
    G1JointID.RightElbow,
    G1JointID.RightWristRoll
]

G1_UPPERBODY_JOINTS_23DOF = [
    G1JointID.WaistYaw,
    G1JointID.LeftShoulderPitch,
    G1JointID.LeftShoulderRoll,
    G1JointID.LeftShoulderYaw,
    G1JointID.LeftElbow,
    G1JointID.LeftWristRoll,
    G1JointID.RightShoulderPitch,
    G1JointID.RightShoulderRoll,
    G1JointID.RightShoulderYaw,
    G1JointID.RightElbow,
    G1JointID.RightWristRoll
]

ACTIV_SENSOR_IDS = [
    SensorID.LowerBack,
    SensorID.UpperArmLeft,
    SensorID.UpperArmRight,
    SensorID.ForearmLeft,
    SensorID.ForearmRight,
    SensorID.HandLeft,
    SensorID.HandRight,
    SensorID.ThighLeft,
    SensorID.ThighRight,
    SensorID.ShankLeft,
    SensorID.ShankRight,
    SensorID.FootLeft,
    SensorID.FootRight,
    SensorID.ShoulderLeft,
    SensorID.ShoulderRight,
    SensorID.UpperBack,
    SensorID.Head,
]


class G1TeleopNode(Node):

    def __init__(self, mode):

        super().__init__('g1_teleop_node')

        if mode == "whole-body":
            self.upperbody = False
        else:
            self.upperbody = True
        self.imus_subscription = self.create_subscription(
            ImuReadings,
            'sensorsuit/imus',
            self.listener_callback,
            10,
            )

        self.calib_lock = threading.Lock()


        if not self.upperbody:
            self.current_angles_subscription = self.create_subscription(
                LowState,
                'lowstate',
                self.current_angles_callback,
                10,
            )

        if self.upperbody:
            teleop_topic = '/arm_sdk'
            self.robot_joints = G1_UPPERBODY_JOINTS_23DOF
        else:
            teleop_topic = '/lowcmd'
            self.robot_joints = G1_JOINTS_23DOF


        self.root = "LowerBack"

        # Initialize skeleton data
        self.offset_orientation = _get_sensor_frame_offsets(offset_lowerback_deg=10)
        self.offset_position = _get_sensor_local_offsets()
        self.link_base_mapping = _get_sensor_kinematic_chain()

        self.human_data = {
            "LowerBack":        [[None]*3, [None]*4], # Root
            "UpperBack":        [[None]*3, [None]*4],
            "ThighLeft":        [[None]*3, [None]*4],
            "ThighRight":       [[None]*3, [None]*4],
            "ShankLeft":        [[None]*3, [None]*4],
            "ShankRight":        [[None]*3, [None]*4],
            "FootLeft":         [[None]*3, [None]*4],
            "FootRight":        [[None]*3, [None]*4],
            "UpperArmLeft":     [[None]*3, [None]*4],
            "UpperArmRight":    [[None]*3, [None]*4],
            "ForeArmLeft":      [[None]*3, [None]*4],
            "ForeArmRight":      [[None]*3, [None]*4],
            "HandLeft":         [[None]*3, [None]*4],
            "HandRight":        [[None]*3, [None]*4],
        }
        self.human_data[self.root] = [self.offset_position[self.root], self.offset_orientation[self.root].as_quat(scalar_first=True)]

        self.name_id_mapping = {
            "LowerBack":        SensorID.LowerBack,
            "UpperBack":        SensorID.UpperBack,
            "ThighLeft":        SensorID.ThighLeft,
            "ThighRight":       SensorID.ThighRight,
            "ShankLeft":        SensorID.ShankLeft,
            "ShankRight":       SensorID.ShankRight,
            "FootLeft":         SensorID.FootLeft,
            "FootRight":        SensorID.FootRight,
            "UpperArmLeft":     SensorID.UpperArmLeft,
            "UpperArmRight":    SensorID.UpperArmRight,
            "ForeArmLeft":      SensorID.ForearmLeft,
            "ForeArmRight":     SensorID.ForearmRight,
            "HandLeft":         SensorID.HandLeft,
            "HandRight":        SensorID.HandRight,
        }
        self.id_name_mapping = {id: name for name, id in self.name_id_mapping.items()}

        self.shutdown_requested = False
        self.shutting_down = False
        self.exit_teleop = False
                
        self.publish_frequency = 500 #Hz
        
        self.teleop_enabled = False
        self.current_arm_sdk = 0.0
        
        # PD gains
        self.kp = 50.0
        self.kd = 1.0
        
        self.lowcmd_publisher = self.create_publisher(LowCmd, teleop_topic, 10)

        self.step_fast = np.radians(2)
        self.step_med = np.radians(1)
        self.step_slow = np.radians(0.5)
        self.arm_sdk_step = 0.002
        # self.step_sizes = {id: np.radians(self.step) for id in self.robot_joints}
        


        # IMU IDs
        self.calib_orientation = {name: 0.0 for name in self.name_id_mapping.keys()}
        self.calibrated = False

        self.sensor_orientations_local = {name: None for name in self.name_id_mapping.keys()}

        self.sensor_orientations_global = {name: None for name in self.name_id_mapping.keys()}
        self.sensor_orientations_global[self.root] = self.offset_orientation[self.root]


        self.g1_target_joint_angles = {id: 0.0 for id in self.robot_joints}
        self.g1_current_cmd = {id: 0.0 for id in self.robot_joints}
        self.current_cmd_initialized = False

        self.g1_current_joint_angles = {id: 0.0 for id in self.robot_joints}

        self.crc = CRC()

        self.sensor_orientations = {id: None for id in ACTIV_SENSOR_IDS}

        self.left_elbow_roll = 0.0
        self.left_elbow_yaw = 0.0
        self.right_elbow_roll = 0.0
        self.right_elbow_yaw = 0.0

        self.retargeter = GMR(
                    src_human="sensorsuit",
                    tgt_robot="unitree_g1",
                    actual_human_height=1.75,
                    verbose = True
                )


        self.publishAngles_thread = threading.Thread(target=self.publishAngles)
        self.publishAngles_thread.start()

        self.gmr = threading.Thread(target=self.gmr)
        self.gmr.start()

        self.enableTeleop_thread = threading.Thread(target=self.enableTeleop)
        self.enableTeleop_thread.start()
    

    def calibrate(self):
        """Calibrate once using a known pose."""
        for name in self.name_id_mapping.keys():
                self.calib_orientation[name] = self.sensor_orientations_local[self.root] * self.offset_orientation[name] * self.sensor_orientations_local[name].inv()

        print("Calibration complete!")
        user_input = input("Press [ENTER] to start teleop or [q] to quit...")
        print()

        if user_input == 'q':
            self.exit_teleop = True
            self.shutting_down = True
        else:
            self.teleop_enabled = True
            if self.upperbody:
                pbar = tqdm(total=100, ncols=60, desc = "Starting teleop", bar_format="{l_bar}{bar}")
                prev=0
                while self.current_arm_sdk < 1.0:
                    current = int(100 * self.current_arm_sdk)
                    pbar.update(abs(prev-current))
                    prev = current
                    time.sleep(0.1)
                pbar.update(abs(prev-100))
                pbar.close()
                print()
            self.calibrated = True

    def set_angle(self, index):

        target = self.g1_target_joint_angles[index]
        current_cmd = self.g1_current_cmd[index]
        # step_size = self.step_sizes[index]
        # if target == 0.0 and index in [G1JointID.LeftShoulderRoll, G1JointID.LeftShoulderPitch, G1JointID.RightShoulderRoll, G1JointID.RightShoulderPitch]:
        #     step_size = np.radians(self.step/5)

        diff = target-current_cmd
        sign_diff = 1 if diff >= 0 else -1

        if not self.teleop_enabled:
            step_size = self.step_slow
        elif abs(diff) > np.radians(20):
            step_size = self.step_med
        else:
            step_size = self.step_fast

        if abs(diff) < step_size:
            current_cmd = target 
        else:
            current_cmd += sign_diff * step_size

        # if index == G1JointID.RightShoulderRoll:
        #     print("current_cmd: ", np.degrees(self.g1_current_cmd[index]))
        #     print("target: ", np.degrees(target))
        #     print("sign_diff: ", sign_diff)
        #     print(sign_diff * step_size)
        #     print()
        
        self.g1_current_cmd[index] = current_cmd
        return current_cmd

    
        
    def enableTeleop(self):
        while not self.shutting_down:
            if self.calibrated:
                if self.teleop_enabled:
                    input("Teleop is running, press [ENTER] to stop or [Ctrl+C] to quit...")
                    print()
                    self.teleop_enabled = False
                    if self.upperbody: 
                        pbar = tqdm(total=100, ncols=60, desc = "Stopping teleop", bar_format="{l_bar}{bar}")
                        prev=0
                        while self.current_arm_sdk > 0.0 and not self.shutting_down:
                            current = int(100 * (1-self.current_arm_sdk))
                            pbar.update(abs(prev-current))
                            prev = current
                            time.sleep(0.1)
                        pbar.update(abs(prev-100))
                        pbar.close()
                        print()
                        print()
                
                else:
                    input("Teleop is not running, press [ENTER] to start or [Ctrl+C] to quit...")
                    print()
                    self.teleop_enabled = True
                    if self.upperbody:
                        pbar = tqdm(total=100, ncols=60, desc = "Starting teleop", bar_format="{l_bar}{bar}")
                        prev=0  
                        while self.current_arm_sdk < 1.0 and not self.shutting_down:
                            current = int(100 * self.current_arm_sdk)
                            pbar.update(abs(prev-current))
                            prev = current
                            time.sleep(0.1)
                        pbar.update(abs(prev-100))
                        pbar.close()
                        print()
                        print()
            time.sleep(0.02)

    def publishAngles(self):
        while not self.exit_teleop:
            if self.current_cmd_initialized or self.upperbody:

                if self.current_cmd_initialized:
                    self.destroy_subscription(self.current_angles_subscription)

                msg = LowCmd()
                # msg.mode_pr = 0
                # msg.mode_machine = 4

                if self.upperbody:
                    if not self.teleop_enabled: #and all(abs(angle) < np.radians(2) for angle in self.g1_current_joint_angles.values()):
                        self.current_arm_sdk = max(self.current_arm_sdk-self.arm_sdk_step, 0.0)
                        msg.motor_cmd[G1JointID.NotUsedJoint].q = self.current_arm_sdk
                    elif self.teleop_enabled:
                        self.current_arm_sdk = min(self.current_arm_sdk+self.arm_sdk_step, 1.0)            
                        msg.motor_cmd[G1JointID.NotUsedJoint].q = self.current_arm_sdk 
                
                # print(self.current_arm_sdk)

                # if not self.teleop_enabled or self.current_arm_sdk == 1.0:
                for i in self.robot_joints:
                    # msg.motor_cmd[i].mode = 1
                    msg.motor_cmd[i].q = self.set_angle(i)
                    msg.motor_cmd[i].dq = 0.0
                    msg.motor_cmd[i].kp = self.kp
                    msg.motor_cmd[i].kd = self.kd
                    msg.motor_cmd[i].tau = 0.0

                msg.crc = self.crc.Crc(msg)
                self.lowcmd_publisher.publish(msg)

            time.sleep(1/self.publish_frequency)

    def current_angles_callback(self, msg):
        for id in self.g1_current_joint_angles.keys():
            self.g1_current_joint_angles[id] = msg.motor_state[id].q
        if not self.current_cmd_initialized:
            self.g1_current_cmd = dict(self.g1_current_joint_angles)
            self.current_cmd_initialized = True
        # if not self.teleop_enabled:
        #     self.g1_current_cmd = dict(self.g1_current_joint_angles)
        # print(np.degrees(self.g1_current_joint_angles[G1JointID.LeftShoulderRoll]))
        # print(np.degrees(self.g1_target_joint_angles[G1JointID.LeftShoulderRoll]))
        # print()

    def gmr(self):
        while not self.exit_teleop:
            t = time.time()
            if (self.upperbody or self.teleop_enabled) and self.calibrated:

                for link in self.human_data.keys():
                    if link == self.root:
                        continue
                
                    self.sensor_orientations_global[link] = self.sensor_orientations_local[self.root].inv() * self.calib_orientation[link] * self.sensor_orientations_local[link]
                    self.human_data[link][1] = self.sensor_orientations_global[link].as_quat(scalar_first=True)

                positions = compute_sensor_forward_kinematics(
                    self.sensor_orientations_global,
                    self.human_data[self.root][0],
                    self.link_base_mapping,
                    self.offset_position
                )

                for link, pos in positions.items():
                    self.human_data[link][0] = pos
            
                # pos = [round(p,3) for p in self.human_data["HandLeft"][0]]
                # print(len(self.human_data.items()))

                qpos = self.retargeter.retarget(self.human_data)
                # print(np.degrees(qpos[G1JointID.LeftShoulderRoll+7]))
                
                # print(len(qpos))
                # print()

                for id in self.g1_target_joint_angles.keys():
                    self.g1_target_joint_angles[id] = qpos[id+7]
            else:
                for id in self.g1_target_joint_angles.keys():
                    self.g1_target_joint_angles[id] = 0.0
                        
            time.sleep(0.0)
        

    def listener_callback(self, msg):
        try:
            # Extract quaternions from the two IMUs
            for reading in msg.readings:

                quat = [reading.orientation.x,
                        reading.orientation.y,
                        reading.orientation.z,
                        reading.orientation.w]
                
                if reading.id in self.name_id_mapping.values():
                    self.sensor_orientations_local[self.id_name_mapping[reading.id]] = R.from_quat(quat)

            if any(v is None for v in self.sensor_orientations_local.values()):
                return

            # Calibration step
            with self.calib_lock:
                if not self.calibrated:
                    self.calibrate()
                    return

        except Exception as e:
            _,_,tb = sys.exc_info()
            line = tb.tb_lineno
            print("Error: ", e)
            print("Line: ", line)

    def cleanup(self):
        self.shutting_down = True
        self.teleop_enabled = False
        if self.upperbody:
            pbar = tqdm(total=100, ncols=60, desc = "Stopping teleop", bar_format="{l_bar}{bar}")
            prev=0
            while self.current_arm_sdk > 0.0:
                current = int(100 * (1-self.current_arm_sdk))
                pbar.update(abs(prev-current))
                prev = current
                time.sleep(0.1)
            pbar.update(abs(prev-100))
            pbar.close()
            print()

        
        
        # while not all(abs(angle) < np.radians(2) for angle in self.g1_current_joint_angles.values()):
        #     time.sleep(0.01)
        #     continue
        if not self.upperbody:
            time.sleep(3)

        self.exit_teleop = True

def sigint_handler(signum, frame):
    global node
    node.shutdown_requested = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', 
        choices = ["whole-body", "upper-body"],
        default = "upper-body",
    )

    args = parser.parse_args()
    global node

    rclpy.init()

    node = G1TeleopNode(args.mode)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)

            if node.shutdown_requested and not node.shutting_down:
                shutdown_thread = threading.Thread(target=node.cleanup)
                shutdown_thread.start()

            if (args.mode == "whole-body" or node.current_arm_sdk == 0.0) and node.exit_teleop:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown


if __name__ == '__main__':
    main()
