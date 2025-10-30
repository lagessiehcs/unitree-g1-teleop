import rclpy
from rclpy.node import Node


from sensorsuit_msgs.msg import ImuReadings
from scipy.spatial.transform import Rotation as R
from unitree_hg.msg import LowCmd, LowState

from utils.crc import CRC
import numpy as np
import threading
import time

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

G1_ACTIVE_JOINTS_23DOF = [
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

SENSOR_ID_PAIRS = [
    [SensorID.LowerBack, SensorID.ThighLeft],           # Left hip
    [SensorID.ThighLeft, SensorID.ShankLeft],           # Left knee
    [SensorID.ShankLeft, SensorID.FootLeft],            # Left ankle
    [SensorID.LowerBack, SensorID.ThighRight],          # Right hip
    [SensorID.ThighRight, SensorID.ShankRight],         # Right knee
    [SensorID.ShankRight, SensorID.FootRight],          # Right ankle
    [SensorID.UpperBack, SensorID.LowerBack],           # Waist
    [SensorID.ShoulderLeft, SensorID.UpperArmLeft],     # Left shoulder
    [SensorID.UpperArmLeft, SensorID.ForearmLeft],      # Left elbow
    [SensorID.ForearmLeft, SensorID.HandLeft],          # Left wrist
    [SensorID.ShoulderRight, SensorID.UpperArmRight],   # Right shoulder
    [SensorID.UpperArmRight, SensorID.ForearmRight],    # Right elbow
    [SensorID.ForearmRight, SensorID.HandRight],        # Right wrist
]

G1_JOINT_ID_GROUPS = [
    [G1JointID.LeftHipPitch, G1JointID.LeftHipRoll, G1JointID.LeftHipYaw],                      # Left hip
    [G1JointID.LeftKnee],                                                                       # Left knee
    [G1JointID.LeftAnklePitch, G1JointID.LeftAnkleRoll],                                        # Left ankle
    [G1JointID.RightHipPitch, G1JointID.RightHipRoll, G1JointID.RightHipYaw],                   # Right hip
    [G1JointID.RightKnee],                                                                      # Right knee
    [G1JointID.RightAnklePitch, G1JointID.RightAnkleRoll],                                      # Right ankle
    [G1JointID.WaistYaw],                                                                       # Waist
    [G1JointID.LeftShoulderPitch, G1JointID.LeftShoulderRoll, G1JointID.LeftShoulderYaw],       # Left shoulder
    [G1JointID.LeftElbow],                                                                      # Left elbow
    [G1JointID.LeftWristRoll],                                                                  # Left wrist
    [G1JointID.RightShoulderPitch, G1JointID.RightShoulderRoll, G1JointID.RightShoulderYaw],    # Right shoulder
    [G1JointID.RightElbow],                                                                     # Right elbow
    [G1JointID.RightWristRoll],                                                                 # Right wrist
]

class G1TeleopNode(Node):

    def __init__(self):
        super().__init__('shoulder_kinematics_node')
        self.imus_subscription = self.create_subscription(
            ImuReadings,
            'sensorsuit/imus',
            self.listener_callback,
            10)
        
        self.current_angles_subscription = self.create_subscription(
            LowState,
            'lowstate',
            self.current_angles_callback,
            10)
        
        # PD gains
        self.kp = 50.0
        self.kd = 1.0
        
        self.lowcmd_publisher = self.create_publisher(LowCmd, '/lowcmd', 10)

        self.step_size = {id: np.radians(10.0) for id in G1_ACTIVE_JOINTS_23DOF}
        self.step_size[G1JointID.LeftElbow] = np.radians(5.0)
        self.step_size[G1JointID.RightElbow] = np.radians(5.0)


        # IMU IDs
        self.calib_joint_rotations = [None] * len(SENSOR_ID_PAIRS)
        self.calibrated = False

        self.g1_target_joint_angles = {id: 0.0 for id in G1_ACTIVE_JOINTS_23DOF}
        self.g1_current_joint_angles = {id: 0.0 for id in G1_ACTIVE_JOINTS_23DOF}

        self.crc = CRC()

        self.sensor_orientations = {id: None for id in ACTIV_SENSOR_IDS}

        self.left_elbow_roll = 0.0
        self.left_elbow_yaw = 0.0
        self.right_elbow_roll = 0.0
        self.right_elbow_yaw = 0.0

        self.publishAngles_thread = threading.Thread(target=self.publishAngles)
        self.publishAngles_thread.start()
    

    def calibrate(self, sensor_orientations):
        """Calibrate once using a known pose."""
        for i, sensor_id_pair in enumerate(SENSOR_ID_PAIRS):
                self.calib_joint_rotations[i] = sensor_orientations[sensor_id_pair[0]] * sensor_orientations[sensor_id_pair[1]].inv()

        self.calibrated = True
        self.get_logger().info("Calibration complete!")

    def set_angle(self, current, target, step_size):
        diff = target-current
        sign = 1 if diff >= 0 else -1
        if abs(diff) > np.radians(20):
            return current + sign * step_size
        else:
            return target

    def publishAngles(self):
        while True:
            msg = LowCmd()
            msg.mode_pr = 0
            msg.mode_machine = 4

            for i in G1_ACTIVE_JOINTS_23DOF:
                msg.motor_cmd[i].mode = 1
                # msg.motor_cmd[i].q = self.g1_target_joint_angles[i]
                msg.motor_cmd[i].q = self.set_angle(self.g1_current_joint_angles[i], self.g1_target_joint_angles[i], self.step_size[i])
                if i == G1JointID.LeftShoulderRoll:
                    print(np.degrees(self.g1_current_joint_angles[i]))
                    print(np.degrees(self.g1_target_joint_angles[i]))
                    print(np.degrees(msg.motor_cmd[i].q))
                    print()
                msg.motor_cmd[i].dq = 0.0
                msg.motor_cmd[i].kp = self.kp
                msg.motor_cmd[i].kd = self.kd
                msg.motor_cmd[i].tau = 0.0

            msg.crc = self.crc.Crc(msg)
            self.lowcmd_publisher.publish(msg)
            time.sleep(0.02)

    def current_angles_callback(self, msg):
        for id in self.g1_current_joint_angles.keys():
            self.g1_current_joint_angles[id] = msg.motor_state[id].q
        
        # print(np.degrees(self.g1_current_joint_angles[G1JointID.LeftShoulderRoll]))
        # print(np.degrees(self.g1_target_joint_angles[G1JointID.LeftShoulderRoll]))
        # print()

    def listener_callback(self, msg):

        # Extract quaternions from the two IMUs
        for reading in msg.readings:

            quat = [reading.orientation.x,
                             reading.orientation.y,
                             reading.orientation.z,
                             reading.orientation.w]
            
            self.sensor_orientations[reading.id] = R.from_quat(quat)

        if any(v is None for v in self.sensor_orientations.values()):
            return

        # Calibration step
        if not self.calibrated:
            self.calibrate(self.sensor_orientations)
            return

        for calib_joint_rotation, sensor_id_pair, g1_joint_id_group  in zip(self.calib_joint_rotations, SENSOR_ID_PAIRS, G1_JOINT_ID_GROUPS):
            joint_rotation = self.sensor_orientations[sensor_id_pair[0]].inv() * calib_joint_rotation * self.sensor_orientations[sensor_id_pair[1]]
            rot_matrix = joint_rotation.as_matrix() # Rotation matrix representing the orientation of the sensor 0 w.r.t. the sensor 1
            
            if sensor_id_pair == [SensorID.ShoulderLeft, SensorID.UpperArmLeft]:
                angles = self.pry_shoulder(rot_matrix, "left")

                # print(np.degrees(angles[0]))
                # print(np.degrees(angles[1]))
                # print(np.degrees(angles[2]))
                # print()

                for g1_joint_id, angle in zip(g1_joint_id_group, angles):
                    # if abs(self.g1_target_joint_angles[g1_joint_id] - angle) < np.radians(200):
                    #     self.g1_target_joint_angles[g1_joint_id] = angle
                    self.g1_target_joint_angles[g1_joint_id] = angle
                self.g1_target_joint_angles[G1JointID.LeftShoulderYaw] += self.left_elbow_roll
            
            elif sensor_id_pair == [SensorID.ShoulderRight, SensorID.UpperArmRight]:
                angles = self.pry_shoulder(rot_matrix, "right")

                # print(np.degrees(angles[0]))
                # print(np.degrees(angles[1]))
                # print(np.degrees(angles[2]))
                # print()

                for g1_joint_id, angle in zip(g1_joint_id_group, angles):
                    # if abs(self.g1_target_joint_angles[g1_joint_id] - angle) < np.radians(200):
                    #     self.g1_target_joint_angles[g1_joint_id] = angle
                    self.g1_target_joint_angles[g1_joint_id] = angle
                self.g1_target_joint_angles[G1JointID.RightShoulderYaw] += self.right_elbow_roll
            
            elif sensor_id_pair == [SensorID.UpperArmLeft, SensorID.ForearmLeft]:
                pitch, roll, yaw = self.pr_elbow(rot_matrix, "left")

                # print(np.degrees(pitch))
                # print(np.degrees(roll))
                # print(np.degrees(yaw))
                # print()

                self.g1_target_joint_angles[g1_joint_id_group[0]] = pitch
                self.left_elbow_roll = roll
                self.left_elbow_yaw = yaw

                # print(g1_joint_id_group)
            
            elif sensor_id_pair == [SensorID.UpperArmRight, SensorID.ForearmRight]:
                pitch, roll, yaw = self.pr_elbow(rot_matrix, "right")

                # print(np.degrees(pitch))
                # print(np.degrees(roll))
                # print(np.degrees(yaw))
                # print()

                self.g1_target_joint_angles[g1_joint_id_group[0]] = pitch
                self.right_elbow_roll = roll
                self.right_elbow_yaw = yaw

                # print(g1_joint_id_group)

            elif sensor_id_pair == [SensorID.ForearmLeft, SensorID.HandLeft]:
                pitch, roll, yaw = self.pry_wrist(rot_matrix, "left")

                # print(np.degrees(pitch))
                # print(np.degrees(roll))
                # print()

                self.g1_target_joint_angles[g1_joint_id_group[0]] = roll + self.left_elbow_yaw

                # print(g1_joint_id_group)

            elif sensor_id_pair == [SensorID.ForearmRight, SensorID.HandRight]:
                pitch, roll, yaw = self.pry_wrist(rot_matrix, "right")

                # print(np.degrees(pitch))
                # print(np.degrees(roll))
                # print(np.degrees(yaw))
                # print()

                self.g1_target_joint_angles[g1_joint_id_group[0]] = roll + self.right_elbow_yaw

                # print(g1_joint_id_group)

            elif sensor_id_pair == [SensorID.UpperBack, SensorID.LowerBack]:
                pitch, roll, yaw = self.pry_waist(rot_matrix)

                # print(np.degrees(pitch))
                # print(np.degrees(roll))
                # print(np.degrees(yaw))
                # print()

                self.g1_target_joint_angles[g1_joint_id_group[0]] = roll

                # print(g1_joint_id_group)

            else:
                for g1_joint_id in g1_joint_id_group:
                    self.g1_target_joint_angles[g1_joint_id] = 0.0

    def signed_angle(self, v1, v2, axis, degrees=False):
        v1 = np.array(v1)
        v2 = np.array(v2)
        axis = np.array(axis)
        
        # Normalize vectors
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        axis_u = axis / np.linalg.norm(axis)
        
        # Unsigned angle
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        # Determine sign using cross product
        sign = np.sign(np.dot(np.cross(v1_u, v2_u), axis_u))
        signed_angle = angle * sign
        
        if degrees:
            signed_angle = np.degrees(signed_angle)
        
        return signed_angle
    
    def pry_shoulder(self, rot_matrix, side):
        y_parent = rot_matrix[:,1] # local y-axis represented in shoulder frame 
        # if side == "left":
            # print("y")
            # print(y_parent[0])
            # print(y_parent[1])
            # print(y_parent[2])
            # print()


        # Pitch calculation
        eps = 0.2
        k = 5

        r = np.hypot(y_parent[0], y_parent[2])
        
        w = 1.0 - np.exp(- (r / eps)**k) # Filter for pitch angle near y-axis

        if side == "left":
            if self.g1_target_joint_angles[G1JointID.LeftShoulderRoll] < np.pi/2:
                pitch = -w * np.arctan2(y_parent[0],y_parent[2]) 
            else:
                pitch = w * np.arctan2(y_parent[0],-y_parent[2])

        elif side == "right":
            if self.g1_target_joint_angles[G1JointID.RightShoulderRoll] > -np.pi/2:
                pitch = w * np.arctan2(y_parent[0],y_parent[2]) 
            else:
                pitch = -w * np.arctan2(y_parent[0],-y_parent[2]) 

        # Roll calculation
        roll = np.arcsin(y_parent[1]) if y_parent[2]>=0 or abs(pitch) > np.pi/2 else np.pi-np.arcsin(y_parent[1])
        # print("Roll         ", np.degrees(roll))
        # print("y_parent:   ", y_parent)
        # print("pitch:       ", np.degrees(pitch))
        # print()
        # Yaw calculation
        sign = 1 if y_parent[2]>0 else -1
        theta_x = abs(sign*np.arccos(y_parent[1])) # rotation around x

        # if side == "left":
        #     print("theta_x: ",theta_x)
        
        theta_y = -pitch if side == "left" else pitch # rotation around y
        
        rot = R.from_euler('YX', [theta_y, theta_x], degrees=False)
        
        R_mat_no_yaw = rot.as_matrix()
        z_0 = R_mat_no_yaw[:,2] # without yaw
        z_1 = rot_matrix[:,2] # with yaw

        yaw = self.signed_angle(z_0, z_1, y_parent, degrees=False)
        if roll >= np.radians(135):
            yaw = -yaw
        sign = 1 if yaw >=0 else -1
        
        if side == "right":
            roll = -roll

        return pitch, roll, yaw
    
    def pr_elbow(self, rot_matrix, side):
        y_parent = rot_matrix[:,1] # local y-axis represented in shoulder frame 

        # print(y_parent[0])
        # print(y_parent[1])
        # print(y_parent[2])
        # print()

        # Pitch calculation
        sign = 1 if y_parent[0]>0 else -1
        pitch = sign * np.arccos(y_parent[1]) 

        # Roll calculation
        eps = 0.2
        k = 5
        r = np.hypot(y_parent[0], y_parent[2])
        w = 1.0 - np.exp(- (r / eps)**k) # Filter for pitch angle near y-axis

        if side == "left":
            # print("pitch: ", pitch)
            if pitch > 0:
                unfiltered_roll = np.arctan2(y_parent[2],y_parent[0])
                sign_roll = -1
            else:
                unfiltered_roll = np.arctan2(y_parent[2],-y_parent[0])
                sign_roll = 1
            roll = sign_roll * w * unfiltered_roll

        elif side == "right":
            if pitch <= 0:
                unfiltered_roll = np.arctan2(y_parent[2],-y_parent[0])
                sign_roll = 1
            else:
                unfiltered_roll = np.arctan2(y_parent[2],y_parent[0])
                sign_roll = -1
            roll = sign_roll * w * unfiltered_roll
            

        # Yaw calculation
        sign = 1 if y_parent[0]>0 else -1

        theta_z = abs(sign*np.arccos(y_parent[1])) # rotation around z
        if side == "left":
            theta_y = roll   # rotation around y
        elif side == "right":
            theta_y = roll   # rotation around y

        
        
        rot = R.from_euler('YZ', [theta_y, theta_z], degrees=False)
        
        R_mat_no_yaw = rot.as_matrix()
        z_0 = R_mat_no_yaw[:,2] # without yaw
        z_1 = rot_matrix[:,2] # with yaw

        # if side == "right":
        #     print(z_0)
        #     print(z_1)
        #     print()

        yaw = self.signed_angle(z_0, z_1, y_parent, degrees=False)
        yaw = -yaw
        
        if side =="left":
            pitch = np.pi/2-pitch  
        elif side == "right":
            pitch = np.pi/2+pitch  

        # print(np.degrees(yaw))

        return pitch, roll, yaw
    
    def pry_wrist(self, rot_matrix, side):
       
        r = R.from_matrix(rot_matrix)
        roll,pitch,yaw = r.as_euler('yzx')

        yaw = -yaw
        roll = -roll
        if side == "right":
            pitch = -pitch

        # print(np.degrees(roll))
        # print(np.degrees(pitch))
        # print(np.degrees(yaw))
        # print()
        return pitch, roll, yaw
    
    def pry_waist(self, rot_matrix):
       
        r = R.from_matrix(rot_matrix)
        roll,pitch,yaw = r.as_euler('yzx')

        yaw = -yaw
        roll = -roll
    

        # print(np.degrees(roll))
        # print(np.degrees(pitch))
        # print(np.degrees(yaw))
        # print()
        return pitch, roll, yaw


def main(args=None):
    rclpy.init(args=args)

    node = G1TeleopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()