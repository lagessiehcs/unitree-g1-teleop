#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import argparse
from unitree_hg.msg import LowCmd, LowState

from utils.crc import CRC
import os
import time

class ReplayNode(Node):
    def __init__(self, file_path, frequency):
        super().__init__('lowcmd_replay_node')
        topic = '/arm_sdk'
        with open(file_path, "r") as f:
            first_line = f.readline()  # read only the first line
            data = json.loads(first_line)  # parse it as JSON
            if data["upperbody"] == False:
                topic = '/lowcmd'

        self.file_path = file_path
        self.frequency = frequency
        self.pub = self.create_publisher(LowCmd, topic, 10)
        self.crc = CRC()

        self.get_logger().info(f"Reading commands from: {self.file_path}")

    def run(self):
        if not os.path.exists(self.file_path):
            self.get_logger().error(f"File not found: {self.file_path}")
            return


        with open(self.file_path, "r") as f:
            next(f)
            for line in f:
                if not rclpy.ok():
                    break

                data = json.loads(line)
                msg = LowCmd()

                # arm sdk
                msg.motor_cmd[29].q = data["arm_sdk"]

                # joints
                for joint_id, v in data["joints"].items():
                    i = int(joint_id)
                    msg.motor_cmd[i].q = v["q"]
                    msg.motor_cmd[i].dq = v["dq"]
                    msg.motor_cmd[i].kp = v["kp"]
                    msg.motor_cmd[i].kd = v["kd"]
                    msg.motor_cmd[i].tau = v["tau"]

                # CRC
                msg.crc = self.crc.Crc(msg)


                self.pub.publish(msg)
                time.sleep(1 / self.frequency)



def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to commands file")
    parser.add_argument("--freq", type=int, default=500, help="Publish frequency")
    args = parser.parse_args()

    node = ReplayNode(args.path, args.freq)

    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()