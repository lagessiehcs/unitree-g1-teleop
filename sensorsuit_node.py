"""SensorSuit driver node."""

import asyncio
import logging
import os
import threading

import rclpy
from rclpy.node import Node

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

from google.protobuf.internal import decoder

# pylint: disable=no-name-in-module
from sensorsuit_driver.external_communication_pb2 import SensorData
from sensorsuit_msgs.msg import ImuReadings, ImuReading
from geometry_msgs.msg import Quaternion

# Get from environment and use 127.0.0.1 if not set
SENSORSUIT_SERVER_ADDRESS = os.getenv(
    'SENSORSUIT_SERVER_ADDRESS', "sensorsuit-puppet-quan.local:5000")


class SensorSuitDriver(Node):
    """SensorSuit driver node."""

    def __init__(self):
        super().__init__('sensor_suit_driver')
        self.publisher = self.create_publisher(
            ImuReadings, 'sensorsuit/imus', 10)
        self.timer = self.create_timer(
            0.001, self.publish_timer_callback)  # 100 Hz
        self.latest_data = None

    def receive_data(self, data_list):
        """This method will be called by the WebRTC callback with new data."""
        self.latest_data = data_list

    def publish_timer_callback(self):
        """Publish IMU data or dummy message at 100 Hz."""
        msg = ImuReadings()
        msg.readings = []

        if self.latest_data is not None:
            for item in self.latest_data:
                imu = ImuReading()
                imu.id = int(item['id']) if isinstance(item, dict) else item.id
                imu.status = int(item['status']) if isinstance(
                    item, dict) else item.status
                q_data = item['quaternion'] if isinstance(
                    item, dict) else item.quaternion
                q = Quaternion(x=q_data[0], y=q_data[1],
                               z=q_data[2], w=q_data[3])
                imu.orientation = q
                msg.readings.append(imu)

            self.latest_data = None  # Reset after processing

        self.publisher.publish(msg)


async def run_webrtc(driver_node: SensorSuitDriver):
    """Async function simulating WebRTC data reception."""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f'http://{SENSORSUIT_SERVER_ADDRESS}/v1/controller/sensordata/offer'
            ) as resp:
                offer = await resp.json()
    except aiohttp.ClientConnectorError:
        logging.error(
            "Could not connect to the sensor suit server at %s.",
            SENSORSUIT_SERVER_ADDRESS)
        # self.update_data(None)
        return

    logging.info("Received offer data: %s", offer)

    pc = RTCPeerConnection()

    def on_message(message):
        logging.debug("Received message: %s", message)
        # pylint: disable=protected-access
        length, new_position = decoder._DecodeVarint32(message, 0)
        sensor_data = SensorData()
        sensor_data.ParseFromString(
            message[new_position: new_position + length])
        # Map sensor data to IMU readings
        imu_readings = []
        for imu_info in sensor_data.imu_data.imu_infos:
            imu_reading = {
                'id': imu_info.sensor_id,
                'status': imu_info.status_code,
                'quaternion': [imu_info.quaternion.x,
                               imu_info.quaternion.y,
                               imu_info.quaternion.z,
                               imu_info.quaternion.w
                               ]
            }
            imu_readings.append(imu_reading)
        driver_node.receive_data(imu_readings)

    @pc.on("datachannel")
    def on_datachannel(channel):
        """ Callback when the data channel is created. """
        logging.debug("Data channel created: %s", channel.label)
        channel.on("message", on_message)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        """ Callback when the ICE connection state changes."""
        logging.debug("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState in ('failed', 'closed'):
            logging.info("ICE connection state is %s", pc.iceConnectionState)

    await pc.setRemoteDescription(RTCSessionDescription(offer['sdp'], offer['type']))

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    async with aiohttp.ClientSession() as session:
        await session.post(
            f'http://{SENSORSUIT_SERVER_ADDRESS}/v1/controller/sensordata/accept',
            json={
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            })
    logging.info("Sent answer. Connection should establish now.")

    while True:
        # Simulate incoming async WebRTC data
        await asyncio.sleep(0.05)  # Simulate 20 Hz data rate


def main(args=None):
    """Main function for the SensorSuit driver node."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting SensorSuit driver node")
    rclpy.init(args=args)
    node = SensorSuitDriver()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        asyncio.run(run_webrtc(node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
