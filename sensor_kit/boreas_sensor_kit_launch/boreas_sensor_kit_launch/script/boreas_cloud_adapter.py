#!/usr/bin/env python3
"""Publish Boreas measurements in the form the Autoware sensing pipeline uses.

The concatenated lidar cloud is Autoware PointXYZIRC in base_link:
float32 x, y, z, uint8 intensity, uint8 return_type, uint16 channel.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Imu, NavSatFix, PointCloud2, PointField
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class BoreasCloudAdapter(Node):
    def __init__(self):
        super().__init__("boreas_cloud_adapter")
        self.declare_parameter("input_pointcloud", "/boreas/pointcloud")
        self.declare_parameter("output_pointcloud", "lidar/concatenated/pointcloud")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("input_imu", "/boreas/imu")
        self.declare_parameter("output_imu", "imu/imu_data")
        self.declare_parameter("imu_frame", "imu")
        self.declare_parameter("input_gnss", "/boreas/gnss/fix")
        self.declare_parameter("output_gnss", "gnss/ublox/nav_sat_fix")

        self._output_frame = str(self.get_parameter("output_frame").value)
        self._imu_frame = str(self.get_parameter("imu_frame").value)
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cloud_pub = self.create_publisher(
            PointCloud2, str(self.get_parameter("output_pointcloud").value), sensor_qos
        )
        self._imu_pub = self.create_publisher(
            Imu, str(self.get_parameter("output_imu").value), sensor_qos
        )
        self._gnss_pub = self.create_publisher(
            NavSatFix, str(self.get_parameter("output_gnss").value), sensor_qos
        )
        self.create_subscription(
            PointCloud2, str(self.get_parameter("input_pointcloud").value), self._on_cloud, sensor_qos
        )
        self.create_subscription(
            Imu, str(self.get_parameter("input_imu").value), self._on_imu, sensor_qos
        )
        self.create_subscription(
            NavSatFix, str(self.get_parameter("input_gnss").value), self._on_gnss, sensor_qos
        )
        self._warned = False

    def _on_cloud(self, msg: PointCloud2):
        if msg.header.frame_id != self._output_frame:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._output_frame, msg.header.frame_id, rclpy.time.Time()
                )
            except Exception as exc:
                if not self._warned:
                    self.get_logger().warn(f"Waiting for {msg.header.frame_id} -> {self._output_frame}: {exc}")
                    self._warned = True
                return
            msg = do_transform_cloud(msg, transform)
        self._cloud_pub.publish(self._to_point_xyzirc(msg))

    def _to_point_xyzirc(self, msg: PointCloud2) -> PointCloud2:
        """Pack Autoware PointXYZIRC. Intensity is uint8, not float32."""
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        xyz = np.ascontiguousarray(raw[:, 0:12]).view(np.float32).reshape(-1, 3)
        intensity = self._float_field(raw, msg, "intensity", 12)
        channel = self._float_field(raw, msg, "channel", 16)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        intensity = np.clip(np.rint(intensity[finite]), 0, 255).astype(np.uint8)
        channel = np.clip(np.rint(channel[finite]), 0, 65535).astype(np.uint16)
        packed = np.empty(
            xyz.shape[0],
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("intensity", "u1"),
                    ("return_type", "u1"),
                    ("channel", "<u2"),
                ]
            ),
        )
        packed["x"] = xyz[:, 0]
        packed["y"] = xyz[:, 1]
        packed["z"] = xyz[:, 2]
        packed["intensity"] = intensity
        packed["return_type"] = np.uint8(1)
        packed["channel"] = channel

        out = PointCloud2()
        out.header = msg.header
        out.header.frame_id = self._output_frame
        out.height = 1
        out.width = packed.shape[0]
        out.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
            PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
            PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 16
        out.row_step = 16 * packed.shape[0]
        out.is_dense = True
        out.data = packed.tobytes()
        return out

    @staticmethod
    def _float_field(raw, msg, name, default_offset):
        offset = default_offset
        for field in msg.fields:
            if field.name == name:
                offset = field.offset
                break
        return np.ascontiguousarray(raw[:, offset : offset + 4]).view(np.float32).reshape(-1)

    def _on_imu(self, msg: Imu):
        msg.header.frame_id = self._imu_frame
        self._imu_pub.publish(msg)

    def _on_gnss(self, msg: NavSatFix):
        self._gnss_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BoreasCloudAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
