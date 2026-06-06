from launch import LaunchDescription
from ros_gz_bridge.actions import RosGzBridge

WORLD = "semantic_spatial_factory_demo"
PREFIX = f"/world/{WORLD}/model/factory_bot/link/camera_link/sensor"

BRIDGES = {
    "rgb_image": {
        "ros_topic_name": "/camera/color/image_raw",
        "gz_topic_name": f"{PREFIX}/rgb_camera/image",
        "ros_type_name": "sensor_msgs/msg/Image",
        "gz_type_name": "gz.msgs.Image",
        "direction": "GZ_TO_ROS",
    },
    "rgb_info": {
        "ros_topic_name": "/camera/color/camera_info",
        "gz_topic_name": f"{PREFIX}/rgb_camera/camera_info",
        "ros_type_name": "sensor_msgs/msg/CameraInfo",
        "gz_type_name": "gz.msgs.CameraInfo",
        "direction": "GZ_TO_ROS",
    },
    "depth_image": {
        "ros_topic_name": "/camera/depth/image_raw",
        "gz_topic_name": f"{PREFIX}/depth_camera/depth_image",
        "ros_type_name": "sensor_msgs/msg/Image",
        "gz_type_name": "gz.msgs.Image",
        "direction": "GZ_TO_ROS",
    },
    "depth_info": {
        "ros_topic_name": "/camera/depth/camera_info",
        "gz_topic_name": f"{PREFIX}/depth_camera/camera_info",
        "ros_type_name": "sensor_msgs/msg/CameraInfo",
        "gz_type_name": "gz.msgs.CameraInfo",
        "direction": "GZ_TO_ROS",
    },
    "odom": {
        "ros_topic_name": "/odom",
        "gz_topic_name": "/model/factory_bot/odometry",
        "ros_type_name": "nav_msgs/msg/Odometry",
        "gz_type_name": "gz.msgs.Odometry",
        "direction": "GZ_TO_ROS",
    },
    "cmd_vel": {
        "ros_topic_name": "/factory_bot/cmd_vel",
        "gz_topic_name": "/model/factory_bot/cmd_vel",
        "ros_type_name": "geometry_msgs/msg/Twist",
        "gz_type_name": "gz.msgs.Twist",
        "direction": "ROS_TO_GZ",
    },
}


def generate_launch_description():
    return LaunchDescription([
        RosGzBridge(
            bridge_name="factory_bot_bridge",
            extra_bridge_params={
                "bridge_names": sorted(BRIDGES.keys()),
                "bridges": BRIDGES,
            },
        )
    ])
