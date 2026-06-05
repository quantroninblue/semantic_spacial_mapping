from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("semantic_spatial_mapping_ros"), "config", "gazebo.yaml"]
    )
    config_path = LaunchConfiguration("config_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=default_config,
                description="Runtime YAML config.",
            ),
            Node(
                package="semantic_spatial_mapping_ros",
                executable="semantic_spatial_node",
                name="semantic_spatial_node",
                output="screen",
                parameters=[{"config_path": config_path}],
            ),
        ]
    )
