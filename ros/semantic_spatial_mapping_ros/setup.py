from pathlib import Path

from setuptools import find_namespace_packages, setup


package_name = "semantic_spatial_mapping_ros"
repo_root = Path(__file__).resolve().parents[2]

repo_packages = find_namespace_packages(
    where=str(repo_root),
    include=[
        "geometry*",
        "mapping*",
        "motion*",
        "runtime*",
        "segmentation*",
        "tracking*",
        "world*",
    ],
    exclude=["external_anirudh_vslam*"],
)

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name] + repo_packages,
    package_dir={
        "": str(repo_root),
        package_name: package_name,
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/gazebo.yaml", "config/embedded_oakd.yaml"]),
        (
            f"share/{package_name}/launch",
            ["launch/gazebo_runtime.launch.py", "launch/embedded_runtime.launch.py"],
        ),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="semantic_spatial_mapping",
    maintainer_email="robotics@example.com",
    description="Deployment ROS2 runtime for semantic spatial mapping and visual SLAM.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "semantic_spatial_node = semantic_spatial_mapping_ros.semantic_spatial_node:main",
        ],
    },
)
