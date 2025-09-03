from setuptools import setup, find_packages

package_name = "sam2_realtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools"
    ],
    zip_safe=True,
    maintainer="Rodrigo Serra",
    maintainer_email="rodrigo.serra@tecnico.ulisboa.pt",
    description="ROS 2 LifecycleNode wrapper for real-time SAM2 tracking",
    license="GPL-3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sam2_realtime_node = sam2_realtime.sam2_realtime_node:main",
            "bbox_prompt_node = sam2_realtime.bbox_prompt_node:main",
            "yolo_prompt_node = sam2_realtime.yolo_prompt_node:main",
            "yolo_mask_prompt_node = sam2_realtime.yolo_mask_prompt_node:main",
            "mask2pcl = sam2_realtime.mask2pcl:main",
            "track_node_2 = sam2_realtime.track_node_2:main",
            "debug_node = sam2_realtime.debug_node:main",
        ],
    },
)
