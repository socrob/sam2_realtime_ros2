from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.easy_install import ScriptWriter


class DevelopWithEnvPython(develop):
    """
    Keep the `#!/usr/bin/env python3` shebang in `colcon build --symlink-install`.

    Symlink installs run `setup.py develop`, which ignores `[build_scripts] executable`
    in setup.cfg and writes the build's interpreter (`/usr/bin/python3`). The nodes
    would then miss the packages of an activated virtual environment.
    """

    def install_wrapper_scripts(self, dist):
        if self.exclude_scripts:
            return

        header = ScriptWriter.get_header(executable="/usr/bin/env python3")
        for args in ScriptWriter.best().get_args(dist, header):
            self.write_script(*args)


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
    cmdclass={"develop": DevelopWithEnvPython},
    maintainer="Rodrigo Serra",
    maintainer_email="rodrigo.serra@tecnico.ulisboa.pt",
    description="ROS 2 LifecycleNode wrapper for real-time SAM2 tracking",
    license="GPL-3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sam2_realtime_node = sam2_realtime.sam2_realtime_node:main",
            "mask2pcl = sam2_realtime.mask2pcl:main",
            "track_node = sam2_realtime.track_node:main",
        ],
    },
)
