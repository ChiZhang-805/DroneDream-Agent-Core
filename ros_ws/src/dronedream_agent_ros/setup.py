from setuptools import find_packages, setup

package_name = "dronedream_agent_ros"

# 独立安装当前 ROS 包及其纯逻辑/I/O 模块；不依赖桌面前端进程或旧工作树的 PYTHONPATH。
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="DroneDream",
    maintainer_email="engineering@dronedream.local",
    description="ROS 2 runtime nodes for the DroneDream flight-agent core",
    license="Proprietary",
    entry_points={
        # 三个入口各自负责真实观测、动作服务及安全中止桥接，不能用探针替代产品入口。
        "console_scripts": [
            "gazebo_pose_observer = dronedream_agent_ros.gazebo_pose_observer:main",
            "domain_action_server = dronedream_agent_ros.domain_action_server:main",
            "safety_event_guard = dronedream_agent_ros.safety_event_guard:main",
        ],
    },
)
