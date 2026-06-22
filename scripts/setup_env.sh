#!/usr/bin/env bash
# 开发板环境配置脚本（整理自手册第 3 章「准备环境」）。
# ⚠️ 仅在 Atlas 200I DK A2 开发板（ARM/Ubuntu）上执行，不要在开发 PC 上运行。
set -e

echo "==> 1. 关闭 conda（不使用机械臂时请注释 ~/.bashrc 中的 conda deactivate）"
# 在 ~/.bashrc 末尾追加：conda deactivate

echo "==> 2. 安装 ROS2 Humble + colcon + pip"
apt install -y software-properties-common
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt update
apt install -y libegl-mesa0 ros-humble-desktop python3-colcon-common-extensions pip

echo "==> 3. 安装 ais_bench / aclruntime（昇腾推理）"
pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend'
pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench'

echo "==> 4. 安装其他 Python 依赖"
pip3 install -r requirements.txt
dpkg -i libconsole-bridge0.4_0.4.4+dfsg-1build1_arm64.deb || true
dpkg -i liburdfdom-world_1.0.0-2ubuntu0.1_arm64.deb || true

echo "==> 5. 配置 LD_LIBRARY_PATH 并复制运动学库"
# 在 ~/.bashrc 末尾追加：
#   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/lib/aarch64-linux-gpu:/usr/local/lib
cp ./libdofbot_kinemarics.so /usr/lib

echo "==> 6. 编译 orocos_kdl"
cd orocos_kdl && mkdir -p build && cd build && cmake .. && make -j4 && make install && cd ../..

echo "==> 7. 安装机械臂驱动"
cd 0.py_install && python3 setup.py install && cd ..

echo "==> 8. 编译 ROS2 工作空间"
source setenv.sh
cd ros2_ws && colcon build

echo "==> 环境配置完成。后续每开新终端都需 source setenv.sh"
