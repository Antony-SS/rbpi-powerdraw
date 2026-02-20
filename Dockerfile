# =============================================================
# rbpi-powerdraw – ROS 2 Humble + TFLite YOLO on Raspberry Pi
# =============================================================
# Base: official ROS 2 Humble image (Ubuntu 22.04, arm64-native)
# Runs on Raspberry Pi OS Bookworm via Docker.
# =============================================================
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        libopencv-dev \
        libcap-dev \
        libatlas-base-dev \
        libdrm-dev \
        ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- ROS 2 environment sourced on every shell ----------------
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# ---- Default entrypoint -------------------------------------
# Source ROS 2, then exec whatever command is passed.
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
