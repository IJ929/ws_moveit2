# Start from the official MoveIt 2 Humble source image.
FROM moveit/moveit2:main-humble-tutorial-source

# Switch to the root user for system-level commands.
USER root

# This block fully resets the ROS repository configuration.
# 1. Remove the old, conflicting source list from the base image.
# 2. Download the current GPG key.
# 3. Create a new, correct source list file pointing to the new key.
RUN rm -f /etc/apt/sources.list.d/ros2.list /etc/apt/sources.list.d/ros2-latest.list && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# With the key and source list fixed, now update and install your packages.
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy custom bashrc and set it up for root user
COPY .bashrc /root/.bashrc
RUN chmod +x /root/.bashrc
# # Source the ROS 2 Humble setup file for the root user.
# RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc