# Start from the official MoveIt 2 Humble source image.
FROM moveit/moveit2:main-humble-tutorial-source

# Install Gazebo and its ROS 2 integration packages.
# RUN apt-get update && apt-get install -y \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Copy custom bashrc and set it up for root user
COPY .bashrc /root/.bashrc
RUN chmod +x /root/.bashrc

# # Source the ROS 2 Humble setup file for the root user.
# RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc