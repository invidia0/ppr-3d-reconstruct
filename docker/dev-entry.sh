mkdir -p ros2_ws/docker

cat > ros2_ws/docker/dev-entry.sh << 'EOF'  
#!/usr/bin/env bash
set -e

# ROS env
source /opt/ros/jazzy/setup.bash

# Workspace
cd /workspaces/ros2_ws

# Build, but don't kill container if it fails
colcon build --symlink-install || true

# Stay alive with an interactive shell
exec bash
EOF

chmod +x ros2_ws/docker/dev-entry.sh
