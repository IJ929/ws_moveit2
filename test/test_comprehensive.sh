#!/bin/bash

echo "=== Testing ROS 2 Humble Installation ==="

# Test ROS 2 environment
echo "1. Testing ROS 2 environment..."
if command -v ros2 &> /dev/null; then
    echo "✓ ROS 2 command found"
    ros2 --version
else
    echo "✗ ROS 2 command not found"
    exit 1
fi

# Test ROS 2 packages
echo -e "\n2. Checking ROS 2 packages..."
PACKAGE_COUNT=$(ros2 pkg list | wc -l)
echo "✓ Found $PACKAGE_COUNT ROS 2 packages"

# Test MoveIt 2 packages
echo -e "\n3. Checking MoveIt 2 packages..."
MOVEIT_PACKAGES=$(ros2 pkg list | grep moveit | wc -l)
if [ $MOVEIT_PACKAGES -gt 0 ]; then
    echo "✓ Found $MOVEIT_PACKAGES MoveIt 2 packages"
    echo "Available MoveIt packages:"
    ros2 pkg list | grep moveit | head -10
else
    echo "✗ No MoveIt 2 packages found"
fi

# Test workspace sourcing
echo -e "\n4. Testing workspace sourcing..."
if [ -n "$AMENT_PREFIX_PATH" ]; then
    echo "✓ AMENT_PREFIX_PATH is set:"
    echo "  $AMENT_PREFIX_PATH"
else
    echo "✗ AMENT_PREFIX_PATH not set"
fi

# Test ROS domain
echo -e "\n5. Testing ROS domain..."
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-'not set (default: 0)'}"

# Test basic ROS 2 functionality
echo -e "\n6. Testing ROS 2 daemon..."
ros2 daemon status

echo -e "\n=== Installation Test Complete ==="