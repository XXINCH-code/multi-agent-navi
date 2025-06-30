#!/usr/bin/env python3
"""
ROS Node for Autonomous Navigation of a Single TurtleBot Using ArUco Markers

This script navigates a TurtleBot along a predefined path by reading waypoints from a YAML file.
It uses ArUco markers for localization and computes the required transformations to map simulation
coordinates to real-world coordinates.

Date: 20 November 2024

Requirements:
- ROS (Robot Operating System)
- OpenCV with ArUco module
- PyYAML
"""

import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
import yaml
import cv2
import threading
from collections import deque
import numpy as np


# 设置队列大小
QUEUE_SIZE = 10

# 初始化队列
pose_queue = {
    "id101": deque(maxlen=QUEUE_SIZE),  # 用于存储id101的Pose数据
    "id100": deque(maxlen=QUEUE_SIZE),  # 用于存储id100的Pose数据
}

# 过滤函数
def median_filter(queue):
    """
    对队列中的Pose数据进行中值滤波
    """
    if len(queue) == QUEUE_SIZE:
        # 提取队列中的x, y, z坐标，假设PoseStamped中的位置在pose.position
        positions = np.array([[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in queue])
        filtered_position = np.median(positions, axis=0)
        return filtered_position
    return None

# 回调函数处理不同的ArUco ID
def pose_callback(aruco_id, msg):
    """
    根据aruco_id存储Pose信息并进行中值滤波
    """
    # 将当前的Pose信息加入对应的队列
    pose_queue[aruco_id].append(msg)

    # 执行中值滤波
    filtered_position = median_filter(pose_queue[aruco_id])
    
    if filtered_position is not None:
        rospy.loginfo(f"Filtered Position for {aruco_id}: {filtered_position}")




def convert_sim_to_real_pose(x, y, matrix):
    """
    Converts simulation coordinates to real-world coordinates using a perspective transformation matrix.

    Parameters:
    - x (float): X-coordinate in simulation.
    - y (float): Y-coordinate in simulation.
    - matrix (np.ndarray): 3x3 perspective transformation matrix.

    Returns:
    - Tuple[float, float]: Transformed X and Y coordinates in real-world.
    """
    # Create a homogeneous coordinate for the point
    point = np.array([x, y, 1])

    # Apply the perspective transformation
    transformed_point = np.dot(matrix, point)

    # Normalize to get the actual coordinates
    transformed_point = transformed_point / transformed_point[2]

    return transformed_point[0], transformed_point[1]

def check_goal_reached(current_pose, goal_x, goal_y, tolerance):
    """
    Checks if the robot has reached the goal position within a specified tolerance.

    Parameters:
    - current_pose (PoseStamped): Current pose of the robot.
    - goal_x (float): Goal X-coordinate.
    - goal_y (float): Goal Y-coordinate.
    - tolerance (float): Acceptable distance from the goal to consider it reached.

    Returns:
    - bool: True if goal is reached, False otherwise.
    """
    # Get current position
    current_x = current_pose.pose.position.x
    current_y = current_pose.pose.position.y

    # Check if within tolerance
    if (abs(current_x - goal_x) <= tolerance and abs(current_y - goal_y) <= tolerance):
        return True
    else:
        return False



def navigation(turtlebot_name, aruco_id, goal_list):
    """
    Navigates the TurtleBot through a list of waypoints.

    Parameters:
    - turtlebot_name (str): Name of the TurtleBot.
    - aruco_id (str): ArUco marker ID used for localization.
    - goal_list (List[Tuple[float, float]]): List of (X, Y) coordinates as waypoints.
    """
    current_position_idx = 0  # Index of the current waypoint

    # Publisher to send velocity commands to the robot
    cmd_pub = rospy.Publisher(f'/{turtlebot_name}/cmd_vel', Twist, queue_size=1)

    # Wait for the initial pose message from the ArUco marker
    init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped)

    # Initialize Twist message for velocity commands
    twist = Twist()

    # Loop until all waypoints are reached or ROS is shut down
    while current_position_idx < len(goal_list) and not rospy.is_shutdown():
        # Get current goal coordinates
        goal_x, goal_y = goal_list[current_position_idx]

        if current_position_idx == 13 and aruco_id == "id101":
            rospy.sleep(4)
        # Check if the goal has been reached
        if check_goal_reached(init_pose, goal_x, goal_y, tolerance=0.2):
            rospy.loginfo(f"Waypoint {current_position_idx + 1} reached: Moving to next waypoint.")
            current_position_idx += 1  # Move to the next waypoint

            # If all waypoints are reached, exit the loop
            if current_position_idx >= len(goal_list):
                rospy.loginfo("All waypoints have been reached.")
                break


        # Update the current pose
        init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped)

        # Extract the current orientation in radians from quaternion
        orientation_q = init_pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        current_orientation = yaw  # Current heading of the robot

        # Calculate the difference between the goal and current position
        dx = goal_x - init_pose.pose.position.x
        dy = goal_y - init_pose.pose.position.y
        distance = math.hypot(dx, dy)  # Euclidean distance to the goal
        goal_direction = math.atan2(dy, dx)  # Angle to the goal

        # Normalize angles to range [0, 2π)
        current_orientation = (current_orientation + 2 * math.pi) % (2 * math.pi)
        goal_direction = (goal_direction + 2 * math.pi) % (2 * math.pi)

        # Compute the smallest angle difference
        theta = goal_direction - current_orientation

        # Adjust theta to be within [-π, π]
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta < -math.pi:
            theta += 2 * math.pi

        # Log debug information
        rospy.logdebug(f"Current Position: ({init_pose.pose.position.x:.2f}, {init_pose.pose.position.y:.2f})")
        rospy.logdebug(f"Goal Position: ({goal_x:.2f}, {goal_y:.2f})")
        rospy.logdebug(f"Current Orientation: {current_orientation:.2f} rad")
        rospy.logdebug(f"Goal Direction: {goal_direction:.2f} rad")
        rospy.logdebug(f"Theta (Angle to Goal): {theta:.2f} rad")
        rospy.logdebug(f"Distance to Goal: {distance:.2f} meters")

        # Control parameters (adjust these as needed)
        k_linear = 2    # Linear speed gain
        k_angular = 4.0   # Angular speed gain

        # Compute control commands
        linear_velocity = k_linear * distance * math.cos(theta)  # Move forward towards the goal
        angular_velocity = -k_angular * theta  # Rotate towards the goal direction

        # Limit maximum speeds if necessary
        max_linear_speed = 0.17  # meters per second
        max_angular_speed = 2.0  # radians per second

        if angular_velocity>0.7:
            linear_velocity = 0

        linear_velocity = max(-max_linear_speed, min(max_linear_speed, linear_velocity))
        angular_velocity = max(-max_angular_speed, min(max_angular_speed, angular_velocity))

        # Set Twist message
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity

        # Publish the velocity commands
        cmd_pub.publish(twist)

        # Sleep to maintain the loop rate
        rospy.sleep(0.1)  # Adjust the sleep duration as needed

def get_transformation_matrix(aruco_markers):
    """
    Detects corner ArUco markers and calculates the perspective transformation matrix.

    Parameters:
    - aruco_markers (List[str]): List of ArUco marker IDs used for the transformation.

    Returns:
    - np.ndarray: 3x3 perspective transformation matrix.
    """
    # Dictionary to store the poses of the ArUco markers
    marker_poses = {}

    # Wait for ArUco marker poses to define transformation between simulation and real-world coordinates
    for marker_id in aruco_markers:
        try:
            # Wait for the pose of each ArUco marker
            pose = rospy.wait_for_message(f'/{marker_id}/aruco_single/pose', PoseStamped, timeout=5)
            marker_poses[marker_id] = (pose.pose.position.x, pose.pose.position.y)
            rospy.loginfo(f"Received pose for marker {marker_id}: x={pose.pose.position.x}, y={pose.pose.position.y}")
        except rospy.ROSException:
            rospy.logerr(f"Timeout while waiting for pose of marker {marker_id}")
            raise

    # Define real-world and simulation points for the perspective transformation
    real_points = np.float32([
        marker_poses['id503'],  # Bottom-left corner in real world
        marker_poses['id502'],  # Bottom-right corner in real world
        marker_poses['id500'],  # Top-left corner in real world
        marker_poses['id501']   # Top-right corner in real world
    ])

    sim_points = np.float32([
        [-1, -1],     # Bottom-left corner in simulation
        [10, -1],    # Bottom-right corner in simulation
        [-1, 10],    # Top-left corner in simulation
        [10, 10]    # Top-right corner in simulation
    ])

    # sim_points = np.float32([
    #     [0, 0],     # Bottom-left corner in simulation
    #     [10, 0],    # Bottom-right corner in simulation
    #     [0, 10],    # Top-left corner in simulation
    #     [10, 10]    # Top-right corner in simulation
    # ])

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(sim_points, real_points)

    rospy.loginfo("Perspective transformation matrix calculated successfully.")

    return matrix

def read_and_transform_waypoints(file_path, matrix):
    """
    Reads waypoints from a YAML file and transforms them from simulation to real-world coordinates.

    Parameters:
    - file_path (str): Path to the YAML file containing the schedule.
    - matrix (np.ndarray): Perspective transformation matrix.

    Returns:
    - List[Tuple[float, float]]: List of transformed waypoints.
    """
    # Read the schedule from the YAML file
    def read_yaml_file(file_path):
        """
        Reads the schedule from a YAML file.

        Parameters:
        - file_path (str): Path to the YAML file.

        Returns:
        - dict: Dictionary containing the schedule data.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data['schedule']  # Returns a dictionary of steps

    try:
        # Load schedule data from YAML file
        schedule_data = read_yaml_file(file_path)
    except Exception as e:
        rospy.logerr(f"Failed to read schedule YAML file: {e}")
        raise

    coordinates = [[],[]]  # List to store transformed waypoints

    # Process waypoints for each agent
    for agent_id, steps in schedule_data.items():
        rospy.loginfo(f"Processing agent {agent_id}")

        for step in steps:
            # Simulation coordinates
            sim_x = step['x']
            sim_y = step['y']

            # Transform simulation coordinates to real-world coordinates
            real_x, real_y = convert_sim_to_real_pose(sim_x, sim_y, matrix)

            rospy.loginfo(f"Transformed simulation coordinates ({sim_x}, {sim_y}) to real-world coordinates ({real_x:.2f}, {real_y:.2f})")

            # Append the transformed coordinates to the list
            coordinates[agent_id-1].append((real_x, real_y))

        # break  # Remove this if you want to process multiple agents

    return coordinates

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_closest_index(point1, coord):
    # 查找距离 (0, 3) 最近的点的索引
    min_distance = float('inf')  # 初始化最小距离为无穷大
    closest_index = -1  # 初始化索引为 -1，表示未找到

    for idx, point in enumerate(coord[2:]):
        distance = euclidean_distance(point1, point)
        if distance < min_distance:
            min_distance = distance
            closest_index = idx

    return closest_index

def main():
    """
    Main function to initialize the ROS node and start the navigation process.
    """
    rospy.init_node('goal_pose')

    # List of ArUco marker IDs used for the transformation
    aruco_markers = ['id500', 'id501', 'id502', 'id503']

    try:
        # Get the transformation matrix using the corner detection function
        matrix = get_transformation_matrix(aruco_markers)
    except Exception as e:
        rospy.logerr(f"Failed to get transformation matrix: {e}")
        return

    try:
        # Read and transform waypoints from the YAML file
        coordinates = read_and_transform_waypoints("./cbs_output copy.yaml", matrix)
    except Exception as e:
        rospy.logerr(f"Failed to read and transform waypoints: {e}")
        return
    
    #listener()
    
    #right
    # Start navigation with the first agent's waypoints
    turtlebot1_name = "tb3_0"  # Name of your TurtleBot
    aruco1_id = "id101"          # ArUco marker ID for localization

    #left
    # Start navigation with the first agent's waypoints
    turtlebot2_name = "tb3_1"  # Name of your TurtleBot
    aruco2_id = "id100"          # ArUco marker ID for localization

    # Begin the navigation process

    pose102 = rospy.wait_for_message(f'/id102/aruco_single/pose', PoseStamped, timeout=5)
    pose103 = rospy.wait_for_message(f'/id103/aruco_single/pose', PoseStamped, timeout=5)

    index1 = find_closest_index((pose102.pose.position.x,pose102.pose.position.y),coordinates[0])
    index2 = find_closest_index((pose103.pose.position.x,pose103.pose.position.y),coordinates[1])

    print(index1,index2)
    rospy.sleep(3)

    # coordinates[0].insert(12,(pose102.pose.position.x,pose102.pose.position.y))
    # coordinates[1].insert(7,(pose103.pose.position.x,pose103.pose.position.y))

    t1 = threading.Thread(target=navigation, args=(turtlebot1_name, aruco1_id, coordinates[0]))
    t2 = threading.Thread(target=navigation, args=(turtlebot2_name, aruco2_id, coordinates[1]))

    t1.start()
    t2.start()

    # 等待两个完成
    t1.join()
    t2.join()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


#tow car should be sync when running by timestap.