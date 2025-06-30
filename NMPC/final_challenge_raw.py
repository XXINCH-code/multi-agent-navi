import pybullet as p
import time
import pybullet_data
import yaml
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import minimize, Bounds

from cbs import cbs
from cbs import cbsfxy


def create_boundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("./final_challenge/assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, -1, 0.5])

def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """
    with open(yaml_file, 'r') as f:
        try:
            env_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e) 
            
    # Create env boundaries
    dimensions = env_params["map"]["dimensions"]
    create_boundaries(dimensions[0], dimensions[1])

    # Create env obstacles
    for obstacle in env_params["map"]["obstacles"]:
        p.loadURDF("./final_challenge/assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])
    return env_params

def create_agents(yaml_file):
    """
    Creates and loads turtlebot agents.

    Returns list of agent IDs and dictionary of agent IDs mapped to each agent's goal.
    """
    agent_box_ids = []
    box_id_to_goal = {}
    agent_name_to_box_id = {}
    with open(yaml_file, 'r') as f:
        try:
            agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
        
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
        agent_box_ids.append(box_id)
        box_id_to_goal[box_id] = agent["goal"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params

def read_cbs_output(file):
    """
        Read file from output.yaml, store path list.

        Args:

        output_yaml_file: output file from cbs.

        Returns:

        schedule: path to goal position for each robot.
    """
    with open(file, 'r') as f:
        try:
            params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return params["schedule"]

def checkPosWithBias(Pos, goal, bias):
    """
        Check if pos is at goal with bias

        Args:

        Pos: Position to be checked, [x, y]

        goal: goal position, [x, y]

        bias: bias allowed

        Returns:

        True if pos is at goal, False otherwise
    """
    if(Pos[0] < goal[0] + bias and Pos[0] > goal[0] - bias and Pos[1] < goal[1] + bias and Pos[1] > goal[1] - bias):
        return True
    else:
        return False

def navigation(agent, goal, schedule):
    """
    Set velocity for robots to follow the path in the schedule using PID control.

    Args:
        agent: ID of the robot agent.
        goal: Target position [x, y].
        schedule: Dictionary with agent IDs as keys and the list of waypoints to the goal as values.

    Returns:
        None
    """
    # Initialize PID parameters
    k1_p, k1_i, k1_d = 20, 0.5, 2  # Distance PID parameters
    k2_p, k2_i, k2_d = 13, 0.3, 1  # Angular PID parameters


    # PID state variables
    distance_error_prev = 0
    theta_error_prev = 0
    distance_error_sum = 0
    theta_error_sum = 0

    # Time initialization
    last_time = time.time()

    # Other configurations
    ideal_path = []  

    basePos = p.getBasePositionAndOrientation(agent)
    index = 0
    dis_th = 0.4  # Distance threshold to consider the waypoint reached

    while not checkPosWithBias(basePos[0], goal, dis_th):
        basePos = p.getBasePositionAndOrientation(agent)
        ideal_path.append((basePos[0][0], basePos[0][1]))
        if index < len(schedule):
            next_wp = [schedule[index]["x"], schedule[index]["y"]]
            if checkPosWithBias(basePos[0], next_wp, dis_th):
                index += 1

        # Stop if the schedule is exhausted
        if index == len(schedule):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break

        # Get current position and orientation
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))

        # Normalize angles to [0, 2π]
        if Orientation < 0:
            Orientation += 2 * math.pi
        if goal_direction < 0:
            goal_direction += 2 * math.pi

        # Calculate angular error (theta)
        theta = goal_direction - Orientation
        if theta < -math.pi:
            theta += 2 * math.pi
        elif theta > math.pi:
            theta -= 2 * math.pi

        # Calculate distance to the next waypoint
        current = [x, y]
        distance = math.dist(current, [schedule[index]["x"], schedule[index]["y"]])

        # PID calculations
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Distance PID
        distance_error = distance
        distance_error_sum += distance_error * dt
        distance_error_delta = (distance_error - distance_error_prev) / dt if dt > 0 else 0
        distance_error_prev = distance_error
        distance_output = (k1_p * distance_error +
                           k1_i * distance_error_sum +
                           k1_d * distance_error_delta)

        # Angular PID
        theta_error = theta
        theta_error_sum += theta_error * dt
        theta_error_delta = (theta_error - theta_error_prev) / dt if dt > 0 else 0
        theta_error_prev = theta_error
        theta_output = (k2_p * theta_error +
                        k2_i * theta_error_sum +
                        k2_d * theta_error_delta)

        # Compute wheel velocities
        rightWheelVelocity = distance_output + theta_output
        leftWheelVelocity = distance_output - theta_output

        
        # Limit linear velocity if angular velocity is high
        if abs(theta_output) > 1:
            distance_output = 0
        
        # Append current position to the path
        if agent == 71:
            x_practical1.append(x)
            y_practical1.append(y)
        elif agent == 72:
            x_practical2.append(x)
            y_practical2.append(y)

        # Apply wheel velocities
        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)

    print(agent, "reached the goal")

# ---------------------------------------------------
# NMPC Controller
# ---------------------------------------------------
def vehicle_dynamics(state, control, dt):
    """
    Update the robot state based on the control input.
    Args:
        state: [x, y, theta]
        control: [v, omega]
        dt: timestep
    Returns:
        new_state: updated [x, y, theta]
    """
    x, y, theta = state
    v, omega = control
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    return np.array([x_new, y_new, theta_new])

def cost_function(u, *args):
    """
    Cost function for NMPC.
    Args:
        u: control inputs (flattened array of [v1, omega1, v2, omega2, ...])
        args: additional arguments including:
              - current state
              - reference trajectory
              - prediction horizon
              - timestep
    Returns:
        total_cost: scalar cost value
    """
    state, ref_trajectory, horizon, dt, weights = args
    total_cost = 0
    u = u.reshape((horizon, 2))  # Reshape the control input
    
    
    for t in range(horizon):
        v, omega = u[t]
        state = vehicle_dynamics(state, [v, omega], dt)
        ref = ref_trajectory[t]
        position_cost = weights['position'] * np.linalg.norm(state[:2] - ref[:2])
        orientation_cost = weights['orientation'] * (state[2] - ref[2])**2
        control_cost = weights['control'] * (v**2 + omega**2)
        total_cost += position_cost + orientation_cost + control_cost

    return total_cost

def nmpc_controller(state, ref_trajectory, horizon, dt):
    """
    Nonlinear Model Predictive Controller for trajectory tracking.
    Args:
        state: current robot state [x, y, theta]
        ref_trajectory: reference trajectory to follow
        horizon: prediction horizon
        dt: timestep
    Returns:
        optimal_control: [v, omega]
    """
    weights = {'position': 30.0, 'orientation': 10.0, 'control': 0.02}
    u0 = np.random.rand(horizon, 2)  # Generate random initial guess for controls [v, omega]
    bounds = [(-350.0, 350.0), (-2*np.pi, 2*np.pi)] * horizon  # Bounds for [v, omega]

    result = minimize(cost_function, u0.flatten(), args=(state, ref_trajectory, horizon, dt, weights),
                      method='SLSQP', bounds=bounds)
    
    if result.success:
        optimal_control = result.x[:2]  # Extract the first control input [v, omega]
    else:
        print("NMPC optimization failed")
        optimal_control = [0.0, 0.0]  # Default to no motion if optimization fails

    return optimal_control

def navigation_nmpc(agent, goal, schedule):
    """
    Set velocity for robots to follow the path in the schedule using PID control.

    Args:
        agent: ID of the robot agent.
        goal: Target position [x, y].
        schedule: Dictionary with agent IDs as keys and the list of waypoints to the goal as values.

    Returns:
        None
    """
    # Other configurations
    index = 0
    dis_th = 0.4
    trajectory = []  
    dt=0.5
    horizon=2
    basePos = p.getBasePositionAndOrientation(agent)
    
    print("schedule", schedule)

    while not checkPosWithBias(basePos[0], goal, dis_th):
        basePos = p.getBasePositionAndOrientation(agent)
        if index < len(schedule):
            next_wp = [schedule[index]["x"], schedule[index]["y"]]
            if checkPosWithBias(basePos[0], next_wp, dis_th):
                index += 1

        # Stop if the schedule is exhausted
        if index == len(schedule):
            p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            break

        # Get current position and orientation
        x = basePos[0][0]
        y = basePos[0][1]
        Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        
        state = np.array([x, y, Orientation]) 
        for i in range(len(schedule) - horizon):
            #ref_trajectory = schedule[i:i + horizon]
            ref_trajectory = [(schedule[j]['x'], schedule[j]['y']) for j in range(i, i + horizon)]
            control = nmpc_controller(state, ref_trajectory, horizon, dt)
            #state = vehicle_dynamics(state, control, dt)
            
        
        goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))

        # Normalize angles to [0, 2π]
        if Orientation < 0:
            Orientation += 2 * math.pi
        if goal_direction < 0:
            goal_direction += 2 * math.pi

        # Calculate angular error (theta)
        theta = goal_direction - Orientation
        if theta < -math.pi:
            theta += 2 * math.pi
        elif theta > math.pi:
            theta -= 2 * math.pi
        
        trajectory.append(state)

        # Append current position to the path
        if agent == 71:
            x_practical1.append(x)
            y_practical1.append(y)
        elif agent == 72:
            x_practical2.append(x)
            y_practical2.append(y)

        lin_vel = control[0]
        ang_vel = control[1]
        leftWheelVelocity = lin_vel - ang_vel
        rightWheelVelocity = lin_vel + ang_vel

        # Apply wheel velocities
        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)

    print(agent, "reached the goal")


def run(agents, goals, schedule):
    """
        Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.

        Args:

        agents: array containing the boxID for each agent

        schedule: dictionary with boxID as key and path to the goal as list for each robot.

        goals: dictionary with boxID as the key and the corresponding goal positions as values
    """
    threads = []
    for index, agent in enumerate(agents):
        #t = threading.Thread(target=navigation_nmpc, args=(agent, goals[agent], schedule[agent]))
        t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

# -------------------------
# Configurations
# -------------------------
# physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=multi_3.mp4 --mp4fps=30')
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

plane_id = p.loadURDF("plane.urdf")

global env_loaded
env_loaded = False

# Create environment
env_params = create_env("./final_challenge/env.yaml")

# Create turtlebots
agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("./final_challenge/actors.yaml")

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
                                     cameraTargetPosition=[4.5, 4.5, 4])


# -------------------------
# Shedule of CBS based on A*
# -------------------------
#cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="./final_challenge/cbs_output.yaml")
#schedule = read_cbs_output("./final_challenge/cbs_output.yaml")

# -------------------------
# Shedule of CBS based on dijkstra
# -------------------------
#cbsfxy.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="./final_challenge/cbsfxy_output.yaml")
#schedule = read_cbs_output("./final_challenge/cbsfxy_output.yaml")

# -------------------------
# Shedule of RRT*
# -------------------------
schedule = read_cbs_output("./output_rrt_star.yaml")
'''
path_agent1 = [(9, 9), (7.648331686304761, 9.00063391255553), (6.648377309782922, 9.010186130874812),
               (5.650754658051451, 8.94127281755825), (4.657259950865711, 9.055151115727485),
               (3.9033146188078307, 8.398213952622724), (4.1377371291183405, 7.426079143730604),
               (3.252562311658911, 6.960820567736925), (2.2730548022000185, 6.759412924104322),
               (1.3326311261482582, 5.963953868983381), (1.804336812002677, 5.005480735265726),
               (2.8042236465740165, 4.99043686439063), (3.8041600122792025, 4.979155705748714),
               (4.803939582960816, 4.95816022890991), (5.513862170196241, 4.2538805116576235),
               (6.2355501932014645, 3.329977180423897), (7.181699914728095, 3.006247811266449),
               (7.470817069558154, 2.2379917551722017), (6.902724804115767, 1.4169309671903918),
               (5.998091153951604, 0.9907409168716836), (5.005604211515799, 0.8683903177949862),
               (4.027120832479579, 0.7288987612988429), (3.0561250776862483, 0.38434070690924416)]

path_agent2 = [(0, 9), (0.9999595000679592, 9.00899990132375), (1.9998460886086216, 8.993939686965511),
               (2.96921444352278, 8.748328220435509), (3.703474600192613, 8.069460028275063),
               (4.509525020231316, 6.891187864867366), (5.507831947055712, 6.949353833048331),
               (6.493687986993805, 7.116948190106026), (7.483065118463835, 6.971576625747185),
               (8.02202600240544, 6.129245832305035), (7.509721628817375, 4.966160094879645),
               (6.511214279810601, 4.911542570407507), (5.678339782004123, 3.9572998725249904),
               (6.31116716949699, 3.183006970814693), (7.201639556431954, 2.727969688708768),
               (8.11095936039451, 2.311871750572204), (7.7211209699114445, 1.0027371339405267),
               (6.652934489981001, 0.22612328714082586), (6.29195019244826, 0.10024823759098855)]
schedule = np.array([path_agent1, path_agent2], dtype=object)
'''
# Replace agent name with box id in cbs_schedule
box_id_to_schedule = {}

for name, value in schedule.items():
    box_id_to_schedule[agent_name_to_box_id[name]] = value

x_practical1, x_practical2, y_practical1, y_practical2 = [], [], [], []

run(agent_box_ids, box_id_to_goal, box_id_to_schedule)
time.sleep(2)

# ---------------------------------------------------
# Plot the practical and ideal paths of the agents
# ---------------------------------------------------
start = (9, 9)
goal = (3, 0)
start_agent2 = (0, 9)
goal_agent2 = (6, 0)
size = 1
obstacles = [
        [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
        [-1, 4], [1, 4], [2, 4], [3, 4], [4, 4],
        [7, 4], [8, 4], [9, 4], [3, 6], [4, 6],
        [5, 6], [6, 6], [7, 6], [-1, 8], [1, 8],
        [2, 8], [5, 8], [6, 8], [7, 8], [8, 8],
        [9, 8], [-1, 0], [10, 0], [10, 1], [10, 2],
        [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], 
        [10, 8], [10, 9], [10, 10], [9, 10], [8, 10], 
        [7, 10], [6, 10], [5, 10], [4, 10], [3, 10], 
        [2, 10], [1, 10], [-1, 10], [-1, 9], [-1, 8], 
        [-1, 7], [-1, 6], [-1, 5], [-1, 4], [-1, 3], 
        [-1, 2], [-1, 1], [0, 10], [0, 8], [0, 4]]
obstacles = np.array(obstacles)
new_column = np.full((obstacles.shape[0], 1), 1)
obstacles = np.hstack((obstacles, new_column))


# Plot obstacles
fig, ax = plt.subplots(figsize=(10, 10))
for (x, y, size) in obstacles:
        lower_left_x = x - size / 2
        lower_left_y = y - size / 2
        square = patches.Rectangle((lower_left_x, lower_left_y), size, size, 
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(square)
ax.set_xlim(-2, 11)
ax.set_ylim(-1, 11)

# Plot agent path
for agent_id, path in box_id_to_schedule.items():
    # plot ideal path
    cbs_path_x = [point["x"] for point in path]
    cbs_path_y = [point["y"] for point in path]
    plt.plot(cbs_path_x, cbs_path_y, linestyle='--', label=f"Agent {agent_id} CBS Path")
#plt.plot(cbs_path_x, linestyle='--', label=f"Agent 1 CBS Path")
#plt.plot(cbs_path_y, linestyle='--', label=f"Agent 2 CBS Path")
plt.plot(x_practical1, y_practical1, label='Agent 71 Actual Path', color='blue')
plt.plot(x_practical2, y_practical2, label='Agent 72 Actual Path', color='red')

plt.plot(start[0], start[1], 'go', label='Start 1')
plt.plot(goal[0], goal[1], 'mo', label='Goal 1')
plt.plot(start_agent2[0], start_agent2[1], 'co', label='Start 2')
plt.plot(goal_agent2[0], goal_agent2[1], 'yo', label='Goal 2')
plt.title('Paths of Agents 71 and 72')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid()
plt.show()