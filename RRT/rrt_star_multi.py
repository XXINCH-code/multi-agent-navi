import numpy as np
import yaml
import pybullet_data
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Node:
    """
    Define the Node class, representing each search point.

    Attributes:
        point (np.array): The coordinates of the node in space.
        cost (float): The total path cost from the start point to the current node.
        parent (Node): The parent node of the current node, used for path tracing.
    """

    def __init__(self, point, cost=0, parent=None):
        self.point = point  # node position
        self.cost = cost  
        self.parent = parent  # parent node
    def __repr__(self):
        return f"Node(point={self.point})"
    def __str__(self):
        return f"Node at {self.point}"

def distance(point1, point2):
    """Calculate the Euclidean distance between two points
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def nearest(nodes, q_rand):
    """Find the node closest to the random point q_rand among the existing nodes.
    """
    # In min, lambda is called to calculate each node return the node closest to q_rand.
    return min(nodes, key=lambda node: distance(node.point, q_rand))

def steer(q_near, q_rand, step_size=1):
    """Generate a new node from q_near in the direction of q_rand, but no more than step_size away.
    """
    direction = np.array(q_rand) - np.array(q_near.point)
    length = np.linalg.norm(direction)
    direction = direction / length  
    length = min(step_size, length)  
    return Node(q_near.point + direction * length)  

def is_collision_free(node, obstacles, other_agent_path=None, timestep=None, safe_distance=0.4):
    """
    Check if the node collides with obstacles or other agents at the same timestep.

    Args:
        node (Node): The current agent's node.
        obstacles (list): A list of obstacles, where each element is (x, y, size).
        other_agent_path (list of tuples): The path of another agent, where each element is (x, y).
        timestep (int): The current timestep, corresponding to other_agent_path.
        safe_distance (float): The minimum safe distance between agents.

    Returns:
        bool: Returns True if the node is collision-free, otherwise returns False.
    """
    # Checking for collisions with obstacles
    for (ox, oy, size) in obstacles:
        if np.linalg.norm([node.point[0] - ox, node.point[1] - oy]) <= 1*size:
            return False  # If collision, return False

    # Checking for collisions with other agents
    if other_agent_path and timestep is not None:
        if timestep < len(other_agent_path):
            other_point = other_agent_path[timestep]
            if np.linalg.norm(node.point - np.array(other_point)) <= safe_distance:
                return False  # If collision, return False

    return True  # No collision, return True

def find_path(nodes, start, goal, goal_threshold=0.5):
    # Find the node closest to the end point as the end point
    goal_node = min([node for node in nodes if distance(node.point, goal) < goal_threshold], key=lambda n: n.cost, default=None)
    path = []
    if goal_node is None:
        return path
    while goal_node is not None:
        path.append(tuple(goal_node.point))  # Adding a node to a path
        goal_node = goal_node.parent  # Backtracking to the parent node
    return path[::-1]  # Reverse Path

def rrt_star(start, goal, obstacles, num_iterations=3000, search_radius=0.8, other_paths=None):
    """perform rrt* algorithm
    """
    nodes = [Node(start)] 
    for t in range(num_iterations):
        q_rand = np.random.uniform(0, 10, 2)  # generate a random point
        q_near = nearest(nodes, q_rand)  # find the nearest node
        q_new = steer(q_near, q_rand)  # generate a new node
        if is_collision_free(q_new, obstacles, other_agent_path=other_paths, timestep=t):  # ensure no collision
            neighbors = [node for node in nodes if distance(node.point, q_new.point) < search_radius and is_collision_free(node, obstacles)]
            q_new.parent = min(neighbors, key=lambda node: node.cost + distance(node.point, q_new.point), default=q_near) if neighbors else q_near
            q_new.cost = q_new.parent.cost + distance(q_new.parent.point, q_new.point)
            nodes.append(q_new)  # add new nodes

    path = find_path(nodes, start, goal)  # find the path   

    return path


# -------------------------
# Configuration
# -------------------------
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

# -------------------------
# Find paths for two agents
# -------------------------
path_agent1 = rrt_star(start, goal, obstacles)
path_agent2 = rrt_star(start_agent2, goal_agent2, obstacles, other_paths=[path_agent1])

print("Path for agent1: ", path_agent1)
print("Path for agent2: ", path_agent2)

# -------------------------
# Save the output to a YAML file
# -------------------------
def create_schedule(paths):
    schedule = {}
    for agent_id, path in enumerate(paths, start=1):
         schedule[agent_id] = [
            {
                "t": t,
                "x": float(coord[0]),
                "y": float(coord[1])   
            }
            for t, coord in enumerate(path)
        ]
    return schedule
paths = [path_agent1, path_agent2]
schedule = create_schedule(paths)
output = {
    "cost": sum(len(path) for path in paths), 
    "schedule": schedule,
}
output_file = "output_rrt_star.yaml"
with open(output_file, "w") as file:
    yaml.dump(output, file)
print(f"YAML file is saved at {output_file}")

# -------------------------
# Plotting
# -------------------------
fig, ax = plt.subplots(figsize=(5, 5))
for (x, y, size) in obstacles:
        lower_left_x = x - size / 2
        lower_left_y = y - size / 2
        square = patches.Rectangle((lower_left_x, lower_left_y), size, size, 
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(square)

ax.set_xlim(-2, 11)
ax.set_ylim(-1, 11)
plt.title('Obstacles as Filled Squares')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().set_aspect('equal', adjustable='box') 

# plot agent1 path
for i in range(len(path_agent1) - 1):
    plt.plot([path_agent1[i][0], path_agent1[i+1][0]], [path_agent1[i][1], path_agent1[i+1][1]], 'r-')
plt.plot(start[0], start[1], 'go')
plt.plot(goal[0], goal[1], 'mo')

# plot agent2 path
for i in range(len(path_agent2) - 1):
    plt.plot([path_agent2[i][0], path_agent2[i+1][0]], [path_agent2[i][1], path_agent2[i+1][1]], 'g-')
plt.plot(start_agent2[0], start_agent2[1], 'co')
plt.plot(goal_agent2[0], goal_agent2[1], 'yo')
plt.grid()
plt.show()
