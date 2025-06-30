import numpy as np
import yaml
import pybullet_data
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node:
    """定义节点类，代表每一个搜索点。
    Attributes:
        point (np.array): 节点在空间中的坐标。
        cost (float): 从起点到当前节点的总路径成本。
        parent (Node): 当前节点的父节点，用于追溯路径。
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

def steer(q_near, q_rand, step_size=1.0):
    """Generate a new node from q_near in the direction of q_rand, but no more than step_size away.
    """
    direction = np.array(q_rand) - np.array(q_near.point)
    length = np.linalg.norm(direction)
    direction = direction / length  
    length = min(step_size, length)  
    return Node(q_near.point + direction * length)  

def is_collision_free(node, obstacles):
    """check collision
    """
    
    for (ox, oy, size) in obstacles:
        if np.linalg.norm([node.point[0] - ox, node.point[1] - oy]) <= 1*size: # this threshold = sprt(2)/2 + radius of the robot
            return False  # if collision, return False
    return True  # no collision, return True

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

def rrt_star(start, goal, obstacles, num_iterations=2000, search_radius=1.5):
    """perform rrt* algorithm
    """
    nodes = [Node(start)] 
    for _ in range(num_iterations):
        q_rand = np.random.uniform(0, 10, 2)  # generate a random point
        q_near = nearest(nodes, q_rand)  # find the nearest node
        q_new = steer(q_near, q_rand)  # generate a new node
        if is_collision_free(q_new, obstacles):  # ensure no collision
            neighbors = [node for node in nodes if distance(node.point, q_new.point) < search_radius and is_collision_free(node, obstacles)]
            q_new.parent = min(neighbors, key=lambda node: node.cost + distance(node.point, q_new.point), default=q_near) if neighbors else q_near
            q_new.cost = q_new.parent.cost + distance(q_new.parent.point, q_new.point)
            nodes.append(q_new)  # add new nodes

    path = find_path(nodes, start, goal)  # find the path
    print(len(path))
    
    # plot the path  
    fig, ax = plt.subplots(figsize=(11, 11))
    for node in nodes:
        if node.parent:
            plt.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], 'b-')  # plot the search tree
    for i in range(len(path) - 1):
        plt.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], 'r-')  # plot the best path
    plt.plot(start[0], start[1], 'go')  
    plt.plot(goal[0], goal[1], 'mo')  
    
    for (x, y, size) in obstacles:
        lower_left_x = x - size / 2
        lower_left_y = y - size / 2
        square = patches.Rectangle((lower_left_x, lower_left_y), size, size, 
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(square)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.grid()
    plt.title('RRT Single Robot Result')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().set_aspect('equal', adjustable='box') 
    
    '''
    for (ox, oy, radius) in obstacles:
        circle = plt.Circle((ox, oy), radius, color='k', fill=True)
        plt.gca().add_patch(circle) 
    '''
    plt.show()


def simplify_path(path, epsilon=0.5):
    """使用 Ramer-Douglas-Peucker 算法简化路径。
    """
    if len(path) < 3:
        return path

start = (9, 9)
goal = (6, 0)
size = 1

obstacles = [(-5, -3, 2),(2, 5, 0.5),(-6, 5, 1), (6, -7, 1.5), (7, 7, 1)]
obstacles = [
        [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
        [0, 4], [1, 4], [2, 4], [3, 4], [4, 4],
        [7, 4], [8, 4], [9, 4], [3, 6], [4, 6],
        [5, 6], [6, 6], [7, 6], [0, 8], [1, 8],
        [2, 8], [5, 8], [6, 8], [7, 8], [8, 8],
        [9, 8], [0, 0], [10, 0], [10, 1], [10, 2],
        [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], 
        [10, 8], [10, 9], [10, 10], [9, 10], [8, 10], 
        [7, 10], [6, 10], [5, 10], [4, 10], [3, 10], 
        [2, 10], [1, 10], [0, 10], [0, 9], [0, 8], 
        [0, 7], [0, 6], [0, 5], [0, 4], [0, 3], 
        [0, 2], [0, 1]]

obstacles = np.array(obstacles)
new_column = np.full((obstacles.shape[0], 1), 1)
obstacles = np.hstack((obstacles, new_column))

nodes = rrt_star(start, goal, obstacles)  
