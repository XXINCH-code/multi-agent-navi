class Dijkstra():
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        Low-level search (Dijkstra's algorithm)
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        unvisited_nodes = {initial_state: 0}  # Initial state with distance 0
        came_from = {}  # Track the shortest path

        distances = {}  # Distance from the start to each node
        distances[initial_state] = 0

        while unvisited_nodes:
            # Find the node with the smallest distance
            current = min(unvisited_nodes, key=unvisited_nodes.get)
            current_distance = unvisited_nodes[current]

            # Check if the goal is reached
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            # Remove current node from unvisited nodes
            unvisited_nodes.pop(current)

            # Get neighbors and evaluate distances
            neighbor_list = self.get_neighbors(current)
            for neighbor in neighbor_list:
                tentative_distance = current_distance + step_cost

                # If this path is shorter or neighbor is unvisited
                if neighbor not in distances or tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    came_from[neighbor] = current
                    unvisited_nodes[neighbor] = tentative_distance

        return False  # If the goal is not reachable
