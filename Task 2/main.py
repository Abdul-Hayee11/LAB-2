# Task One Implement Stack and Braket Checker Funtion

from pythonds.basic.stack import Stack

def parChecker(symbolString):
	s = Stack()
	balanced = True
	index = 0
	while index < len(symbolString) and balanced:
		symbol = symbolString[index]
		if symbol in '[{<(':
			s.push(symbol)
		else:
			if s.isEmpty():
				balanced = False
			else:
				top = s.pop()
				if not matches(top, symbol):
					balanced = False
		index = index + 1
	if balanced and s.isEmpty():
		return True
	else:
		return False



def matches(open, close):
	opens = '[{<('
	closers = ']}>)'
	return opens.index(open) == closers.index(close)

print(parChecker('<()>'))


# Task Two Converter Binary to Decimal

def divideByTwo(dec_number):
    remstack = Stack()
    
    while dec_number > 0:
        rem = dec_number % 2
        remstack.push(rem)
        dec_number = dec_number // 2
    
    binString = ''
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())
    return binString


print(divideByTwo(562))
        
        
def divideBy16(dec_number):
    remstack = Stack()
    hexdigits = '0123456789ABCDEF'
    
    while dec_number > 0:
        rem = dec_number % 16
        remstack.push(rem)
        dec_number = dec_number // 16
    
    hexString = ''
    while not remstack.isEmpty():
        hexString = hexString + hexdigits[remstack.pop()]
    return hexString


print(divideBy16(1000))

# Task 3 - Bellman Algorithm 
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('A', 'B', weight=1)
G.add_edge('B', 'C', weight=1)
G.add_edge('C', 'D', weight=1)
G.add_edge('A', 'C', weight=1)
G.add_edge('A', 'E', weight=1)
G.add_edge('A', 'F', weight=1)
G.add_edge('F', 'G', weight=1)
G.add_edge('D', 'G', weight=1)

def bellman_ford(graph, source):
    shortest_paths = {node: float('inf') for node in graph.nodes()}
    shortest_paths[source] = 0
    for _ in range(len(graph.nodes())-1):
        for u, v, data in graph.edges(data=True):
             weight = data['weight']
             if shortest_paths[u] + weight < shortest_paths[v]:
                 shortest_paths[v] = shortest_paths[u] + weight
             if shortest_paths[v] + weight < shortest_paths[u]:
                 shortest_paths[u] = shortest_paths[v] + weight
             
    return shortest_paths

shortest_paths_from_u = bellman_ford(G, 'A')
print('Shortest paths from A: ', shortest_paths_from_u)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="green")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()


# Task 4 - Dijkstra Algorithm
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('u', 'v', weight=3)
G.add_edge('u', 'w', weight=2)
G.add_edge('v', 'x', weight=1)
G.add_edge('w', 'x', weight=1)
G.add_edge('v', 'y', weight=2)
G.add_edge('x', 'z', weight=4)
G.add_edge('y', 'z', weight=1)
G.add_edge('x', 't', weight=5)
G.add_edge('w', 's', weight=4)
G.add_edge('s', 't', weight=3)

def dijkstra(graph, source):
    shortest_paths = {node: float('inf') for node in graph.nodes()}
    shortest_paths[source] = 0
    unvisited_nodes = set(graph.nodes())
    
    while unvisited_nodes:
        min_node = None
        for node in unvisited_nodes:
            if min_node is None or shortest_paths[node] < shortest_paths[min_node]:
                min_node = node
        unvisited_nodes.remove(min_node)
        current_weight = shortest_paths[min_node]
        
        for neighbor in graph.neighbors(min_node):
            weight = current_weight + graph[min_node][neighbor]['weight']
            if weight < shortest_paths[neighbor]:
                shortest_paths[neighbor] = weight
    return shortest_paths

shortest_paths_from_u = dijkstra(G, 'u')
print('Shortest paths from U: ', shortest_paths_from_u)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="red")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()


# Task 5 - Prims Algorithm
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edge('u', 'v', weight=3)
G.add_edge('u', 'w', weight=2)
G.add_edge('v', 'x', weight=1)
G.add_edge('w', 'x', weight=1)
G.add_edge('v', 'y', weight=2)
G.add_edge('x', 'z', weight=4)
G.add_edge('y', 'z', weight=1)
G.add_edge('x', 't', weight=5)
G.add_edge('w', 's', weight=4)
G.add_edge('s', 't', weight=3)

def prims(graph):
    minimum_spanning_tree = nx.Graph()
    start_node = list(graph.nodes())[0]
    minimum_spanning_tree.add_node(start_node)
    
    while len(minimum_spanning_tree) < len(graph):
        edge_to_add = None
        min_weight = float('inf')
        for node in minimum_spanning_tree.nodes():
            for neighbor in graph.neighbors(node):
                if neighbor not in minimum_spanning_tree.nodes():
                    weight = graph[node][neighbor]['weight']
                    if weight < min_weight:
                        edge_to_add = (node, neighbor)
                        min_weight = weight
        minimum_spanning_tree.add_edge(*edge_to_add)
    return minimum_spanning_tree

minimum_spanning_tree = prims(G)
print('Minimum spanning tree edges: ', minimum_spanning_tree.edges())
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="yellow")
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()


# Taks 6 - Q-learn
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict

# Create a graph with 25 nodes and random weights
def create_graph(n, edge_prob=0.3):  # Reduce edge probability for less complexity
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_prob:  # Increase edge probability if graph is too sparse
                G.add_edge(i, j, weight=random.randint(1, 10))
    return G

# Q-Learning implementation with max_steps to avoid infinite loops
def q_learning(graph, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=200, max_steps=100):
    Q = defaultdict(lambda: defaultdict(float))
    all_nodes = list(graph.nodes())
    
    for episode in range(episodes):
        state = random.choice(all_nodes)
        steps = 0
        
        while steps < max_steps:  # Limit max steps per episode
            # Choose action: either random or the best-known action
            if random.random() < epsilon:
                action = random.choice(list(graph.neighbors(state)))
            else:
                if Q[state]:
                    action = max(Q[state], key=Q[state].get)
                else:
                    action = random.choice(list(graph.neighbors(state)))
            
            # Reward is negative of the edge weight
            reward = -graph[state][action]['weight']
            next_state = action
            
            max_future_q = max(Q[next_state].values(), default=0)
            Q[state][action] += alpha * (reward + gamma * max_future_q - Q[state][action])
            
            state = next_state
            steps += 1  # Increase steps for each iteration
            
            if state == random.choice(all_nodes):  # Stop episode when it reaches a random node
                break
    
    return Q

# Get the shortest path from the Q-table
def get_q_path(Q, start, goal):
    path = [start]
    while start != goal:
        if Q[start]:
            start = max(Q[start], key=Q[start].get)
        else:
            break  # Exit if no path found in Q
        path.append(start)
    return path

# Dijkstra's algorithm implementation
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor in graph.neighbors(current_node):
            distance = current_distance + graph[current_node][neighbor]['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

# Prim's algorithm implementation
def prim(graph):
    start = random.choice(list(graph.nodes()))
    mst = nx.Graph()
    visited = set([start])
    edges = [(graph[start][to]['weight'], start, to) for to in graph.neighbors(start)]
    heapq.heapify(edges)
    
    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add_edge(frm, to, weight=weight)
            for next_node in graph.neighbors(to):
                if next_node not in visited:
                    heapq.heappush(edges, (graph[to][next_node]['weight'], to, next_node))
    
    return mst

# Main comparison with separate visualizations
def compare_algorithms():
    G = create_graph(25, edge_prob=0.3)  # Lower edge probability for faster execution
    
    start, goal = 0, 24  # Fixed start and goal nodes

    # 1. Apply Q-Learning
    print("Running Q-Learning...")
    Q = q_learning(G, episodes=200, max_steps=50)  # Reduce episodes and limit steps
    try:
        q_path = get_q_path(Q, start, goal)
        if q_path:
            print(f"Q-Learning Path from {start} to {goal}: {q_path}")
        else:
            print(f"No path found by Q-Learning from {start} to {goal}")
    except Exception as e:
        print(f"Q-Learning failed: {e}")
    
    # 2. Apply Dijkstra's Algorithm
    print("Running Dijkstra's Algorithm...")
    dijkstra_distances = dijkstra(G, start)
    dijkstra_path = [start]
    current_node = start
    while current_node != goal:
        neighbors = list(G.neighbors(current_node))
        next_node = min(neighbors, key=lambda n: dijkstra_distances.get(n, float('inf')))
        dijkstra_path.append(next_node)
        current_node = next_node
    print(f"Dijkstra's Path from {start} to {goal}: {dijkstra_path}")
    
    # 3. Apply Prim's Algorithm (Minimum Spanning Tree)
    print("Running Prim's Algorithm...")
    mst = prim(G)
    print(f"Prim's Algorithm created Minimum Spanning Tree with {mst.number_of_edges()} edges.")
    
    # 4. Visualize the results in separate graphs
    pos = nx.spring_layout(G)  # Use the same layout for all visualizations

    # Q-Learning Path Visualization
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    path_edges = [(q_path[i], q_path[i+1]) for i in range(len(q_path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="blue", width=2, label="Q-Learning Path")
    plt.title("Q-Learning Path")
    plt.show()

    # Dijkstra's Path Visualization
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", edge_color="gray")
    dijkstra_edges = [(dijkstra_path[i], dijkstra_path[i+1]) for i in range(len(dijkstra_path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=dijkstra_edges, edge_color="orange", width=2, label="Dijkstra's Path")
    plt.title("Dijkstra's Shortest Path")
    plt.show()

    # Prim's Minimum Spanning Tree Visualization
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightyellow", edge_color="gray")
    nx.draw_networkx_edges(mst, pos, edge_color="red", width=2, label="Prim's MST")
    plt.title("Prim's Minimum Spanning Tree")
    plt.show()

compare_algorithms()
