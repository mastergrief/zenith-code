"""
CALM Graph theory backend — adjacency, paths, cycles, components, properties.

Models botch graph algorithm outputs. Pure computation on adjacency lists.
"""

from __future__ import annotations

from collections import deque


def adjacency_list(edges: list) -> dict:
    """Build adjacency list from edge list [(u,v), ...]."""
    adj = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def adjacency_matrix(edges: list, n: int) -> list:
    """Build n×n adjacency matrix from edge list [(u,v), ...]. 0-indexed."""
    mat = [[0] * int(n) for _ in range(int(n))]
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        mat[u][v] = 1
        mat[v][u] = 1
    return mat


def bfs(adj: dict, start) -> list:
    """BFS traversal order from start node."""
    visited = set()
    order = []
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adj.get(node, [])):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def dfs(adj: dict, start) -> list:
    """DFS traversal order from start node."""
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in sorted(adj.get(node, []), reverse=True):
                if neighbor not in visited:
                    stack.append(neighbor)
    return order


def connected_components(adj: dict) -> list:
    """Find all connected components. Returns list of sets."""
    visited = set()
    components = []
    for node in adj:
        if node not in visited:
            component = set()
            queue = deque([node])
            while queue:
                n = queue.popleft()
                if n not in visited:
                    visited.add(n)
                    component.add(n)
                    for neighbor in adj.get(n, []):
                        if neighbor not in visited:
                            queue.append(neighbor)
            components.append(component)
    return components


def has_cycle(adj: dict) -> bool:
    """Whether an undirected graph has a cycle."""
    visited = set()
    for start in adj:
        if start in visited:
            continue
        stack = [(start, None)]
        while stack:
            node, parent = stack.pop()
            if node in visited:
                return True
            visited.add(node)
            for neighbor in adj.get(node, []):
                if neighbor != parent and neighbor not in visited:
                    stack.append((neighbor, node))
                elif neighbor != parent and neighbor in visited:
                    return True
    return False


def degree(adj: dict, node) -> int:
    """Degree of a node (number of edges)."""
    return len(adj.get(node, []))


def degree_sequence(adj: dict) -> list:
    """Degree sequence (sorted descending)."""
    return sorted([len(neighbors) for neighbors in adj.values()], reverse=True)


def is_bipartite(adj: dict) -> bool:
    """Whether a graph is bipartite (2-colorable)."""
    color = {}
    for start in adj:
        if start in color:
            continue
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


def vertex_count(adj: dict) -> int:
    """Number of vertices."""
    return len(adj)


def edge_count(adj: dict) -> int:
    """Number of edges (undirected: count/2)."""
    return sum(len(v) for v in adj.values()) // 2


def is_tree(adj: dict) -> bool:
    """Whether graph is a tree (connected + acyclic + V-1 edges)."""
    v = vertex_count(adj)
    e = edge_count(adj)
    if e != v - 1:
        return False
    return len(connected_components(adj)) == 1


def diameter(adj: dict) -> int:
    """Graph diameter: longest shortest path. -1 if disconnected."""
    if len(connected_components(adj)) > 1:
        return -1
    max_dist = 0
    for start in adj:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    queue.append(neighbor)
        max_dist = max(max_dist, max(dist.values()))
    return max_dist


def shortest_path_length(adj: dict, start, end) -> int:
    """Shortest path length (BFS, unweighted). -1 if no path."""
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


def density(adj: dict) -> float:
    """Graph density: 2E / (V(V-1))."""
    v = vertex_count(adj)
    e = edge_count(adj)
    if v < 2:
        return 0.0
    return round(2 * e / (v * (v - 1)), 4)


GRAPH_THEORY_FUNCTIONS = {
    "adjacency_list": adjacency_list,
    "adjacency_matrix": adjacency_matrix,
    "bfs": bfs,
    "dfs": dfs,
    "connected_components": connected_components,
    "has_cycle": has_cycle,
    "degree": degree,
    "degree_sequence": degree_sequence,
    "is_bipartite": is_bipartite,
    "vertex_count": vertex_count,
    "edge_count": edge_count,
    "is_tree": is_tree,
    "diameter": diameter,
    "shortest_path_length": shortest_path_length,
    "density": density,
}

GRAPH_THEORY_NL_PATTERNS = [
    (r'(?:is)\s+(?:the\s+)?graph\s+(?:a\s+)?(?:tree)', None),
    (r'(?:is)\s+(?:the\s+)?graph\s+bipartite', None),
    (r'(?:does)\s+(?:the\s+)?graph\s+(?:have|contain)\s+(?:a\s+)?cycle', None),
    (r'(?:graph|network)\s+diameter', None),
    (r'(?:graph|network)\s+density', None),
    (r'(?:connected)\s+components', None),
    (r'(?:BFS|DFS|breadth.first|depth.first)\s+(?:from|starting)', None),
]
