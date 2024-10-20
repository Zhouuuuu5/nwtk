import networkx as nx
import numpy as np

def degree_preserving_randomization(G, n_iter=1000):
    """
    Perform degree-preserving randomization on a graph.

    Degree-preserving randomization, also known as edge swapping or rewiring,
    is a method for creating randomized versions of a graph while preserving
    the degree distribution of each node. This is achieved by repeatedly
    swapping pairs of edges in the graph, ensuring that the degree (number of
    edges connected) of each node remains unchanged. The result is a graph
    with the same degree distribution but a randomized edge structure, which
    can be used as a null model to compare with the original network.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to be randomized. The graph can be directed or
        undirected, but it must be simple (i.e., no self-loops or parallel edges).

    n_iter : int, optional (default=1000)
        The number of edge swap iterations to perform. A higher number of
        iterations leads to more randomization, but the degree distribution
        remains preserved. Typically, the number of iterations should be
        proportional to the number of edges in the graph for sufficient
        randomization.

    Returns
    -------
    G_random : networkx.Graph
        A randomized graph with the same degree distribution as the original
        graph `G`, but with a shuffled edge structure.

    Notes
    -----
    - This method works by selecting two edges at random, say (u, v) and (x, y),
      and attempting to swap them to (u, y) and (x, v) (or (u, x) and (v, y)),
      ensuring that no self-loops or parallel edges are created in the process.
    - Degree-preserving randomization is particularly useful for creating null
      models in network analysis, as it allows for the investigation of whether
      specific network properties (e.g., clustering, path lengths) are a result
      of the network's structure or just its degree distribution.
    - The effectiveness of randomization depends on the number of iterations
      (`n_iter`). As a rule of thumb, using about 10 times the number of edges
      in the graph for `n_iter` often provides sufficient randomization.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G_random = degree_preserving_randomization(G, n_iter=100)

    Citations
    ---------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002).
    Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.

    Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks.
    *Science*, 296(5569), 910-913.
    """

    G_random = G.copy()
    edges = list(G_random.edges())
    num_edges = len(edges)

    for _ in range(n_iter):
        # Select two random edges (u, v) and (x, y)
        edge1_id = np.random.choice(list(range(len(edges))))
        u, v = edges[edge1_id]
        edge2_id = np.random.choice(list(range(len(edges))))
        x, y = edges[edge2_id]

        # Avoid selecting the same edge pair or creating self-loops
        if len({u, v, x, y}) == 4:
            # Swap the edges with some probability
            if np.random.rand() > 0.5:
                # Swap (u, v) with (u, y) and (x, v)
                if not (G_random.has_edge(u, y) or G_random.has_edge(x, v)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, y)
                    G_random.add_edge(x, v)
            else:
                # Swap (u, v) with (u, x) and (v, y)
                if not (G_random.has_edge(u, x) or G_random.has_edge(v, y)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, x)
                    G_random.add_edge(v, y)

        # Update edge list after changes
        edges = list(G_random.edges())


    return G_random

def configuration_model_from_degree_sequence(degree_sequence, return_simple=True):
    """
    Generate a random graph using the configuration model from a given degree sequence
    without using the NetworkX built-in function.

    The configuration model generates a random graph that preserves the degree
    sequence of nodes by assigning "stubs" or "half-edges" to each node and
    randomly pairing these stubs to form edges. This process can result in
    graphs with self-loops and parallel edges, which can be removed if needed.

    Parameters
    ----------
    degree_sequence : list of int
        A list representing the degree of each node in the graph. The sum of
        the degrees in this sequence must be even for the configuration model
        to create a valid graph.

    Returns
    -------
    G : networkx.MultiGraph
        A multigraph generated from the given degree sequence. The graph may
        contain self-loops and parallel edges, which are allowed in the
        configuration model.

    Notes
    -----
    - This method works by assigning "stubs" or "half-edges" to nodes based on
      their degree and then randomly pairing them to form edges. The resulting
      graph can have self-loops and parallel edges.
    - Self-loops and parallel edges can be removed post-generation if a simple
      graph is required using NetworkX's `nx.Graph(G)`.
    - The degree sequence must have an even sum for a valid graph construction.
      If the sum of the degrees is odd, no graph can be constructed.

    Time Complexity
    ---------------
    The time complexity is O(E), where E is the number of edges in the graph.

    Example
    -------
    >>> degree_sequence = [3, 3, 2, 2, 1, 1]
    >>> G = configuration_model_from_degree_sequence(degree_sequence)
    >>> nx.is_graphical(degree_sequence)
    True
    """

    # Check if the degree sequence is valid (sum of degrees must be even)
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("The sum of the degree sequence must be even.")

    # Create stubs list: node i appears degree_sequence[i] times
    stubs = []
    for node, degree in enumerate(degree_sequence):
        stubs.extend([node] * degree)

    # Shuffle stubs to randomize the pairing process
    np.random.shuffle(stubs)

    # Initialize an empty multigraph
    G = nx.MultiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(degree_sequence)))

    # Pair stubs to create edges
    while stubs:
        u = stubs.pop()
        v = stubs.pop()

        # Add the edge to the graph
        G.add_edge(u, v)

    if return_simple:
        # Convert the multigraph to a simple graph (remove parallel edges and self-loops)
        G_simple = nx.Graph(G)  # This removes parallel edges and self-loops by default

        return G_simple

    else:
        return G