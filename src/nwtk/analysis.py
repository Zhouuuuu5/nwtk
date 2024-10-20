import networkx as nx
import numpy as np

def degree_distribution(G, number_of_bins=15, log_binning=True, density=True, directed=False):
    """
    Given a degree sequence, return the y values (probability) and the
    x values (support) of a degree distribution that you're going to plot.

    Parameters
    ----------
    G (nx.Graph):
        the network whose degree distribution to calculate

    number_of_bins (int):
        length of output vectors

    log_binning (bool):
        if you are plotting on a log-log axis, then this is useful

    density (bool):
        whether to return counts or probability density (default: True)
        Note: probability densities integrate to 1 but do not sum to 1.

    directed (bool or str):
        if False, this assumes the network is undirected. Otherwise, the
        function requires an 'in' or 'out' as input, which will create the
        in- or out-degree distributions, respectively.

    Returns
    -------
    bins_out, probs (np.ndarray):
        probability density if density=True node counts if density=False; binned edges

    """

    # Step 0: Do we want the directed or undirected degree distribution?
    if directed:
        if directed=='in':
            k = list(dict(G.in_degree()).values()) # get the in degree of each node
        elif directed=='out':
            k = list(dict(G.out_degree()).values()) # get the out degree of each node
        else:
            out_error = "Help! if directed!=False, the input needs to be either 'in' or 'out'"
            print(out_error)
            # Question: Is this the correct way to raise an error message in Python?
            #           See "raise" function...
            return out_error
    else:
        k = list(dict(G.degree()).values()) # get the degree of each node


    # Step 1: We will first need to define the support of our distribution
    kmax = np.max(k)    # get the maximum degree
    kmin = 0            # let's assume kmin must be 0


    # Step 2: Then we'll need to construct bins
    if log_binning:
        # array of bin edges including rightmost and leftmost
        bins = np.logspace(0, np.log10(kmax+1), number_of_bins+1)
    else:
        bins = np.linspace(0, kmax+1, num=number_of_bins+1)


    # Step 3: Then we can compute the histogram using numpy
    probs, _ = np.histogram(k, bins, density=density)


    # Step 4: Return not the "bins" but the midpoint between adjacent bin
    #         values. This is a better way to plot the distribution.
    bins_out = bins[1:] - np.diff(bins)/2.0

    return bins_out, probs


def closeness_centrality(G):
    """
    Calculate the closeness centrality for each node in a graph from scratch.

    Closeness centrality is a measure of how close a node is to all other nodes
    in the network. It is calculated as the reciprocal of the sum of the shortest
    path distances from a node to all other nodes in the graph. This function
    computes the closeness centrality for all nodes in the graph `G` without
    using any external library functions for the centrality calculation.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which the closeness centrality is calculated. It can
        be any type of graph (undirected, directed, etc.) supported by NetworkX.

    Returns
    -------
    centrality : dict
        A dictionary where the keys are nodes in the graph and the values are
        their corresponding closeness centrality scores. If a node is isolated,
        its centrality will be 0.

    Notes
    -----
    - For each node, this function computes the sum of shortest path lengths to
      all other reachable nodes in the graph using NetworkX's `shortest_path_length`.
    - Nodes that are disconnected from the rest of the graph will have a centrality
      of 0.0.
    - This function assumes that the graph is connected; however, it gracefully
      handles isolated nodes by assigning them a centrality score of 0.0.

    Time Complexity
    ---------------
    The time complexity is O(N * (V + E)), where N is the number of nodes, V is
    the number of vertices, and E is the number of edges, due to the use of
    shortest path calculations for each node.

    Citations
    ---------
    Bavelas, A. (1950). Communication patterns in task-oriented groups. The Journal
    of the Acoustical Society of America, 22(6), 725-730.

    Sabidussi, G. (1966). The centrality index of a graph. Psychometrika, 31(4), 581–603.

    Freeman, L. C. (1979). Centrality in social networks conceptual clarification.
    Social Networks, 1(3), 215–239.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.path_graph(4)
    >>> closeness_centrality_from_scratch(G)
    {0: 0.6666666666666666, 1: 1.0, 2: 1.0, 3: 0.6666666666666666}
    """

    centrality = {}
    N = G.number_of_nodes()  # Total number of nodes in the graph

    for node_i in G.nodes():
        # Compute shortest paths from node_i to all other nodes
        shortest_paths = nx.shortest_path_length(G, source=node_i)

        # Sum the lengths of the shortest paths
        total_distance = sum(shortest_paths.values())

        # Closeness centrality calculation (ignoring disconnected components)
        if total_distance > 0 and N > 1:
            centrality[node_i] = (N - 1) / total_distance
        else:
            centrality[node_i] = 0.0  # In case the node is isolated

    return centrality


def eigenvector_centrality(G, max_iter=100, tol=1e-08):
    """
    Calculate the eigenvector centrality for each node in a graph from scratch.

    Eigenvector centrality is a measure of a node's influence in a network based on
    the idea that connections to high-scoring nodes contribute more to the score
    of a node than equal connections to low-scoring nodes. This centrality measure
    assigns relative scores to all nodes in the network based on the principle that
    a node's centrality is determined by the centrality of its neighbors.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which the eigenvector centrality is calculated. It can be
        any type of graph (undirected, directed, etc.) supported by NetworkX.

    max_iter : int, optional (default=100)
        Maximum number of iterations for the power iteration method used to compute
        the centrality values. Higher values may be required for large graphs.

    tol : float, optional (default=1e-06)
        Tolerance for the convergence of the eigenvector centrality values. The
        algorithm iterates until the change in centrality values is smaller than this
        threshold.

    Returns
    -------
    centrality : dict
        A dictionary where the keys are nodes in the graph and the values are
        their corresponding eigenvector centrality scores.

    Notes
    -----
    - Eigenvector centrality was introduced by Bonacich (1972) as an extension
      of degree centrality, emphasizing the importance of connections to high-degree
      or highly influential nodes.
    - This algorithm computes eigenvector centrality using the power iteration
      method, which involves iteratively updating the centrality scores of each
      node based on the scores of their neighbors until convergence.
    - Eigenvector centrality works best in connected, undirected graphs; for
      directed or disconnected graphs, results may vary or be undefined.
    - The algorithm will stop either after `max_iter` iterations or when the
      centrality values converge to within the specified `tol`.

    Time Complexity
    ---------------
    The time complexity is O(V * E * I), where V is the number of vertices,
    E is the number of edges, and I is the number of iterations (limited by `max_iter`).

    Citations
    ---------
    Bonacich, P. (1972). Factoring and weighting approaches to status scores and
    clique identification. *Journal of Mathematical Sociology, 2*(1), 113-120.

    Newman, M. E. J. (2008). The mathematics of networks. *The New Palgrave
    Dictionary of Economics, 2*(1), 1-12.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> eigenvector_centrality_from_scratch(G)
    {0: 0.3730400736153818, 1: 0.2082196569730357, 2: 0.20624526357714606, ...}
    """

    # Initialize centrality dict with uniform values for all nodes
    centrality = {node: 1.0 / len(G) for node in G}
    N = len(G)

    # Power iteration method
    for _ in range(max_iter):
        prev_centrality = centrality.copy()
        max_diff = 0  # Track maximum change in centrality values

        for node in G:
            # Update centrality: sum of neighbors' centralities
            centrality[node] = sum(prev_centrality[neighbor] for neighbor in G[node])

        # Normalize centrality values (divide by Euclidean norm)
        norm = np.sqrt(sum(value ** 2 for value in centrality.values()))
        if norm == 0:
            return centrality  # Handle disconnected graphs
        centrality = {node: value / norm for node, value in centrality.items()}

        # Check for convergence
        max_diff = max(abs(centrality[node] - prev_centrality[node]) for node in G)
        if max_diff < tol:
            break

    return centrality


def calculate_modularity(G, partition):
    """
    Calculates the modularity score for a given partition of the graph, whether the graph is weighted or unweighted.

    Modularity is a measure of the strength of division of a network into communities. It compares the actual
    density of edges within communities to the expected density if edges were distributed randomly. For weighted
    graphs, the weight of the edges is taken into account.

    The modularity Q is calculated as:

    Q = (1 / 2m) * sum((A_ij - (k_i * k_j) / (2m)) * delta(c_i, c_j))

    where:
    - A_ij is the weight of the edge between nodes i and j (1 if unweighted).
    - k_i is the degree of node i (or the weighted degree for weighted graphs).
    - m is the total number of edges in the graph, or the total weight of the edges if the graph is weighted.
    - delta(c_i, c_j) is 1 if nodes i and j belong to the same community, and 0 otherwise.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph, which can be undirected and either weighted or unweighted. The graph's nodes represent the
        entities, and its edges represent connections between them.

    partition : list of sets
        A list of sets where each set represents a community. Each set contains the nodes belonging to that community.
        For example, [{0, 1, 2}, {3, 4}] represents two communities, one with nodes 0, 1, and 2, and another with nodes
        3 and 4.

    Returns:
    --------
    float
        The modularity score for the given partition of the graph. A higher score indicates stronger community structure,
        and a lower (or negative) score suggests weak or no community structure.

    Notes:
    ------
    - If the graph has weights, they will be used in the modularity calculation. If no weights are present, the function
      assumes each edge has a weight of 1 (i.e., unweighted).

    - The function assumes that all nodes in the graph are assigned to exactly one community. If any node is missing
      from the community list, it is treated as not belonging to any community, and the results may not be accurate.

    - If the graph has no edges, the modularity is undefined, and this function will return 0 because the total number
      of edges (2m) would be zero.

    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
    >>> modularity_score = calculate_modularity(G, communities)
    >>> print("Modularity:", modularity_score)

    References:
    -----------
    Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure
    in networks. Physical Review E, 69(2), 026113.
    """

    def remap_partition(partition):
        """
        Converts and remaps a partition to a list-of-lists structure suitable for modularity calculations.

        This function remaps the input partition (whether it's in dictionary form or a flat list of community labels)
        to a list-of-lists format, where each list represents a community and contains the nodes in that community.
        The function also ensures that community labels are contiguous integers starting from 0, which is typically
        required for modularity-based algorithms.
        """

        # if partition is a dictionary where the keys are nodes and values communities
        if type(partition)==dict:
            unique_comms = np.unique(list(partition.values()))
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in partition.items():
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(list(partition.values()))
            communities = [[] for i in unique_comms]
            for i,j in partition.items():
                communities[j].append(i)

            return communities

        # if partition is a list of community assignments
        elif type(partition)==list and\
                not any(isinstance(el, list) for el in partition):
            unique_comms = np.unique(partition)
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in enumerate(partition):
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(partition)
            communities = [[] for i in np.unique(partition)]
            for i,j in enumerate(partition):
                communities[j].append(i)

            return communities

        # otherwise assume input is a properly-formatted list of lists
        else:
            communities = partition.copy()
            return communities


    # We now should have a list-of-lists structure for communities
    communities = remap_partition(partition)

    # Total weight of edges in the graph (or number of edges if unweighted)
    if nx.is_weighted(G):
        m = G.size(weight='weight')
        degree = dict(G.degree(weight='weight'))  # Weighted degree for each node
    else:
        m = G.number_of_edges()  # Number of edges in the graph
        degree = dict(G.degree())  # Degree for each node (unweighted)

    # Modularity score
    modularity_score = 0.0

    # Loop over all pairs of nodes i, j within the same community
    for community in communities:
        for i in community:
            for j in community:
                # Get the weight of the edge between i and j, or assume weight 1 if unweighted
                if G.has_edge(i, j):
                    A_ij = G[i][j].get('weight', 1)  # Use weight if available, otherwise assume 1
                else:
                    A_ij = 0  # No edge between i and j

                # Expected number of edges (or weighted edges) between i and j in a random graph
                expected_edges = (degree[i] * degree[j]) / (2 * m)

                # Contribution to modularity
                modularity_score += (A_ij - expected_edges)

    # Normalize by the total number of edges (or total edge weight) 2m
    modularity_score /= (2 * m)


    return modularity_score