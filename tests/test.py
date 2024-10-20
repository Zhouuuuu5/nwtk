import nwtk as nt
import networkx as nx

# Example: Calculate the closeness centrality
G = nx.erdos_renyi_graph(100, 0.05)
cc = nt.analysis.closeness_centrality(G)
print(cc)


# Example: Convert RGB to Hex
hex_color = nt.color.rgb_to_hex((255, 0, 0))
print(hex_color)
