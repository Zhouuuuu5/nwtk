# nwtk

[![PyPI - Version](https://img.shields.io/pypi/v/nwtk.svg)](https://pypi.org/project/nwtk)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nwtk.svg)](https://pypi.org/project/nwtk)

-----
`nwtk` is a Python package containing some useful utilities for network analysis and visualization.

## Table of Contents

- [nwtk](#nwtk)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Features](#features)
    - [Null Models](#null-models)
    - [Network Analysis](#network-analysis)
    - [Color](#color)
  - [Usage](#usage)
  - [License](#license)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Zhouuuuu5/nwtk.git
    ```
2. Activate your environment using conda or venv
   ```bash
   conda activate nwtk
   conda info
   ```
3. Navigate into the project directory:
    ```bash
    cd nwtk
    ```
4. Install:
    ```bash
    python -m pip install -e .
    ```


## Features

### Null Models
`degree_preserving_randomization(G, n_iter=1000)`

- Perform degree-preserving randomization on a graph.

`configuration_model_from_degree_sequence(degree_sequence, return_simple=True)`

- Generate a random graph using the configuration model from a given degree sequence without using the NetworkX built-in function.


### Network Analysis
`degree_distribution(G, number_of_bins=15, log_binning=True, density=True, directed=False)`
- Given a degree sequence, return the y values (probability) and the x values (support) of a degree distribution that you're going to plot.

`closeness_centrality(G)`
- Calculate the closeness centrality for each node in a graph.

`eigenvector_centrality(G, max_iter=100, tol=1e-08)`
- Calculate the eigenvector centrality for each node in a graph.

`calculate_modularity(G, partition)`
- Calculates the modularity score for a given partition of the graph, whether the graph is weighted or unweighted.

### Color
`get_colorblindness_colors(hex_col, colorblind_types='all')`

- Generates color representations for various types of colorblindness.

`lightness(hex_col)`

- Calculates the perceived lightness of a color.

`saturation(hex_col)`

- Calculates the saturation of a given hex color.

`hue(hex_col)`

- Calculates the hue of a given hex color.

`rgb_to_hsv(rgb)`

- Converts an RGB color to HSV format.

`rgb_to_hex(rgb)`

- Converts an RGB color to hex format.

`hex_to_rgb(value)`

- Converts a hex color code to an RGB tuple.

`hex_to_grayscale(hex_col)`

- Converts a hex color code to its grayscale equivalent.


## Usage

```python
import nwtk as nt
import networkx as nx

# Example: Calculate the closeness centrality
G = nx.erdos_renyi_graph(100, 0.05)
cc = nt.analysis.closeness_centrality(G)
print(cc)


# Example: Convert RGB to Hex
hex_color = nt.color.rgb_to_hex((255, 0, 0))
print(hex_color)
```


## License

`nwtk` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
