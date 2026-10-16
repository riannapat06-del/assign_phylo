"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def create_q_matrix(distances: list, total_dist: list) -> list:
    """Build the Q matrix used to choose which pair of nodes to join.
    Q[i][j] = (n - 2) * D[i][j] - r_i - r_j, where r_i is the sum of row i.

    Parameters
    ----------
    distances: list
        distance matrix (list of lists).
    total_dist: list
        Row sums of the distance matrix, one per node.

    Returns
    -------
    list
        Q matrix, with infinity on the diagonal so a node
        is never joined to itself.

    """
    n = len(distances)
    q_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(float("inf"))
            else:
                q_val = ((n - 2) * distances[i][j] - total_dist[i] - total_dist[j])
                row.append(q_val)
        q_matrix.append(row)
    return q_matrix

def find_closest_pair(q_matrix: list) -> tuple:
    """Find the pair of nodes with the smallest Q value.

    Parameters
    ----------
    q_matrix: list

    Returns
    -------
    tuple
        Indices (i, j) of the pair to join

    """
    n = len(q_matrix)
    smallest_q = None
    best_i = None
    best_j = None

    for i in range(n):
        for j in range(i + 1, n):
            if smallest_q is None or q_matrix[i][j] < smallest_q:
                smallest_q = q_matrix[i][j]
                best_i = i
                best_j = j

    return best_i, best_j

def update_distances(distances: list, i: int, j: int) -> tuple:
    """Replace nodes i and j in the distance matrix with their new parent.

    The parent is added at the END of the matrix. Distance to each
    remaining node k is (D[i][k] + D[j][k] - D[i][j]) / 2.

    Parameters
    ----------
    distances: list
        Current square distance matrix.
    i: int
        Index of the first node being joined.
    j: int
        Index of the second node being joined.

    Returns
    -------
    tuple
        (new_distances, remaining_nodes), where new_distances is the updated
        matrix and remaining_nodes is the list of old indices that were kept.

    """
    n = len(distances)

    # Find all nodes except the two being joined
    remaining_nodes = []
    for p in range(n):
        if p != i and p != j:
            remaining_nodes.append(p)

    # Keep distances between remaining nodes to update matrix
    new_distances = []
    for p in remaining_nodes:
        row = []
        # Copy the distances between remaining nodes
        for q in remaining_nodes:
            row.append(distances[p][q])

        # Calculate distance from p to the new parent
        new_dist = (distances[i][p] + distances[j][p] - distances[i][j]) / 2
        row.append(new_dist)
        new_distances.append(row)

    # Add the new parent's row
    parent_row = []
    for p in remaining_nodes:
        new_dist = (distances[i][p] + distances[j][p] - distances[i][j]) / 2
        parent_row.append(new_dist)
    parent_row.append(0)
    new_distances.append(parent_row)

    return new_distances, remaining_nodes

def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Create leaf node for each virus
    nodes = []
    for virus in labels:
        leaf = Node(virus, None, 0, None, 0, None)
        nodes.append(leaf)

    # Convert distance matrix into lists
    distances = [list(row) for row in distances]
    unique_node_count = 1
    
    # Keep joining pairs of nodes until only final two remain
    while len(nodes) > 2:
        # Calculate total distance for each virus
        total_dist = []
        for dist in distances:
            total_dist.append(sum(dist))

        # Calculate Q matrix
        n = len(nodes)
        q_matrix = create_q_matrix(distances, total_dist)
        # Find the pair with the smallest value (closest)
        i, j = find_closest_pair(q_matrix)

        # Calculate branch lengths, note two branch lengths add up to D(i,j)
        dist_i = (distances[i][j] / 2 + (total_dist[i] - total_dist[j]) / (2 * (n - 2)))
        dist_j = distances[i][j] - dist_i

        # Create the parent node joining i and j
        parent = Node(f"node_{unique_node_count}", nodes[i], dist_i, nodes[j], dist_j)
        unique_node_count += 1

        # Remove i and j from the matrix and add the new parent at the end
        distances, remaining_nodes = update_distances(distances, i, j)

        # Update active nodes where the original joined nodes are now children of parent
        nodes = [nodes[p] for p in remaining_nodes]
        nodes.append(parent)

    # Once only two nodes remain, connect them under a final root
    # Split the remaining distance equally between the two branches
    final_dist = distances[0][1] / 2
    root = Node("root", nodes[0], final_dist, nodes[1], final_dist)
    return root

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    leaves = []
    stack = [tree]

    while stack:
        node = stack.pop()
        # Find all leave nodes
        if node.left is None and node.right is None:
            leaves.append(node)
        else:
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    # Assign y-coord to leaves
    pos = {}
    i = 0

    for leaf in leaves:
        pos[leaf] = (None, i)
        i = i + 1

    # Assign x-coord based on node info
    root = tree
    root_x_coord = 0
    visited = False

    stack = [(root, root_x_coord, visited)]
    while stack:
        node, x, visited = stack.pop()
        if node in leaves:
            y_coord = pos[node][1]
            pos[node] = (x, y_coord)
        elif not visited:
            visited = True
            stack.append((node, x, visited))
            # Add right child
            if node.right:
                x_right = x + node.right_distance
                visited = False
                stack.append((node.right, x_right, visited))
            # Add left child
            if node.left:
                x_left = x + node.left_distance
                visited = False
                stack.append((node.left, x_left, visited))
        else:
            # Parent node being proocessed for second time
            left = pos[node.left][1]
            right = pos[node.right][1]
            avg = (left + right) / 2
            pos[node] = (x, avg) 

    # Draw tree using Matplotlib
    stack = [tree]
    while stack:
        node = stack.pop()

        if node.left is not None and node.right is not None:
            # Get coordinates
            x, y = pos[node]
            left_x, left_y = pos[node.left]
            right_x, right_y = pos[node.right]

            # Draw vertical line between children
            ax.plot([x, x], [left_y, right_y], color="black")

            # Draw horizontal line to left child
            ax.plot([x, left_x], [left_y, left_y], color="black")

            # Draw horizontal line to right child
            ax.plot([x, right_x], [right_y, right_y], color="black")

            # Process children
            stack.append(node.right)
            stack.append(node.left)

    # label leaves
    for leaf in leaves:
        x, y = pos[leaf]
        label = leaf.name.split("|")[0]
        ax.text(x, y, label, va="center")

    # Format plot
    ax.set_xlabel("Sequence distance")
    ax.set_yticks([])
    ax.margins(x=0.15)

    # Remove top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    # Traverse the tree and store all nodes
    stack = [tree]
    leaves = {}
    ordered_node = []
    while stack:
        node = stack.pop()

        # Add children to the stack
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

        ordered_node.append(node)

    # Reverse order so children are processed before parents
    ordered_node.reverse()

    for node in ordered_node:
        # Leaf nodes contain exactly one leaf
        if not node.left and not node.right:
            leaves[node] = 1
        else:
            # Count the total leaves in both subtrees
            leaves[node] = leaves[node.left] + leaves[node.right]

            # Put the subtree with fewer leaves on the left
            if leaves[node.left] > leaves[node.right]:
                node.left, node.right = node.right, node.left

                # Swap branch lengths to preserve distances
                node.left_distance, node.right_distance = (node.right_distance, node.left_distance)



def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()