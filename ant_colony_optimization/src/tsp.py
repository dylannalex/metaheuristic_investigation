import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cost(solution: np.ndarray, distance_matrix: np.ndarray, round_trip: bool):
    """Returns the cost of a solution.

    Parameters
    ----------
    solution : np.ndarray
        Solution to evaluate.
    distance_matrix : np.ndarray
        Distance matrix between each city.
    round_trip : bool
        Whether the solution is a round trip or not.
    """
    current_cities = solution[:-1]
    next_cities = solution[1:]
    cost = np.sum(distance_matrix[current_cities, next_cities])
    if round_trip:
        cost += distance_matrix[solution[-1], solution[0]]
    return cost


def read_tsp_file(file_name: str):
    """
    Reads the given .tsp file and returns a pandas DataFrame with the cities \
    coordinates.

    Parameters
    ----------
    file_name : str
        File name to read.
    """
    # Find the index of the line where the coordinates start
    with open(file_name, "r") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        if line.startswith("NODE_COORD_SECTION"):
            break

    # Read the file
    df = pd.read_csv(
        file_name, skiprows=index + 1, usecols=[1, 2], delimiter=" ", names=["x", "y"]
    )
    return df[:-1]


def distance_matrix(df: pd.DataFrame):
    """Returns the distance matrix between each city.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the cities coordinates
    """
    cities = df.values
    x = cities[:, 0]  # x-coordinates of all cities
    y = cities[:, 1]  # y-coordinates of all cities

    # Calculate the squared differences for x and y coordinates
    x_diff = x[:, np.newaxis] - x
    y_diff = y[:, np.newaxis] - y

    # Calculate the squared Euclidean distances
    distances_sq = x_diff**2 + y_diff**2

    # Calculate the Euclidean distances
    distances = np.sqrt(distances_sq)

    # Set the diagonal to infinity
    distances[np.diag_indices_from(distances)] = np.inf

    # Avoid division by zero (on heuristic computation)
    distances[distances == 0] += 1e-10

    return distances


def compress_distance(distance_matrix: np.ndarray, k: int):
    """Compresses the distance matrix by keeping only the k nearest cities for \
    each city. Returns the compressed distance matrix and the indices of the \
    k nearest cities.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix between each city.
    k : int
        Number of nearest cities to keep.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays of shape (n_cities, k) containing the distances and the \
        indices of the k nearest cities for each city.
    """
    k_near_distance = np.zeros(
        shape=(distance_matrix.shape[0], k), dtype=distance_matrix.dtype
    )
    k_near_cities = np.zeros(shape=(distance_matrix.shape[0], k), dtype=np.int32)
    for i, dist in enumerate(distance_matrix):
        # Find k nearest cities of city i
        k_near = np.argpartition(dist, k)[:k]
        # Save distances and cities
        k_near_distance[i] = dist[k_near]
        k_near_cities[i] = k_near
    return k_near_distance, k_near_cities


def plot_solution_path(
    df: pd.DataFrame,
    solution: np.ndarray,
    figsize=(10, 10),
    city_size: int = 1,
    path_alpha: float = 0.5,
):
    """
    Plots the path of a given solution.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing the cities coordinates.
    solution : np.ndarray
        Solution to plot.
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 10).
    city_size : int, optional
        City size. Default is 1.
    path_alpha : float, optional
        Path transparency. Default is 0.5.
    """
    df = df.copy()
    df["order"] = np.arange(len(df))
    df = df.iloc[solution]
    # Plot cities
    df.plot.scatter(x="x", y="y", title="Path", figsize=figsize, s=city_size)
    # Plot solution
    plt.plot(df["x"], df["y"], color="grey", alpha=path_alpha)
    # Join last and first point
    plt.plot(
        [df.iloc[-1]["x"], df.iloc[0]["x"]],
        [df.iloc[-1]["y"], df.iloc[0]["y"]],
        color="grey",
        alpha=path_alpha,
    )
    plt.show()
