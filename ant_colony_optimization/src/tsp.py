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
    with open("data/dj38.tsp", "r") as f:
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

    return distances


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
