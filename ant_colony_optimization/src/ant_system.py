import numpy as np
from src.tsp import cost


class AntSystem:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        n_ants: int,
        round_trip: bool,
        rng: np.random.Generator,
    ):
        """Ant System algorithm.

        Parameters
        ----------
        distance_matrix : np.ndarray
            A square matrix M of shape (n_cities, n_cities) \
            containing the distance between each city. Mij is \
            the distance between city i and city j
        alpha : float
            Pheromone influence, alpha >= 0
        beta : float
            Distance influence, beta >= 0
        evaporation_rate : float
            Evaporation rate, 0 <= rho <= 1
        n_ants : int
            Number of ants
        round_trip : bool
            If True, the ants will return to the starting city.
        rng : np.random.Generator
            Random number generator.
        """
        self.distance_matrix = distance_matrix
        self.alpha = alpha
        self.beta = beta
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.round_trip = round_trip
        self.cities = np.arange(distance_matrix.shape[0])  # [0, 1, 2, ..., n_cities]
        self.pheromone = np.ones(distance_matrix.shape)
        self.best_solution: np.ndarray = None
        self.best_solution_cost: float = np.inf
        self.rng = rng

    def initialization(self) -> None:
        """Initialize the tabu list.

        Place the ants randomly on the graph.
        """
        self.tabu_list = np.zeros(
            (self.n_ants, self.distance_matrix.shape[0]),
            dtype=int,
        )
        self.tabu_list[:, 0] = self.rng.choice(
            self.cities,
            size=self.n_ants,
        )  # Place ants randomly on the graph

    def next_city(self, ant: int, current_city_index: int) -> int:
        """Return the next city to visit by an ant.

        Parameters
        ----------
        ant : int
            Ant index.
        current_city_index : int
            The current city index of the tabu list.

        Returns
        -------
        int
            The next city to visit.
        """
        visited_cities = self.tabu_list[ant, :current_city_index]
        current_city = visited_cities[-1]
        unvisited_cities = np.setdiff1d(self.cities, visited_cities)
        pheromone = self.pheromone[current_city, unvisited_cities]
        heuristic = self.distance_matrix[current_city, unvisited_cities]
        heuristic = 1 / heuristic
        probabilities = (pheromone**self.alpha) * (heuristic**self.beta)
        probabilities = probabilities / probabilities.sum()
        next_city = self.rng.choice(unvisited_cities, p=probabilities)
        return next_city

    def cycle(self) -> None:
        """Run a cycle of the algorithm."""
        for ant in range(self.n_ants):
            for city in range(1, self.distance_matrix.shape[0]):
                self.tabu_list[ant, city] = self.next_city(ant, city)

    def cycle_best_solutions(self, n: int = 1) -> np.ndarray:
        """Return the n best solutions of the cycle.

        Parameters
        ----------
        n : int, optional
            Number of solutions to return, by default 1.

        Returns
        -------
        np.ndarray
            An array of shape (n, n_cities) containing the n best solutions.
        """
        # Solutions costs
        costs = np.array(
            [
                cost(solution, self.distance_matrix, self.round_trip)
                for solution in self.tabu_list
            ]
        )
        # Update best ant system solution
        best_solution_index = np.argmin(costs)
        best_solution = self.tabu_list[best_solution_index]
        best_solution_cost = costs[best_solution_index]
        if best_solution_cost < self.best_solution_cost:
            self.best_solution = best_solution
            self.best_solution_cost = best_solution_cost
        # Return best solutions
        best_solutions = self.tabu_list[np.argpartition(costs, n)[:n]]
        return best_solutions

    def evaporation(self) -> None:
        """Pheromone evaporation."""
        self.pheromone *= 1 - self.evaporation_rate

    def reinforcement(self, solution: np.ndarray) -> None:
        """Pheromone reinforcement."""
        solution_cost = cost(solution, self.distance_matrix, self.round_trip)
        current_cities = solution[:-1]
        next_cities = solution[1:]
        self.pheromone[current_cities, next_cities] += 1 / solution_cost

    def run(self, max_cycles: int, verbose: bool = False) -> np.ndarray:
        """Run the algorithm.

        Parameters
        ----------
        max_cycles : int
            Maximum number of cycles.
        verbose : bool, optional
            If True, print the best solution cost at each iteration, by default False.
        """
        solutions_per_cycle = 1
        for i in range(max_cycles):
            self.initialization()
            self.cycle()
            solutions = self.cycle_best_solutions(n=solutions_per_cycle)
            # Pheromone update
            self.evaporation()
            for solution in solutions:
                self.reinforcement(solution)
            if verbose:
                print(f"Iteration {i + 1}: {self.best_solution_cost}")
        return self.best_solution
