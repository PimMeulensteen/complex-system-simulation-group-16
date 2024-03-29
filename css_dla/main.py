"""
This module contains the model of the city using the DLA mechanism.

The model is represented by a grid of size w*h, where each cell can be either empty or contain a city.
The model is initialized with a single city at the center of the grid, or with two cities at the center of each half of the grid.
The model is updated by spawning a particle at a random point on the edge of a circle with a radius that grows logarithmically as a function of structure size.
The particle then moves randomly until it encounters an established structure.
The particle then becomes a city and the process repeats.
The model can be run for a given number of iterations using the loop method.
"""

import numpy as np
from scipy.optimize import curve_fit
import random
import math

# Debug mode means that assertions will be checked
DEBUG = False

NO_STRUCTURE = 0
STRUCTURE = 2
DEFAULT_BUFFER_SIZE = 5
R_SCALING_FACTOR = 20
FOCAL_ATTRACTION = 0.2  # Between 0 and 1
DIRECTIONAL_DRIFT = 0.2  # Between 0 and 1
DIRECTIONS = np.array(
    [
        [0, 1],  # N
        [1, 1],  # NE
        [1, 0],  # E
        [1, -1],  # SE
        [0, -1],  # S
        [-1, -1],  # SW
        [-1, 0],  # W
        [-1, 1],  # NW
    ]
)
LEN_DIRECTIONS = len(DIRECTIONS)


def my_random_function(collection, p):
    """Choose a random element from a collection based on the given probabilities.

    Faster than np.random.choice for small collections.
    Based on: https://stackoverflow.com/questions/18622781/why-is-numpy-random-choice-so-slow

    :param collection: The collection to choose from
    :param p: The probabilities of choosing each element
    :return: The chosen element
    """
    if isinstance(collection, int):
        collection = list(range(collection))
    miles = []
    current = 0
    for prob in p:
        miles.append(current)
        current += prob

    assert math.isclose(current, 1), "Probabilities must sum to 1"

    x = random.random()
    _all = list(zip(collection, miles))
    while len(_all) != 1:
        if _all[len(_all) // 2][1] < x:
            _all = _all[len(_all) // 2:]
        else:
            _all = _all[0: len(_all) // 2]
    return _all[0][0]


def assert_if_debug(condition: bool, message: str) -> None:
    """Assert that the condition is true if the program is in debug mode.

    :param condition: The condition to check
    :param message: The message to display if the condition is false
    """
    if DEBUG:
        assert eval(condition), message


def is_numeric(obj: object) -> bool:
    """Check if the object is numeric.

    :param obj: The object to check
    :return: Whether the object is numeric
    """
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


class Model:
    """This class represents the model of the city.
    It contains the grid and the methods modify it, as well as different calculations for the model.
    """

    def __init__(
        self,
        w: int = 100,
        h: int = 100,
        mode: str = "multiple",
        particle_generation="on_radius",
        seed=1,
    ) -> None:
        """Initialize the model with a grid of size w*h.

        :param w: The width of the grid
        :param h: The height of the grid
        :param mode: The mode of the model. Either 'single' or 'multiple'
        """
        assert mode in [
            "single",
            "multiple",
        ], "Mode must be either 'single' or 'multiple'"
        assert w > 0 and h > 0, "Width and height must be positive"
        assert isinstance(
            w, int) and isinstance(
            h, int), "Width and height must be integers"

        random.seed(seed)
        np.random.seed(seed)
        print(seed)

        self.w = w
        self.h = h

        self.grid = np.zeros((self.w, self.h))
        self.mode = mode

        self.direction_index = np.random.randint(0, 8)

        if self.mode == "single":
            # Set the center of the grid to be a city
            self.grid[self.w // 2, self.h // 2] = STRUCTURE
        elif self.mode == "multiple":
            # Place two centers for each settlement
            self.grid[4 * (self.w // 7), 4 * (self.h // 7)] = STRUCTURE
            self.grid[self.w * 3 // 7, self.h * 3 // 7] = STRUCTURE

        self.particle_generation = particle_generation

        assert_if_debug(
            "np.count_nonzero(self.grid) >= 1",
            "There should be at least one city at the start",
        )

    def loop(self, n: int = 100, stickiness: float = 1.0) -> None:
        """Run the model for `n' iterations.

        :param n: The number of iterations to run the model for
        :param stickiness: The stickiness of the model. Between 0 and 1
        """
        # Assert that the argument n is valid
        assert n > 0, "Number of iterations must be positive"
        assert isinstance(n, int), "Number of iterations must be an integer"

        # Assert that the argument stickiness is valid
        assert stickiness >= 0 and stickiness <= 1, "Stickiness must be between 0 and 1"
        assert isinstance(stickiness, float), "Stickiness must be a float"

        for _ in range(n):
            self.update(stickiness)

    def focal_attraction(self, normalized_vector_to_center):
        """Given normalized vector from current position to the center, this function
        assigns probabilites to the directions of next move. Directions to the center
        are more likely due to focal attraction property.

        :param normalized_vector_to_center: The normalized vector from the current position to the center
        :return: The probabilities of the next move
        """

        # Assert that the argument normalized_vector_to_center is valid
        assert normalized_vector_to_center.shape == (
            2,), "Vector must be 2-dimensional"
        assert_if_debug(
            "np.isclose(np.linalg.norm(normalized_vector_to_center), 1)",
            "Vector must be normalized",
        )

        # Calculate dot products to see which direction most aligns with the
        # vector
        dot_prods = np.dot(DIRECTIONS, normalized_vector_to_center)
        # Remove negative values
        positive_dot_prods = np.maximum(dot_prods, 0)

        # Calculate probabilities (based on focal attraction)
        uniform_probs = np.ones(LEN_DIRECTIONS) / LEN_DIRECTIONS
        normalized_dot_prods = positive_dot_prods / positive_dot_prods.sum()
        probabilities_focal_attraction = ((1 - FOCAL_ATTRACTION) * uniform_probs) + (
            FOCAL_ATTRACTION * normalized_dot_prods
        )

        # Assert that probabilities sum to 1
        assert_if_debug(
            "np.isclose(probabilities_focal_attraction.sum(), 1)",
            "Probabilities must sum to 1",
        )

        return probabilities_focal_attraction

    def directional_drift(self):
        """Given the previous direction, this function assigns probabilites
        to the directions of next move. Directions forward (with respect to the
        previous direction) are more likely due to directional drift.

        :return: The probabilities of the next move
        """

        # Calculate probabilities (based on directional drift)
        uniform_probs = np.ones(LEN_DIRECTIONS) / LEN_DIRECTIONS
        max_directional_drift_probs = np.zeros(LEN_DIRECTIONS)
        max_directional_drift_probs[
            (self.direction_index - 1) % LEN_DIRECTIONS
        ] = float(
            1 / 3
        )  # Previous element
        max_directional_drift_probs[self.direction_index % LEN_DIRECTIONS] = float(
            1 / 3
        )  # Current element
        max_directional_drift_probs[
            (self.direction_index + 1) % LEN_DIRECTIONS
        ] = float(
            1 / 3
        )  # Next element
        probabilities_directional_drift = ((1 - DIRECTIONAL_DRIFT) * uniform_probs) + (
            DIRECTIONAL_DRIFT * max_directional_drift_probs
        )

        # Assert that probabilities sum to 1
        assert_if_debug(
            "np.isclose(probabilities_directional_drift.sum(), 1)",
            "Probabilities must sum to 1",
        )
        return probabilities_directional_drift

    def growing_radius(self):
        """Calculates the radius where the particles appear. This radius
        grows logarithmically as a function of structure size

        :return: The radius where the particles appear
        """
        radius = np.log(np.count_nonzero(self.grid) + 1) * R_SCALING_FACTOR

        assert radius >= 0, "Radius must be positive, check R_SCALING_FACTOR."

        # Return the radius if it is within bounds, otherwise return the
        # maximum possible radius
        return min(radius, self.w // 2)

    def norm_vector_to_center(self, x, y, center_x, center_y):
        """Takes coordinates of current position and coordinates of some
        center (settlement). Returns a normalized vector to that center.

        :param x: The x-coordinate of the current position
        :param y: The y-coordinate of the current position
        :param center_x: The x-coordinate of the center
        :param center_y: The y-coordinate of the center
        :return: The normalized vector to the center
        """

        # Assert that the arguments are valid
        assert (
            x >= 0 and x < self.w and y >= 0 and y < self.h
        ), "Current position must be within bounds"
        assert (
            center_x >= 0 and center_x < self.w and center_y >= 0 and center_y < self.h
        ), "Center must be within bounds"

        # Get vector pointing to the center from current position
        vector_to_center = np.array([center_x - x, center_y - y])

        # Get the norm of the vector, but faster than numpy :)
        norm = self.distance(x, y, center_x, center_y)

        # Normalize vector
        if norm:
            normalized_vector_to_center = vector_to_center / norm
        else:
            normalized_vector_to_center = np.array([0, 0])

        return normalized_vector_to_center

    def distance(self, x1, y1, x2, y2):
        """Calculates distance between two coordinates.

        :param x1: The x-coordinate of the first point
        :param y1: The y-coordinate of the first point
        :param x2: The x-coordinate of the second point
        :param y2: The y-coordinate of the second point
        :return: The distance between the two points
        """
        assert_if_debug(
            "is_numeric(x1) and is_numeric(y1) and is_numeric(x2) and is_numeric(y2)",
            "Coordinates must be numeric",
        )

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def update(self, stickiness=1.0):
        """Spawns a particle and calculates its next coordinate, taking into account
        the directional drift and focal attraction. If multiple settlements are considered,
        the particle is more likely to head in the direction of the closer one.
        Places a settlement once the particle encounters an established structure.
        Stops if movement goes off the grid.

        :param stickiness: The stickiness of the model. Between 0 and 1
        """
        assert stickiness >= 0 and stickiness <= 1, "Stickiness must be between 0 and 1"

        # Reset initial direction
        self.direction_index = np.random.randint(0, 8)

        # Set grid center coordinates
        center_x, center_y = self.w // 2, self.h // 2

        # For multiple settlements define multiple center points
        if self.mode == "multiple":
            center1_x, center1_y = self.w // 4, self.h // 4
            center2_x, center2_y = self.w * 3 // 4, self.h * 3 // 4
            settlement_centers = [
                [center1_x, center1_y], [center2_x, center2_y]]

        if self.particle_generation == "on_radius":
            # Set growing radius
            radius = self.growing_radius()

            # Set the walker to be a particle at a random point on the edge of
            # a circle
            angle = np.random.uniform(0, 2 * np.pi)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

        elif self.particle_generation == "anywhere":
            # Spawn particle randomly
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)

        while self.grid[x, y] == NO_STRUCTURE:
            old_x, old_y = x, y

            if self.mode == "single":
                normalized_vector_to_center = self.norm_vector_to_center(
                    x, y, center_x, center_y
                )

            elif self.mode == "multiple":
                # Choose a city to go to by calculating probabilites based on
                # which one is closest
                distances_to_settlements = [
                    self.distance(x, y, center1_x, center1_y),
                    self.distance(x, y, center2_x, center2_y),
                ]
                city_choice_probabilities = distances_to_settlements / sum(
                    distances_to_settlements
                )

                city_chosen_index = my_random_function(
                    len(distances_to_settlements), p=city_choice_probabilities
                )

                # Get a normalized vector to chosen center
                center_x, center_y = (
                    settlement_centers[city_chosen_index][0],
                    settlement_centers[city_chosen_index][1],
                )
                normalized_vector_to_center = self.norm_vector_to_center(
                    x, y, center_x, center_y
                )

            # Take into account directional drift and focal attraction
            probabilities_focal_attraction = self.focal_attraction(
                normalized_vector_to_center
            )
            probabilities_directional_drift = self.directional_drift()

            # Combine directional drift and focal attraction
            probabilities = (
                probabilities_focal_attraction + probabilities_directional_drift
            ) / 2

            # Choose direction and move
            self.direction_index = my_random_function(
                LEN_DIRECTIONS, p=probabilities)

            direction = DIRECTIONS[self.direction_index]
            x += direction[0]
            y += direction[1]

            # Check if the walker is within bounds
            if x < 0 or x >= self.w or y < 0 or y >= self.h:
                return

            if self.grid[x, y] != NO_STRUCTURE and np.random.rand(
            ) > stickiness:
                x, y = old_x, old_y

        # Set the walker to be a city
        self.grid[old_x, old_y] = STRUCTURE

    def update_random_walk(self, stickiness=1.0):
        """Spawns a particle on a radius and calculates its next coordinate randomly.
        Places a settlement once the particle encounters an established structure.

        :param stickiness: The stickiness of the model. Between 0 and 1
        """
        assert stickiness >= 0 and stickiness <= 1, "Stickiness must be between 0 and 1"

        # Calculate radius (increases logarithmically as structure grows)
        radius = self.growing_radius()

        # Set the walker to be a particle at a random point on the edge of a
        # circle
        angle = random.uniform(0, 2 * np.pi)

        x = int(self.w // 2 + radius * np.cos(angle))
        y = int(self.h // 2 + radius * np.sin(angle))

        old_x, old_y = x, y
        while self.grid[x, y] == NO_STRUCTURE:
            old_x, old_y = x, y

            # Move the walker
            direction = random.randint(0, 3)
            if direction == 0:
                x += 1
            elif direction == 1:
                x -= 1
            elif direction == 2:
                y += 1
            else:
                y -= 1

            # Check if the walker is within bounds
            if x < 0:
                x = 0
            elif x >= self.w:
                x = self.w - 1

            if y < 0:
                y = 0
            elif y >= self.h:
                y = self.h - 1

            if self.grid[x, y] != NO_STRUCTURE and np.random.rand(
            ) > stickiness:
                x, y = old_x, old_y

        self.grid[old_x, old_y] = STRUCTURE

    def get_box_count(self, s):
        """Helper function for calculating the fractal dimension using the box counting method.
        Calculates and returns N, where N is the number of boxes that contain a developed part
        of the city and s is the size of each box.

        :param s: The size of each box
        :return: N, the number of developed cells.
        """

        # Assert that the argument n_boxes is valid
        assert s > 0, "Number of boxes must be positive"
        assert isinstance(s, int), "Number of boxes must be an integer"

        # The size of each box
        box_size = s
        n_boxes = self.w // s

        # The number of boxes that contain a city
        n_filled_boxes = 0

        # Loop over all boxes
        for i in range(n_boxes):
            for j in range(n_boxes):
                # Check if the box contains a city
                if np.any(
                    self.grid[
                        i * box_size: (i + 1) * box_size,
                        j * box_size: (j + 1) * box_size,
                    ]
                    == STRUCTURE
                ):
                    n_filled_boxes += 1

        # Calculate the fractal dimension
        return n_filled_boxes

    def get_fractal_dim(self):
        """Calculates the fractal dimension using the box counting method.

        This method divides the grid into n_boxes*n_boxes boxes and counts the number of boxes that contain a city.
        The result can be used to calculate the fractal dimension using the following formula:
        N(s)*s^D = C (Mandelbrot 1983)
        We estimate slope D to equation Log(N(s)) = Log(C) + Dlog(1/s) + error [1]
        where N is the number of boxes that contain a city, s is the size of each box, and D is the fractal dimension.

        [1] Guoqiang Shen, "Fractal dimension and fractal growth of urbanized areas"

        :return: The fractal dimension
        """
        # Store for different box sizes
        log_N = []
        log_1_over_s = []

        # Collect data for different box sizes
        for s in range(2, int(self.w // 2)):
            N = self.get_box_count(s)
            log_N.append(np.log(N))
            log_1_over_s.append(np.log(1 / s))

        def line_to_fit(log_1_over_s, C, D):
            return C + D * log_1_over_s

        # Fit a line over the data to get slope D i.e. the fractal dimension
        params, cov = curve_fit(line_to_fit, log_1_over_s, log_N)
        _, D_fit = params

        return D_fit

    def density_gradient(self, buffer_size=None):
        """Calculates the density gradient of the city.

        The density gradient is calculated by dividing the city into rings of increasing radius and calculating the density of each ring.
        The density of a ring is the number of developed cells in the ring divided by the number of cells in the ring.

        :param buffer_size: The size of the buffer between rings
        :return: The distances and densities of the rings
        """
        if not buffer_size:
            buffer_size = DEFAULT_BUFFER_SIZE
        center = self.w // 2
        distances = []
        densities = []
        for radius in range(buffer_size, center, buffer_size):
            # define the outer circle of the current ring
            outer_mask = (
                np.fromfunction(
                    lambda i, j: np.sqrt(
                        (i - center) ** 2 + (j - center) ** 2),
                    self.grid.shape,
                )
                < radius
            )
            # define the inner circle of the current ring
            inner_mask = np.fromfunction(
                lambda i, j: np.sqrt((i - center) ** 2 + (j - center) ** 2),
                self.grid.shape,
            ) < (radius - buffer_size)
            # the difference between the outer and inner masks
            ring_mask = outer_mask & ~inner_mask

            # calculate the number of cells and number of developed cells in
            # the ring
            ring_cells = np.sum(ring_mask)
            developed_cells = np.sum(self.grid[ring_mask] == STRUCTURE)

            # calculate density of the ring
            ring_density = developed_cells / ring_cells if ring_cells > 0 else 0

            if ring_density == 0 and radius > 50:
                break

            densities.append(ring_density)
            distances.append(
                radius - buffer_size / 2
            )  # use the midpoint of the ring as the distance

        assert len(distances) == len(
            densities
        ), "Length of distances and densities must be equal"
        assert np.all(np.array(distances) >= 0), "Distances must be positive"
        assert np.all(np.array(densities) >= 0), "Densities must be positive"
        assert np.all(distances[:-1] <= distances[1:]
                      ), "Distances must be sorted"

        return distances, densities
