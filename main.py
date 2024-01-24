import numpy as np

NO_STRUCTURE = 0
STRUCTURE = 2


class Model:
    """This class represents the model of the city.
    It contains the grid and the methods modify it, as well as different calculations for the model.
    """

    gird = None

    def __init__(self, w=100, h=100) -> None:
        """Initialize the model with a grid of size w*h"""
        self.w = w
        self.h = h

        self.grid = np.zeros((self.w, self.h))

        # Set the center of the grid to be a city
        self.grid[self.w // 2, self.h // 2] = STRUCTURE

    def loop(self, n=100):
        """Run the model for `n' iterations"""
        for _ in range(n):
            self.update()

    def update(self, stickiness=1):
        """Update the model by adding a random walker to the grid. If the random walker is next to a city, it will become a city with probability `stickiness'"""

        # Set the walker to be a particle at a random point on the edge of a circle
        angle = np.random.uniform(0, 2 * np.pi)
        x = int(self.w // 2 + self.w // 2 * np.cos(angle))
        y = int(self.h // 2 + self.h // 2 * np.sin(angle))
        old_x, old_y = x, y

        while self.grid[x, y] == NO_STRUCTURE:
            old_x, old_y = x, y

            # Move the walker
            direction = np.random.randint(0, 4)
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

        # Set the walker to be a city
        self.grid[old_x, old_y] = STRUCTURE

    def get_fractal_dim(self, n_boxes=10):
        """Calculate the fractal dimension using the box counting method.

        This method divides the grid into n_boxes*n_boxes boxes and counts the number of boxes that contain a city.
        The result can be used to calculate the fractal dimension using the following formula:
        D = log(N) / log(1 / s)
        where N is the number of boxes that contain a city and s is the size of each box.

        https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension
        https://en.wikipedia.org/wiki/Box_counting"""

        # The size of each box
        box_size = self.w // n_boxes

        # The number of boxes that contain a city
        n_filled_boxes = 0

        # Loop over all boxes
        for i in range(n_boxes):
            for j in range(n_boxes):
                # Check if the box contains a city
                if np.any(
                    self.grid[
                        i * box_size : (i + 1) * box_size,
                        j * box_size : (j + 1) * box_size,
                    ]
                    == STRUCTURE
                ):
                    n_filled_boxes += 1

        # Calculate the fractal dimension
        return np.log(n_filled_boxes) / np.log(n_boxes)


# Current;ly unused
# def set_if_within_bounds(x, y, value):
#     # Set the value of grid[x,y] to value if x and y are within bounds
#     if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT:
#         grid[x,y] = value
