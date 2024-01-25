import numpy as np

NO_STRUCTURE = 0
STRUCTURE = 2
R_SCALING_FACTOR = 20
FOCAL_ATTRACTION = 0.2 # Between 0 and 1
DIRECTIONAL_DRIFT = 0.2 # Between 0 and 1


class Model:
    """This class represents the model of the city.
    It contains the grid and the methods modify it, as well as different calculations for the model.
    """

    def __init__(self, w=100, h=100) -> None:
        """Initialize the model with a grid of size w*h"""
        self.w = w
        self.h = h

        self.grid = np.zeros((self.w, self.h))

        # Set the center of the grid to be a city
        self.grid[self.w // 2, self.h // 2] = STRUCTURE

        self.direction_index = np.random.randint(0, 8)

    def loop(self, n=100, stickiness=1):
        """Run the model for `n' iterations"""
        for _ in range(n):
            self.update(stickiness)

    def focal_attraction(self, normalized_vector_to_center, directions):
        '''Given normalized vector from current position to the center, this function
        assigns probabilites to the directions of next move. Directions to the center
        are more likely due to focal attraction property.'''
        
        # Calculate dot products to see which direction most aligns with the vector
        dot_prods = np.dot(directions, normalized_vector_to_center)
        # Remove negative values
        positive_dot_prods = np.clip(dot_prods, 0, None)

        # Calculate probabilities (based on focal attraction)
        uniform_probs = np.ones(len(directions)) / len(directions)
        normalized_dot_prods = positive_dot_prods / positive_dot_prods.sum()
        probabilities_focal_attraction = ((1 - FOCAL_ATTRACTION) * uniform_probs)  + (FOCAL_ATTRACTION * normalized_dot_prods)

        return probabilities_focal_attraction
    

    def directional_drift(self, directions):
        '''Given the previous direction, this function assigns probabilites
        to the directions of next move. Directions forward (with respect to the
        previous direction) are more likely due to directional drift.'''

        # Calculate probabilities (based on directional drift)
        uniform_probs = np.ones(len(directions)) / len(directions)
        max_directional_drift_probs = np.zeros(len(directions))
        max_directional_drift_probs[(self.direction_index - 1) % len(directions)] = float(1/3) # Previous element
        max_directional_drift_probs[self.direction_index % len(directions)] = float(1/3)       # Current element
        max_directional_drift_probs[(self.direction_index + 1) % len(directions)] = float(1/3) # Next element
        probabilities_directional_drift = ((1 - DIRECTIONAL_DRIFT) * uniform_probs) + (DIRECTIONAL_DRIFT * max_directional_drift_probs)
        return probabilities_directional_drift

    def growing_radius(self):
        '''Calculates the radius where the particles appear. This radius
        grows logarithmically as a function of structure size'''
        radius = np.log(np.count_nonzero(self.grid) + 1) * R_SCALING_FACTOR
        if radius > self.w//2: radius = self.w // 2 #keeps radius within bounds
        return radius

    def update(self, stickiness=1):
        #Reset initial direction
        self.direction_index = np.random.randint(0, 8)
        
        # Set growing radius
        radius = self.growing_radius()
        
        # Set the walker to be a particle at a random point on the edge of a circle
        center_x, center_y = self.w // 2, self.h // 2
        angle = np.random.uniform(0, 2 * np.pi)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        old_x, old_y = x, y
        
        while self.grid[x, y] == NO_STRUCTURE:
            old_x, old_y = x, y

            # Get vector pointing to the center from current position
            vector_to_center = np.array([center_x - x, center_y - y])
            norm = np.linalg.norm(vector_to_center)

            # Normalize vector
            if norm != 0:
                normalized_vector_to_center = vector_to_center / norm 
            else:
                normalized_vector_to_center = np.array([0, 0])

            directions = np.array([
            [0, 1],   # N
            [1, 1],   # NE
            [1, 0],   # E
            [1, -1],  # SE  
            [0, -1],  # S
            [-1, -1], # SW
            [-1, 0],  # W
            [-1, 1]   # NW
            ])
            
            probabilities_focal_attraction = self.focal_attraction(normalized_vector_to_center, directions)
            probabilities_directional_drift = self.directional_drift(directions)

            # Combine directional drift and focal attraction
            probabilities = (probabilities_focal_attraction + probabilities_directional_drift) / 2

            # Choose direction and move
            self.direction_index = np.random.choice(len(directions), p = probabilities)
            direction = directions[self.direction_index]
            x += direction[0]
            y += direction[1]

            # Check if the walker is within bounds
            if x < 0 or x >= self.w or y < 0 or y >= self.h:
                return
                
        # Set the walker to be a city
        self.grid[old_x,old_y] = STRUCTURE

    def update_random_walk(self, stickiness=1):
        # Calculate radius (increases logarithmically as structure grows)
        radius = np.log(np.count_nonzero(self.grid) + 1) * R_SCALING_FACTOR
        if radius > self.w//2: 
            radius = self.w // 2 #keeps radius within bounds

        # Set the walker to be a particle at a random point on the edge of a circle
        angle = np.random.uniform(0, 2 * np.pi)
        x = int(self.w // 2 + radius * np.cos(angle))
        y = int(self.h // 2 + radius * np.sin(angle))
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

            # Check if the walker moved into a city
            if self.grid[x, y] == STRUCTURE and np.random.rand() > stickiness:
                x, y = old_x, old_y

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
    

    def density_gradient(self):
        center = self.w // 2
        distances = []
        densities = []
        for radius in range(1, center):
            mask = (np.fromfunction(lambda i, j: np.sqrt((i-center)**2 + (j-center)**2), self.grid.shape) < radius)
            densities.append(np.sum(self.grid[mask]) / np.pi / radius**2)
            distances.append(radius)
        return distances, densities





# Current;ly unused
# def set_if_within_bounds(x, y, value):
#     # Set the value of grid[x,y] to value if x and y are within bounds
#     if x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT:
#         grid[x,y] = value
