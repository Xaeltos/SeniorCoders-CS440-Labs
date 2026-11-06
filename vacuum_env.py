import random

class VacuumEnvironment:
    """
    A Grid World for the Vacuum Robot Q-Learning task.
    Accepts a 2D grid array and parses special states automatically.
    """
    def __init__(self, grid, living_cost=-0.1):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.living_cost = living_cost
        
        # Parse the grid to find positions
        self.start_pos = None
        self.goal_pos = None
        self.trap_pos = None
        self.obstacles = set()
        
        self._parse_grid()
        
        self.state = self.start_pos
        
        # Actions mapping: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = [0, 1, 2, 3]
        self.action_deltas = {
            0: (-1, 0), 
            1: (1, 0), 
            2: (0, -1), 
            3: (0, 1)
        }

    def _parse_grid(self):
        """Scans the grid to find start, goal, trap, and obstacles."""
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.grid[r][c]
                if val == "start":
                    self.start_pos = (r, c)
                elif val == 10:
                    self.goal_pos = (r, c)
                elif val == -1:
                    self.trap_pos = (r, c)
                elif val == "X":
                    self.obstacles.add((r, c))
                    
        if self.start_pos is None or self.goal_pos is None:
            raise ValueError("Grid must contain 'start' and '10' (goal).")

    def reset(self):
        """Resets the environment to the start position."""
        self.state = self.start_pos
        return self.state

    def step(self, action):
        """
        Executes an action with stochastic transitions (80% intended, 10% left, 10% right).
        Returns: next_state, reward, done
        """
        # 1. Stochastic transition logic
        prob = random.random()
        
        perpendicular = {
            0: (2, 3),  # Up -> Left, Right
            1: (2, 3),  # Down -> Left, Right
            2: (0, 1),  # Left -> Up, Down
            3: (0, 1)   # Right -> Up, Down
        }

        if prob < 0.8:
            actual_action = action
        elif prob < 0.9:
            actual_action = perpendicular[action][0]
        else:
            actual_action = perpendicular[action][1]


        # 2. Calculate next position
        dr, dc = self.action_deltas[actual_action]
        next_r = self.state[0] + dr
        next_c = self.state[1] + dc
        next_state = (next_r, next_c)

        # 3. Check boundaries and obstacles
        if (next_r < 0 or next_r >= self.rows or 
            next_c < 0 or next_c >= self.cols or 
            next_state in self.obstacles):
            next_state = self.state  # Bounce back

        self.state = next_state

        # 4. Calculate Reward and Done status
        if self.state == self.goal_pos:
            return self.state, 10.0, True
        elif self.state == self.trap_pos:
            return self.state, -1.0, True
        else:
            return self.state, self.living_cost, False