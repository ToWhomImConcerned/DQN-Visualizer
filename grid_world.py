class GridWorld:
    def __init__(self, size=8):
        self.grid_size = size
        self.start = (0, 0)
        self.goal = (7, 7)
        self.walls = [(2, 2), (2, 0), (2, 1), (5, 2), (6, 2), (6, 4), (1, 5), (4, 6)]
        self.agent_pos = self.start

        # actions: 0=up, 1=down, 2=left, 3=right
        self.action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self, start=None):
        if start:
            self.agent_pos = start
        else:
            self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        row, col = self.agent_pos
        d_row, d_col = self.action_map[action]
        new_pos = (row + d_row, col + d_col)

        if (
            new_pos[0] < 0 or new_pos[0] >= self.grid_size or
            new_pos[1] < 0 or new_pos[1] >= self.grid_size or
            new_pos in self.walls
        ):
            return self.agent_pos, -1, False # state, reward, done
        
        self.agent_pos = new_pos

        if self.agent_pos == self.goal:
            return self.agent_pos, +10, True
        
        return self.agent_pos, 0, False
    
    def render(self):
        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                if (row, col) == self.agent_pos:
                    line += " A "
                elif (row, col) == self.goal:
                    line += " G "
                elif (row, col) in self.walls:
                    line += " X "
                else:
                    line += " . "
            print(line)
        print()