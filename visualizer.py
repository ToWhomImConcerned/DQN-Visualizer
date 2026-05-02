import pygame
import numpy
import time
import random
import torch
from grid_world import GridWorld
from dqn_agent import DQNAgent

# Colors
BLACK     = (0, 0, 0)
WHITE     = (255, 255, 255)
DARK_GRAY = (25, 25, 35)
WALL      = (60, 60, 80)
GOAL      = (80, 200, 120)
AGENT     = (100, 180, 255)
GRID_LINE = (20, 20, 20)
TEXT      = (220, 220, 220)

CELL_SIZE = 80
PADDING   = 20
GRID_SIZE = 8

WIDTH  = GRID_SIZE * CELL_SIZE + PADDING * 2
HEIGHT = GRID_SIZE * CELL_SIZE + PADDING * 2 + 110

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def state_to_input(agent_pos, goal, walls, grid_size):
    agent_layer = numpy.zeros(grid_size * grid_size)
    goal_layer = numpy.zeros(grid_size * grid_size)
    wall_layer = numpy.zeros(grid_size * grid_size)

    agent_layer[agent_pos[0] * grid_size + agent_pos[1]] = 1.0
    goal_layer[goal[0] * grid_size + goal[1]] = 1.0

    for wall in walls:
        wall_layer[wall[0] * grid_size + wall[1]] = 1.0

    return numpy.concatenate([agent_layer, goal_layer, wall_layer])

def get_cell_value(row, col, env, agent):
    # can't evaluate wall or goal cells
    if (row, col) in env.walls:
        return None
    if (row, col) == env.goal:
        return None
    
    # build a state vector as if the agent were standing here
    state = state_to_input((row, col), env.goal, env.walls, env.grid_size)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        q_values = agent.network(state_tensor)

    return q_values.max().item()

def get_cell_color(row, col, env, agent):
    if (row, col) in env.walls:
        return WALL
    if (row, col) == env.goal:
        return GOAL
    if (row, col) == env.agent_pos:
        return AGENT
    
    value = get_cell_value(row, col, env, agent)

    if value is None:
        return DARK_GRAY
    
    # normalize between -10 and +10
    normalized = (value - 5) / 6.0
    normalized = max(0.0, min(1.0, normalized))

    r = int(normalized * 220)
    g = int(normalized * 100)
    b = int((1 - normalized) * 180)
    return (r, g, b)

def draw(screen, env, agent, episode, steps, total_reward, font, small_font, epsilon, speed=30):
    screen.fill(BLACK)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = PADDING + col * CELL_SIZE
            y = PADDING + row * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            color = get_cell_color(row, col, env, agent)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 2)

            # labels and q values
            if (row, col) in env.walls:
                label = small_font.render("WALL", True, WHITE)
            elif (row, col) == env.goal:
                label = small_font.render("GOAL", True, BLACK)
            elif (row, col) == env.agent_pos:
                label = small_font.render("AGENT", True, BLACK)
            else:
                value = get_cell_value(row, col, env, agent)
                if value is not None:
                    label = small_font.render(f"{value:.1f}", True, WHITE)
                else:
                    label = None

            if label:
                lw = label.get_width()
                lh = label.get_height()
                screen.blit(label, (x + (CELL_SIZE - lw) // 2,
                                    y + (CELL_SIZE - lh) // 2))

    # hud        
    hud = font.render(
        f"Episode: {episode}  Steps: {steps}  Reward: {total_reward}  Eps: {epsilon:.3f}",
        True, TEXT
    )
    screen.blit(hud, (WIDTH // 2 - hud.get_width() // 2, HEIGHT - 80))

    speed_label = font.render(f"Speed: {speed} fps  (W: faster  S: slower)", True, TEXT)
    screen.blit(speed_label, (WIDTH // 2 - speed_label.get_width() // 2, HEIGHT - 50))

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DQN Live Heatmap")
    font = pygame.font.SysFont("monospace", max(14, WIDTH // 35))
    small_font = pygame.font.SysFont("monospace", max(11, WIDTH // 45))
    clock = pygame.time.Clock()

    env = GridWorld(size=GRID_SIZE)
    state_size = GRID_SIZE * GRID_SIZE * 3
    action_size = 4
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    env.walls = [(2, 2), (3, 3), (4, 4)]
    env.goal = (7, 7)

    episodes = 1500
    reward_history = []
    train_speed = 30

    for episode in range(1, episodes + 1):
        pos = env.reset()
        state = state_to_input(pos, env.goal, env.walls, env.grid_size)
        done = False
        steps = 0
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        train_speed = min(300, train_speed + 20)
                    if event.key == pygame.K_s:
                        train_speed = max(20, train_speed - 20)

            action = agent.choose_action(state)
            new_pos, reward, done = env.step(action)
            next_state = state_to_input(new_pos, env.goal, env.walls, env.grid_size)

            prev_dist = manhattan_distance(pos, env.goal)
            new_dist = manhattan_distance(new_pos, env.goal)
            shaping = (prev_dist - new_dist) * 0.5
            shaped_reward = reward + shaping

            agent.remember(state, action, shaped_reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1
            pos = new_pos

            draw(screen, env, agent, episode, steps, total_reward, font, small_font, agent.epsilon, train_speed)
            clock.tick(train_speed)

            if steps > 200:
                break

        reward_history.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # exploitation loop after training
    print("Training complete!")
    for _ in range(20):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        pos = env.reset()
        state = state_to_input(pos, env.goal, env.walls, env.grid_size)
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
            action = torch.argmax(
                agent.network(torch.tensor(state, dtype=torch.float32))
            ).item()

            new_pos, reward, done = env.step(action)
            state = state_to_input(new_pos, env.goal, env.walls, env.grid_size)
            total_reward += reward
            steps += 1

            draw(screen, env, agent, episode, steps, total_reward, font, small_font, agent.epsilon, train_speed)
            time.sleep(0.3)

if __name__ == "__main__":
    main()