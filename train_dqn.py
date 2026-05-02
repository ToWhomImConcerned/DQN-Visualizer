from grid_world import GridWorld
from dqn_agent import DQNAgent
import numpy

# environment setup
env = GridWorld(size=8)
state_size = env.grid_size * env.grid_size * 3
action_size = 4

agent = DQNAgent(state_size=state_size, action_size=action_size)

episodes = 1500
reward_history = []

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def state_to_input(agent_pos, goal, walls, grid_size):
    agent_layer = numpy.zeros(grid_size * grid_size)
    goal_layer  = numpy.zeros(grid_size * grid_size)
    wall_layer  = numpy.zeros(grid_size * grid_size)

    agent_layer[agent_pos[0] * grid_size + agent_pos[1]] = 1.0
    goal_layer[goal[0] * grid_size + goal[1]] = 1.0

    for wall in walls:
        wall_layer[wall[0] * grid_size + wall[1]] = 1.0

    return numpy.concatenate([agent_layer, goal_layer, wall_layer])

env.walls = [(2, 2), (3, 3), (4, 4)]
env.goal = (7, 7)

for episode in range(1, episodes + 1):
    pos = env.reset()
    state = state_to_input(pos, env.goal, env.walls, env.grid_size)
    done = False
    steps = 0
    total_reward = 0

    while not done:
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

        if steps > 200:
            break

    reward_history.append(total_reward)
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    if episode % 100 == 0:
        avg = numpy.mean(reward_history[-100:])
        print(f"Episode {episode} | Steps: {steps} | Reward: {total_reward} | Avg100: {avg:.1f} | Epsilon: {agent.epsilon:.3f}")