import gymnasium as gym
import math
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. Définition de la structure Node pour MCTS
# ─────────────────────────────────────────────────────────────
class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state, dtype=float)  # état = tableau de 4 floats
        self.parent = parent
        self.children = {}   # action -> Node
        self.visits = 0
        self.reward = 0.0

# ─────────────────────────────────────────────────────────────
# 2. Fonctions du MCTS
# ─────────────────────────────────────────────────────────────

def fully_expanded(node, action_space_n):
    return len(node.children) == action_space_n

def expand(node, action_space_n, make_env):
    for action in range(action_space_n):
        if action not in node.children:
            env = make_env()
            env.reset()
            env.unwrapped.state = node.state.copy()
            _, reward, terminated, truncated, _ = env.step(action)
            next_state = env.unwrapped.state.copy()
            env.close()

            child = Node(next_state, parent=node)
            node.children[action] = child
            return child
    return None

def best_uct(node, c=1.4):
    def uct(child):
        if child.visits == 0:
            return float("inf")
        exploitation = child.reward / child.visits
        exploration = c * math.sqrt(math.log(node.visits) / child.visits)
        return exploitation + exploration

    return max(node.children.values(), key=uct)

def rollout(node, make_env, max_steps=500):
    env = make_env()
    env.reset()
    env.unwrapped.state = node.state.copy()

    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    return total_reward

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent

def monte_carlo_tree_search(root, make_env, simulations=500, c=1.4):
    action_space_n = make_env().action_space.n
    for _ in range(simulations):
        node = root
        while fully_expanded(node, action_space_n):
            node = best_uct(node, c)
        leaf = expand(node, action_space_n, make_env) or node
        result = rollout(leaf, make_env)
        backpropagate(leaf, result)
    best_action = max(root.children, key=lambda a: root.children[a].visits)
    return best_action

# ─────────────────────────────────────────────────────────────
# 3. Environnement principal avec rendu
# ─────────────────────────────────────────────────────────────

def make_env_main(render=False):
    return gym.make("CartPole-v1", render_mode="human" if render else None)

env_main = make_env_main(render=True)
obs, _ = env_main.reset()
state = env_main.unwrapped.state.copy()
root = Node(state)

done = False
total_reward = 0

while not done:
    action = monte_carlo_tree_search(
        root,
        make_env=lambda: make_env_main(render=False),
        simulations=500
    )
    obs, reward, terminated, truncated, _ = env_main.step(action)
    done = terminated or truncated
    total_reward += reward

    if action in root.children:
        root = root.children[action]
        root.parent = None
    else:
        state = env_main.unwrapped.state.copy()
        root = Node(state)

print(f"Score final : {total_reward}")
env_main.close()
