import gymnasium as gym
import math
import numpy as np
import time


# ────────────────────────────────────────────────
# Classe Node avec suivi de profondeur
# ────────────────────────────────────────────────
class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state, dtype=float)
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0.0
        self.depth = 0 if parent is None else parent.depth + 1


# ────────────────────────────────────────────────
# Fonctions MCTS
# ────────────────────────────────────────────────

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
        exploration = c * math.sqrt(math.log(node.visits + 1) / child.visits)
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


def monte_carlo_tree_search(root, make_env, simulations=500, c=1.4, verbose=False):
    action_space_n = make_env().action_space.n
    all_sim_rewards = []
    max_depth = 0

    for _ in range(simulations):
        node = root
        while fully_expanded(node, action_space_n):
            node = best_uct(node, c)
        leaf = expand(node, action_space_n, make_env) or node
        reward = rollout(leaf, make_env)
        backpropagate(leaf, reward)
        all_sim_rewards.append(reward)
        max_depth = max(max_depth, leaf.depth)

    best_action = max(root.children, key=lambda a: root.children[a].visits)

    if verbose:
        print(f"\n🎯 Best action: {best_action}")
        for a, child in root.children.items():
            exploitation = child.reward / child.visits if child.visits else 0
            print(f"  - Action {a}: visits={child.visits}, avg reward={exploitation:.2f}")
        print(f"📊 Simulations done: {simulations}")
        print(f"📈 Avg rollout reward: {np.mean(all_sim_rewards):.2f}, Max tree depth: {max_depth}")

    return best_action


# ────────────────────────────────────────────────
# Lancement de l’agent avec visualisation
# ────────────────────────────────────────────────

def make_env_main(render=False):
    return gym.make("CartPole-v1", render_mode="human" if render else None)


env_main = make_env_main(render=True)
obs, _ = env_main.reset()
state = env_main.unwrapped.state.copy()
root = Node(state)

done = False
total_reward = 0
step_count = 0

print("🚀 Début de l’épisode CartPole avec MCTS")

while not done:
    action = monte_carlo_tree_search(
        root,
        make_env=lambda: make_env_main(render=False),
        simulations=500,
        verbose=True
    )

    obs, reward, terminated, truncated, _ = env_main.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    print(f"\n✅ Step {step_count} — Action: {action}, Reward: {reward}, Total: {total_reward:.0f}")
    time.sleep(0.03)

    if action in root.children:
        root = root.children[action]
        root.parent = None
    else:
        root = Node(env_main.unwrapped.state.copy())

print(f"\n🏁 Épisode terminé — Total steps: {step_count}, Score final: {total_reward}")
env_main.close()
