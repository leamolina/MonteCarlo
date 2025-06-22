import gymnasium as gym
import copy
import math
import numpy as np
import time

# ────────────────────────────────────────────────────────
# 1. Classe Node : chaque nœud contient une copie de l'env
# ────────────────────────────────────────────────────────
class Node:
    def __init__(self, env, parent=None):
        self.env = env                      # copie de l’environnement (sans rendu)
        self.parent = parent
        self.children = {}                  # action -> Node
        self.visits = 0
        self.reward = 0.0
        self.depth = 0 if parent is None else parent.depth + 1

# ────────────────────────────────────────────────────────
# 2. Fonctions MCTS
# ────────────────────────────────────────────────────────
def fully_expanded(node):
    return len(node.children) == node.env.action_space.n

def expand(node):
    for action in range(node.env.action_space.n):
        if action not in node.children:
            env_copy = copy.deepcopy(node.env)
            env_copy.reset()  # 🔧 Nécessaire pour autoriser .step()
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            child = Node(env_copy, parent=node)
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

def rollout(node):
    env_sim = copy.deepcopy(node.env)
    env_sim.reset()  # 🔧 Nécessaire pour autoriser .step()
    done = False
    total_reward = 0
    while not done:
        action = env_sim.action_space.sample()
        _, reward, terminated, truncated, _ = env_sim.step(action)
        total_reward = reward  # Reward final (Blackjack)
        done = terminated or truncated
    return total_reward

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.reward += reward
        node = node.parent

def monte_carlo_tree_search(root, make_env, simulations=1000, c=1.4, verbose=False):
    all_rewards = []
    max_depth = 0

    for _ in range(simulations):
        node = root
        while fully_expanded(node):
            node = best_uct(node, c)
        leaf = expand(node) or node
        reward = rollout(leaf)
        backpropagate(leaf, reward)

        all_rewards.append(reward)
        max_depth = max(max_depth, leaf.depth)

    best_action = max(root.children, key=lambda a: root.children[a].visits)

    if verbose:
        print(f"\n🎯 Action choisie par MCTS : {best_action} ('{'stick' if best_action == 0 else 'hit'}')")
        for a, child in root.children.items():
            avg_r = child.reward / child.visits if child.visits else 0
            print(f"  Action {a}: visits={child.visits:4}, avg_reward={avg_r:+.3f}")
        print(f"📊 Simulations: {simulations},  Avg rollout reward={np.mean(all_rewards):+.3f},  Profondeur max={max_depth}")

    return best_action

# ────────────────────────────────────────────────────────
# 3. Fonction pour créer des environnements
# ────────────────────────────────────────────────────────
def make_env(render=False):
    return gym.make("Blackjack-v1", render_mode="human" if render else None)

# ────────────────────────────────────────────────────────
# 4. Boucle de jeu principale avec rendu
# ────────────────────────────────────────────────────────
env_main = make_env(render=True)
obs, _ = env_main.reset()

# Racine basée sur une copie silencieuse
root = Node(copy.deepcopy(make_env()))

done = False
total_reward = 0
step_idx = 0

print("\n🚀 Début de la partie Blackjack avec MCTS")

while not done:
    action = monte_carlo_tree_search(
        root,
        make_env=make_env,
        simulations=1000,
        verbose=True
    )

    obs, reward, terminated, truncated, _ = env_main.step(action)
    done = terminated or truncated
    step_idx += 1

    print(f"\n✅ Tour {step_idx} — Action jouée: {'stick' if action == 0 else 'hit'}")
    if done:
        print(f"🏁 Partie terminée — Reward final: {reward:+}")
        total_reward = reward
    else:
        print(f"   Jeu continue…")

    # Avancer dans l'arbre ou redémarrer
    if action in root.children:
        root = root.children[action]
        root.parent = None
    else:
        root = Node(copy.deepcopy(make_env()))

env_main.close()
print(f"\n🎉 Score total (gain par partie) : {total_reward:+}\n")
