#!/usr/bin/env python3
"""
Monte Carlo
"""
import numpy as np


def generate_episode(env, policy, max_steps):
    episode = [[], []]
    state = env.reset()[0]

    for step in range(max_steps):
        action = policy(state)
        next_state, reward, done, truncate, info = env.step(action)
        episode[0].append(state)

        if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
            episode[1].append(-1)
            return episode

        if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
            episode[1].append(1)
            return episode

        episode[1].append(0)

    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):

    discounts = np.array([gamma ** i for i in range(max_steps)])
    for ep in range(episodes):
        episode = generate_episode(env, policy, max_steps)

        for i in range(len(episode[0])):
            Gt = np.sum(np.array(episode[1][i:]) *
                        np.array(discounts[:len(episode[1][i:])]))
            V[episode[0][i]] = (V[episode[0][i]] +
                                alpha * (Gt - V[episode[0][i]]))
    return V
