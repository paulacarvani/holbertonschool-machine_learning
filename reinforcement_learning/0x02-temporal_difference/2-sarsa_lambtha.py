#!/usr/bin/env python3
"""
Defines function to perform the SARSA(λ) algorithm
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    max_epsilon = epsilon
    Et = np.zeros((Q.shape))

    for ep in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            Et = Et * lambtha * gamma
            Et[state, action] += 1
            next_state, reward, done, trunc, info = env.step(action)
            next_action = epsilon_greedy(Q, state, epsilon)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1

            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            delta_t = reward + (
                gamma * Q[next_state, next_action]
            ) - Q[state, action]

            Q[state, action] = Q[state, action] + (
                alpha * delta_t * Et[state, action])

            if done:
                break
            state = next_state
            action = next_action

        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep))

    return Q
