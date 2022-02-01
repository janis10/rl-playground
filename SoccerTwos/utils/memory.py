import numpy as np
from collections import namedtuple
import torch.nn.functional as F


class Memory:
    def __init__(self):
        self.memory = []
        self.experience = namedtuple(
            "Experience",
            field_names=["actor_state", "critic_state", "action", "log_prob", "reward"],
        )

    def add(self, actor_state, critic_state, action, log_prob, reward):
        """Add a new experience to memory."""
        exp = self.experience(actor_state, critic_state, action, log_prob, reward)
        self.memory.append(exp)

    def experiences(self, clear=True):
        """Return experiences stored in memory"""
        # Number of experiences is the length of self.memory.
        n_exp = len(self.memory)
        # For each exp in self.memory, stack the
        # (actor_states, critic_states, actions, log_probabilities, rewards)
        actor_states = np.vstack(
            [exp.actor_state for exp in self.memory if exp is not None]
        )
        critic_states = np.vstack(
            [exp.critic_state for exp in self.memory if exp is not None]
        )
        actions = np.vstack([exp.action for exp in self.memory if exp is not None])
        log_probs = np.vstack([exp.log_prob for exp in self.memory if exp is not None])
        rewards = np.vstack([exp.reward for exp in self.memory if exp is not None])

        # Clear memory after returning experiences.
        if clear:
            self.memory.clear()

        return actor_states, critic_states, actions, log_probs, rewards, n_exp

    def delete(self, i):
        del self.memory[i]
