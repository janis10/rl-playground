import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Optimizer:
    def __init__(
        self,
        device,
        actor_model,
        critic_model,
        optimizer,
        n_step,
        batch_size,
        gamma,
        epsilon,
        entropy_weight,
        gradient_clip,
    ):
        # Set device
        self.device = device

        # Set neural nets
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer

        # Set hyperparameters
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.gradient_clip = gradient_clip

    def learn(self, memory):
        # Extract experiences from memory:
        (
            actor_states,
            critic_states,
            actions,
            log_probs,
            rewards,
            n_exp,
        ) = memory.experiences()

        # Discounts: gamma, gamma^2, gamma^3, ...
        discounts = self.gamma ** np.arange(n_exp)
        # Discount the rewards of the episode
        discounted_rewards = rewards.squeeze(1) * discounts
        # Compute the total discounted reward for the episode
        rewards_future = discounted_rewards[::-1].cumsum(axis=0)[::-1]

        # Setup torch tensors
        actor_states = torch.from_numpy(actor_states).float().to(self.device)
        critic_states = torch.from_numpy(critic_states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device).squeeze(1)
        log_probs = torch.from_numpy(log_probs).float().to(self.device).squeeze(1)
        rewards = torch.from_numpy(rewards_future.copy()).float().to(self.device)

        """
        We want the agent to take actions which achieve the greatest reward compared to the average expected reward 
        for that state (as estimated by our critic). We compute the advantage function
        below and normalize it to improve training.
        """
        # Get critic values detached from the training process (eval/inference only)
        self.critic_model.eval()
        with torch.no_grad():
            values = self.critic_model(critic_states).detach()
        self.critic_model.train()

        # Get advantages
        advantages = (rewards - values.squeeze()).detach()
        advantages_normalized = (advantages - advantages.mean()) / (
            advantages.std() + 1.0e-10
        )
        advantages_normalized = (
            torch.tensor(advantages_normalized).float().to(self.device)
        )

        """
        Each epoch has a set of experiences (n_exp). 
        We take a random mini-batch of experiences to train on.
        """
        batches = BatchSampler(
            SubsetRandomSampler(range(0, n_exp)), self.batch_size, drop_last=False
        )
        losses = []
        for batch_indices in batches:
            batch_indices = torch.tensor(batch_indices).long().to(self.device)

            # Get data from the batch
            sampled_actor_states = actor_states[batch_indices]
            sampled_critic_states = critic_states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs = log_probs[batch_indices]
            sampled_rewards = rewards[batch_indices]
            sampled_advantages = advantages_normalized[batch_indices]

            # Get new probability of each action given the state and latest actor policy
            _, new_log_probs, entropies = self.actor_model(
                sampled_actor_states, sampled_actions
            )

            # Compute ratio of how much more likely is the new action choice vs. old choice
            # according to the updated actor
            ratio = (new_log_probs - sampled_log_probs).exp()

            # Compute PPO loss
            """
            The clipping function makes sure that we don't update our weights too much when we find a much better 
            choice. This makes sure we do not charge in a false lead. 
            This is the key idea of Proximal Policy Optimization (PPO). 
            """
            clip = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = torch.min(
                ratio * sampled_advantages, clip * sampled_advantages
            )
            policy_loss = -torch.mean(policy_loss)

            """
            Entropy regularization term steers the new policy towards equal probability of all actions, encouraging
            exploration early on, but decreasing in importance over time. 
            """
            entropy = torch.mean(entropies)
            # Get predicted future rewards to use in backpropagation to improve the critic's estimates
            values = self.critic_model(sampled_critic_states)
            value_loss = F.mse_loss(sampled_rewards, values.squeeze())

            """
            The loss function combines the policy loss with value loss and adds the entropy term. PyTorch will
            backpropagate the respective losses through to each network's parameters and optimize over time.
            """
            loss = policy_loss + (0.5 * value_loss) - (entropy * self.entropy_weight)

            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_( self.actor_model.parameters(), self.GRADIENT_CLIP )
            # nn.utils.clip_grad_norm_( self.critic_model.parameters(), self.GRADIENT_CLIP )
            self.optimizer.step()

            losses.append(loss.data)

        self.epsilon *= 1
        self.entropy_weight *= 0.995

        return np.average(losses)
