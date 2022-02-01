# COMMENT: It would be even better if I 'function-ize' the below for each agent.
# Now since we have only two agents (goalie, striker) that we train, it was easy to code.
# But if I want to have more, I should make nice functions.

import numpy as np
from collections import deque


class Trainer:
    def __init__(self, env, device, epochs, g_agent, g_optimizer, s_agent, s_optimizer):
        # Set environment
        self.env = env
        # Set device
        self.device = device
        # Set agents
        self.goalie_0 = g_agent
        self.goalie_optimizer = g_optimizer
        self.striker_0 = s_agent
        self.striker_optimizer = s_optimizer
        # Set number of epochs
        self.n_epochs = epochs

    def ppo_train(self, GOALIE_1_KEY, STRIKER_1_KEY):

        # CHECKPOINT_GOALIE_ACTOR = self.goalie_optimizer.actor_model.checkpoint_path
        # print(CHECKPOINT_GOALIE_ACTOR)
        # CHECKPOINT_GOALIE_CRITIC = self.goalie_optimizer.critic_model.checkpoint_path
        # CHECKPOINT_STRIKER_ACTOR = self.striker_optimizer.actor_model.checkpoint_path
        # CHECKPOINT_STRIKER_CRITIC = self.striker_optimizer.critic_model.checkpoint_path

        # Get info from environment
        # set the goalie brain
        g_brain_name = self.env.brain_names[0]
        g_brain = self.env.brains[g_brain_name]
        # set the striker brain
        s_brain_name = self.env.brain_names[1]
        s_brain = self.env.brains[s_brain_name]

        # Reset the environment
        env_info = self.env.reset(train_mode=True)
        # Goalie info
        num_g_agents = len(env_info[g_brain_name].agents)
        g_action_size = g_brain.vector_action_space_size
        g_states = env_info[g_brain_name].vector_observations

        # Striker info
        num_s_agents = len(env_info[s_brain_name].agents)
        s_action_size = s_brain.vector_action_space_size
        s_states = env_info[s_brain_name].vector_observations

        # For score storing
        team_0_window_score = deque(maxlen=100)
        team_0_window_score_wins = deque(maxlen=100)
        team_1_window_score = deque(maxlen=100)
        team_1_window_score_wins = deque(maxlen=100)
        draws = deque(maxlen=100)

        """
        We generate episodes via the simulation environment. Then we use these episodes to train the agents.
        """
        for episode in range(self.n_epochs):
            # Reset the environment for each new episode
            env_info = self.env.reset(train_mode=True)
            # Get initial states
            g_states = env_info[g_brain_name].vector_observations
            s_states = env_info[s_brain_name].vector_observations
            # Initialize scores
            g_scores = np.zeros(num_g_agents)
            s_scores = np.zeros(num_s_agents)

            # Execute episode
            steps = 0
            while True:
                # Select actions and send to environment
                # The actions are selected by team 0 agents
                action_g_0, log_prob_g_0 = self.goalie_0.act(
                    g_states[self.goalie_0.KEY]
                )
                action_s_0, log_prob_s_0 = self.striker_0.act(
                    s_states[self.striker_0.KEY]
                )

                # Sample action for team 1
                action_g_1 = np.asarray([np.random.choice(g_action_size)])
                action_s_1 = np.asarray([np.random.choice(s_action_size)])
                # # if I had agents for team 1 I could use them
                # action_g_1, log_prob_g_1 = goalie_1.act( g_states[goalie_1.KEY] )
                # action_s_1, log_prob_s_1 = striker_1.act( s_states[striker_1.KEY] )

                # Store actions:
                # environment takes the input as a dict
                # {brain_name : set_of_associated_actions}
                g_actions = np.array((action_g_0, action_g_1))
                s_actions = np.array((action_s_0, action_s_1))
                actions = dict(
                    zip([g_brain_name, s_brain_name], [g_actions, s_actions])
                )

                # Take one step using the selected actions
                env_info = self.env.step(actions)
                # and get the updated info
                g_next_states = env_info[g_brain_name].vector_observations
                s_next_states = env_info[s_brain_name].vector_observations
                g_rewards = env_info[g_brain_name].rewards
                s_rewards = env_info[s_brain_name].rewards
                g_scores += g_rewards
                s_scores += s_rewards

                # Store experiences
                g_0_reward = g_rewards[self.goalie_0.KEY]
                self.goalie_0.step(
                    g_states[self.goalie_0.KEY],
                    np.concatenate(
                        (
                            g_states[self.goalie_0.KEY],
                            s_states[self.striker_0.KEY],
                            g_states[GOALIE_1_KEY],
                            s_states[STRIKER_1_KEY],
                        ),
                        axis=0,
                    ),
                    action_g_0,
                    log_prob_g_0,
                    g_0_reward,
                )

                s_0_reward = s_rewards[self.striker_0.KEY]
                self.striker_0.step(
                    s_states[self.striker_0.KEY],
                    np.concatenate(
                        (
                            s_states[self.striker_0.KEY],
                            g_states[self.goalie_0.KEY],
                            s_states[STRIKER_1_KEY],
                            g_states[GOALIE_1_KEY],
                        ),
                        axis=0,
                    ),
                    action_s_0,
                    log_prob_s_0,
                    s_0_reward,
                )

                # Check if episode finished
                if np.any(env_info[g_brain_name].local_done):
                    break

                # Update states
                g_states = g_next_states
                s_states = s_next_states

                steps += 1

            # Learn from the generated episodes
            # Each agent has stored in 'agent'.memory
            # the experiences (actor_state, critic_state, action, log_prob, reward)
            # we generated via the simulation
            goalie_loss = self.goalie_optimizer.learn(self.goalie_0.memory)
            striker_loss = self.striker_optimizer.learn(self.striker_0.memory)

            # Save the updated model after the episode
            self.goalie_optimizer.actor_model.checkpoint()
            self.goalie_optimizer.critic_model.checkpoint()
            self.striker_optimizer.actor_model.checkpoint()
            self.striker_optimizer.critic_model.checkpoint()
            # goalie_actor_model.checkpoint(CHECKPOINT_GOALIE_ACTOR)
            # goalie_critic_model.checkpoint(CHECKPOINT_GOALIE_CRITIC)
            # striker_actor_model.checkpoint(CHECKPOINT_STRIKER_ACTOR)
            # striker_critic_model.checkpoint(CHECKPOINT_STRIKER_CRITIC)

            # Update scores for each team
            team_0_score = g_scores[self.goalie_0.KEY] + s_scores[self.striker_0.KEY]
            team_0_window_score.append(team_0_score)
            team_0_window_score_wins.append(1 if team_0_score > 0 else 0)

            team_1_score = g_scores[GOALIE_1_KEY] + s_scores[STRIKER_1_KEY]
            team_1_window_score.append(team_1_score)
            team_1_window_score_wins.append(1 if team_1_score > 0 else 0)

            draws.append(team_0_score == team_1_score)

            # Print updates
            print(
                "Episode: {} \tSteps: \t{} \tGoalie Loss: \t {:.10f} \tStriker Loss: \t {:.10f}".format(
                    episode + 1, steps, goalie_loss, striker_loss
                )
            )
            print(
                "\tRed Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}".format(
                    np.count_nonzero(team_0_window_score_wins),
                    team_0_score,
                    np.sum(team_0_window_score),
                )
            )
            print(
                "\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}".format(
                    np.count_nonzero(team_1_window_score_wins),
                    team_1_score,
                    np.sum(team_1_window_score),
                )
            )
            print("\tDraws: \t{}".format(np.count_nonzero(draws)))

            if np.count_nonzero(team_0_window_score_wins) >= 95:
                break

    # test the trained agents
    def test(self, GOALIE_1_KEY, STRIKER_1_KEY, n_episodes=50):
        # Get info from environment
        # set the goalie brain
        g_brain_name = self.env.brain_names[0]
        g_brain = self.env.brains[g_brain_name]
        # set the striker brain
        s_brain_name = self.env.brain_names[1]
        s_brain = self.env.brains[s_brain_name]

        # Reset the environment
        env_info = self.env.reset(train_mode=True)
        # Goalie info
        num_g_agents = len(env_info[g_brain_name].agents)
        g_action_size = g_brain.vector_action_space_size
        g_states = env_info[g_brain_name].vector_observations

        # Striker info
        num_s_agents = len(env_info[s_brain_name].agents)
        s_action_size = s_brain.vector_action_space_size
        s_states = env_info[s_brain_name].vector_observations

        # For score storing
        team_0_window_score = deque(maxlen=100)
        team_0_window_score_wins = deque(maxlen=100)
        team_1_window_score = deque(maxlen=100)
        team_1_window_score_wins = deque(maxlen=100)
        draws = deque(maxlen=100)

        for episode in range(n_episodes):
            # Reset the environment for each new episode
            env_info = self.env.reset(train_mode=False)
            # Get initial states
            g_states = env_info[g_brain_name].vector_observations
            s_states = env_info[s_brain_name].vector_observations
            # Initialize scores
            g_scores = np.zeros(num_g_agents)
            s_scores = np.zeros(num_s_agents)

            # Execute episode
            steps = 0
            while True:
                # Select actions and send to environment
                # The actions are selected by team 0 agents
                action_g_0, _ = self.goalie_0.act(g_states[self.goalie_0.KEY])
                action_s_0, _ = self.striker_0.act(s_states[self.striker_0.KEY])

                # Sample action for team 1
                action_g_1 = np.asarray([np.random.choice(g_action_size)])
                action_s_1 = np.asarray([np.random.choice(s_action_size)])
                # # if I had agents for team 1 I could use them
                # action_g_1, log_prob_g_1 = goalie_1.act( g_states[goalie_1.KEY] )
                # action_s_1, log_prob_s_1 = striker_1.act( s_states[striker_1.KEY] )

                # Store actions:
                # environment takes the input as a dict
                # {brain_name : set_of_associated_actions}
                g_actions = np.array((action_g_0, action_g_1))
                s_actions = np.array((action_s_0, action_s_1))
                actions = dict(
                    zip([g_brain_name, s_brain_name], [g_actions, s_actions])
                )

                # Take one step using the selected actions
                env_info = self.env.step(actions)
                # and get the updated info
                g_next_states = env_info[g_brain_name].vector_observations
                s_next_states = env_info[s_brain_name].vector_observations
                g_rewards = env_info[g_brain_name].rewards
                s_rewards = env_info[s_brain_name].rewards
                g_scores += g_rewards
                s_scores += s_rewards

                # Check if episode finished
                if np.any(env_info[g_brain_name].local_done):
                    break

                # Update states
                g_states = g_next_states
                s_states = s_next_states

                steps += 1

            team_0_score = g_scores[self.goalie_0.KEY] + s_scores[self.striker_0.KEY]
            team_0_window_score.append(team_0_score)
            team_0_window_score_wins.append(1 if team_0_score > 0 else 0)

            team_1_score = g_scores[GOALIE_1_KEY] + s_scores[STRIKER_1_KEY]
            team_1_window_score.append(team_1_score)
            team_1_window_score_wins.append(1 if team_1_score > 0 else 0)

            draws.append(team_0_score == team_1_score)

            print("Episode {}".format(episode + 1))
            print(
                "\tRed Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}".format(
                    np.count_nonzero(team_0_window_score_wins),
                    team_0_score,
                    np.sum(team_0_window_score),
                )
            )
            print(
                "\tBlue Wins: \t{} \tScore: \t{:.5f} \tAvg: \t{:.2f}".format(
                    np.count_nonzero(team_1_window_score_wins),
                    team_1_score,
                    np.sum(team_1_window_score),
                )
            )
            print("\tDraws: \t{}".format(np.count_nonzero(draws)))
