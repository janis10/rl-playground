import torch
from utils.memory import Memory


class Agent:
    def __init__(self, device, key, actor_model, n_step):
        # Set device
        self.device = device
        # Set key
        self.KEY = key
        # Set neural model
        self.actor_model = actor_model
        # MEMORY
        self.memory = Memory()
        # Set number of steps
        self.N_STEP = n_step

    # Get an action from the actor for each step of game play.
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_model.eval()
        with torch.no_grad():
            action, log_prob, _ = self.actor_model(state)
        self.actor_model.train()
        action = action.cpu().detach().numpy().item()
        log_prob = log_prob.cpu().detach().numpy().item()
        return action, log_prob

    def step(self, actor_state, critic_state, action, log_prob, reward):
        self.memory.add(actor_state, critic_state, action, log_prob, reward)
