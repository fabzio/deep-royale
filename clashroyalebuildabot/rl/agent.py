import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.hidden_states = [] 
        
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.hidden_states[:]

class ActorCritic(nn.Module):
    def __init__(self, action_dims, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        # Embeddings
        self.unit_id_emb = nn.Embedding(1000, 8)
        self.card_id_emb = nn.Embedding(1000, 8) # Using 1000 as hash space
        self.screen_emb = nn.Embedding(20, 4) # Approx screen count
        
        # Feature Extractors
        # Scalars: 5 -> 16
        self.scalar_fc = nn.Linear(5, 16)
        
        # Units: (ID_Emb(8) + 10 features) = 18 -> 16
        self.unit_fc = nn.Linear(18, 16)
        # We will flatten 32 units * 16 = 512
        self.units_compress = nn.Linear(512, 64)
        
        # Hand: (ID_Emb(8) + 2 features) = 10 -> 8
        self.hand_fc = nn.Linear(10, 8)
        # Flatten 4 cards * 8 = 32
        
        # Total Features: 16 (Scalar) + 64 (Allies) + 64 (Enemies) + 32 (Hand) + 4 (Screen) = 180
        self.total_input_dim = 16 + 64 + 64 + 32 + 4
        
        # LSTM
        self.lstm = nn.LSTM(self.total_input_dim, hidden_size, batch_first=True)
        
        # Actor Heads (MultiDiscrete: Card, X, Y)
        # action_dims = [4, 18, 32]
        self.actor_card = nn.Linear(hidden_size, action_dims[0])
        self.actor_x = nn.Linear(hidden_size, action_dims[1])
        self.actor_y = nn.Linear(hidden_size, action_dims[2])
        
        # Critic Head
        self.critic = nn.Linear(hidden_size, 1)
        
        self.hidden_size = hidden_size

    def _process_state(self, state):
        # Expect state to be dict with tensors
        # Or list of dicts if batched, but we handle tensor batching
        
        # Scalars
        scalars = state['scalars'] # (B, 5)
        x_scalars = F.relu(self.scalar_fc(scalars))
        
        # Helper for Units
        def process_units(units_tensor):
            # units_tensor: (B, 32, 11)
            ids = units_tensor[..., 0].long() # (B, 32)
            id_embs = self.unit_id_emb(ids.clamp(0, 999)) # (B, 32, 8)
            feats = units_tensor[..., 1:] # (B, 32, 10)
            x = torch.cat([id_embs, feats], dim=-1) # (B, 32, 18)
            x = F.relu(self.unit_fc(x)) # (B, 32, 16)
            x = x.view(x.size(0), -1) # Flatten -> (B, 512)
            return F.relu(self.units_compress(x)) # (B, 64)

        x_allies = process_units(state['allies'])
        x_enemies = process_units(state['enemies'])
        
        # Hand
        # hand: (B, 4, 3) -> [ID, Cost, Ready]
        h_ids = state['hand'][..., 0].long()
        h_id_embs = self.card_id_emb(h_ids.clamp(0, 999)) # (B, 4, 8)
        h_feats = state['hand'][..., 1:] # (B, 4, 2)
        x_hand = torch.cat([h_id_embs, h_feats], dim=-1) # (B, 4, 10)
        x_hand = F.relu(self.hand_fc(x_hand)) # (B, 4, 8)
        x_hand = x_hand.view(x_hand.size(0), -1) # (B, 32)
        
        # Screen
        scr = state['screen'].long() # (B,) or (B, 1)
        if scr.dim() > 1: scr = scr.squeeze(-1)
        x_screen = self.screen_emb(scr.clamp(0, 19)) # (B, 4)
        
        # Concatenate
        features = torch.cat([x_scalars, x_allies, x_enemies, x_hand, x_screen], dim=1)
        return features

    def forward(self, state, hidden=None):
        features = self._process_state(state) # (B, 180)
        
        # LSTM needs (Batch, Seq, Feats)
        # If we are processing single step with batch B:
        features_seq = features.unsqueeze(1) # (B, 1, 180)
        
        output, new_hidden = self.lstm(features_seq, hidden)
        output = output[:, -1, :] # Take last output (B, Hidden)
        
        return output, new_hidden

    def act(self, state, hidden):
        x, new_hidden = self.forward(state, hidden)
        
        # Heads
        logits_card = self.actor_card(x)
        logits_x = self.actor_x(x)
        logits_y = self.actor_y(x)
        
        dist_card = Categorical(logits=logits_card)
        dist_x = Categorical(logits=logits_x)
        dist_y = Categorical(logits=logits_y)
        
        action_card = dist_card.sample()
        action_x = dist_x.sample()
        action_y = dist_y.sample()
        
        action = torch.stack([action_card, action_x, action_y], dim=1)
        
        logprob = dist_card.log_prob(action_card) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        state_val = self.critic(x)
        
        return action.detach(), logprob.detach(), state_val.detach(), new_hidden

    def evaluate(self, state, action, hidden):
        # Process whole batch sequence if training? 
        # For simplicity in PPO with LSTM, complex BPTT handling is needed.
        # But for basics, we often treat rollout as batch.
        # If hidden is None/Initial, we assume batch processing.
        # For PPO update: we usually disable LSTM statefulness across batch OR handle sequences properly.
        # Here we will do simple stateless equivalent for update (using stored hidden states or just masking).
        # Actually, for PPO + LSTM update, passing (B, Seq, Feat) is best.
        # But our buffer stores flat steps.
        # Simplified: Pass features through LSTM (one big sequence or padded).
        
        # To keep this implementation correctly running without complex sequence padding logic:
        # We will reset LSTM hidden state for evaluation or carry it if we structured the buffer as trajectories.
        # For now, let's treat update as purely feed-forward using the stored hidden states? No, gradients won't flow through time.
        # Let's just do single-step forwarding in update with stored hidden states (Truncated BPTT of length 1).
        # It's sub-optimal but works for fundamental implementation.
        # Better: Recalculate features, pass to LSTM.
        
        x, _ = self.forward(state, hidden) # hidden here should be batched hidden states from rollout?
        # If we use 1-step BPTT, we use the hidden states stored in buffer as input.
        
        logits_card = self.actor_card(x)
        logits_x = self.actor_x(x)
        logits_y = self.actor_y(x)
        
        dist_card = Categorical(logits=logits_card)
        dist_x = Categorical(logits=logits_x)
        dist_y = Categorical(logits=logits_y)
        
        action_card = action[:, 0]
        action_x = action[:, 1]
        action_y = action[:, 2]
        
        logprobs = dist_card.log_prob(action_card) + dist_x.log_prob(action_x) + dist_y.log_prob(action_y)
        dist_entropy = dist_card.entropy() + dist_x.entropy() + dist_y.entropy()
        state_values = self.critic(x)
        
        return logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(self, action_dims=[4, 18, 32], lr=0.002, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(action_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(action_dims).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = RolloutBuffer()
        
        self.mse_loss = nn.MSELoss()
        
        # Current Hidden State for inference
        self.current_hidden = None

    def select_action(self, state):
        with torch.no_grad():
            # Convert state dict to tensors
            t_state = {}
            for k, v in state.items():
                if isinstance(v, (int, float)):
                    v = [v]
                v = np.array(v)
                t_state[k] = torch.FloatTensor(v).to(self.device).unsqueeze(0) # Add batch dim
            
            action, logprob, state_val, new_hidden = self.policy_old.act(t_state, self.current_hidden)
            
            # Store hidden state used to generate this action
            # For update, we might need it. Simpler: Update is "stateless" wrt LSTM (1-step).
            self.buffer.hidden_states.append(self.current_hidden)
            self.current_hidden = new_hidden
            
        self.buffer.states.append(t_state) # Store tensor dict (unbatched for list?)
        # t_state values are (1, ...) . Let's strip batch dim for storage or keep consistent.
        # Let's keep (1, ...) for easy concat later.
        
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(state_val)
        
        return action.cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Stack vars
        # States is list of dicts. We need to collate them.
        old_states = {}
        keys = self.buffer.states[0].keys()
        for k in keys:
             old_states[k] = torch.cat([s[k] for s in self.buffer.states], dim=0).to(self.device).detach()
        
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        
        # Hidden states: List of tuples (h, c). Each (1, 1, 128).
        # We need to detach them.
        # For simple PPO update here, we run forward on the whole batch. 
        # But LSTM state dependency? 
        # We will feed the recorded hidden states to the 'evaluate' call.
        # NOTE: This assumes 1-step logic for gradient. For full BPTT, we need sequence chunks.
        # This implementation does Truncated BPTT (k=1) implicitly by passing stored hidden states.
        
        # Gather hidden states
        h_list = []
        c_list = []
        for hidden in self.buffer.hidden_states:
             if hidden is None:
                 h_list.append(torch.zeros(1, 1, 128).to(self.device))
                 c_list.append(torch.zeros(1, 1, 128).to(self.device))
             else:
                 h_list.append(hidden[0])
                 c_list.append(hidden[1])
                 
        start_h = torch.cat(h_list, dim=1).detach() # (1, B, H) ? No.
        # LSTM hidden is (Layers, Batch, H). batch_first=True affects input, not hidden.
        # We are treating the rollout as a Batch of independent samples for the update (naive PPO-LSTM).
        # So we stack along Batch dimension.
        # stored hidden[0] is (1, 1, 128).
        old_h = torch.cat(h_list, dim=1).detach() # (1, N, 128)
        old_c = torch.cat(c_list, dim=1).detach() # (1, N, 128)
        old_hidden = (old_h, old_c)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_hidden)
            
            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - old_state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
    def reset_hidden(self):
        self.current_hidden = None
