import time
import gymnasium as gym
import numpy as np
import torch
from clashroyalebuildabot.bot import Bot
from clashroyalebuildabot.gui.utils import load_config
from clashroyalebuildabot.gym.env import ClashRoyaleEnv
from clashroyalebuildabot.rl.agent import PPOAgent
from clashroyalebuildabot.actions import (
    BabyDragonAction,
    FireballAction,
    GiantAction,
    KnightAction,
    ZapAction,
)
from clashroyalebuildabot.actions.archers_action import ArchersAction
from clashroyalebuildabot.actions.musketeer_action import MusketeerAction
from clashroyalebuildabot.actions.skeletons_action import SkeletonsActions

def main():
    # 1. Load Config
    config = load_config()
    
    # 2. Define Actions (Deck)
    actions = [
        ZapAction,
        FireballAction,
        BabyDragonAction,
        SkeletonsActions,
        KnightAction,
        ArchersAction,
        MusketeerAction,
        GiantAction,
    ]
    
    # 3. Initialize Bot
    bot = Bot(actions=actions, config=config)
    
    # 4. Initialize Env
    # play_action_delay: Time to wait after clicking. 
    env = ClashRoyaleEnv(bot, play_action_delay=0.5)
    
    # 5. Initialize PPO Agent
    # Action dims are [Card(4), TileX(18), TileY(32)]
    agent = PPOAgent(action_dims=[4, 18, 32], lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2)
    
    print("Environment and PPO Agent initialized.")
    
    # Training Hyperparameters
    max_training_timesteps = 100000
    max_ep_len = 1000
    update_timestep = 2000 # Update policy every n timesteps
    save_model_freq = 5000
    
    time_step = 0
    i_episode = 0
    
    # Load model if exists?
    # agent.policy.load_state_dict(torch.load("ppo_model.pth"))
    
    try:
        while time_step < max_training_timesteps:
            state, info = env.reset()
            agent.reset_hidden() # Reset LSTM hidden state at start of episode
            current_ep_reward = 0
            
            for t in range(1, max_ep_len+1):
                # Select action with policy
                action = agent.select_action(state)
                
                # Execute action
                state, reward, terminated, truncated, info = env.step(action)
                
                # Save reward and is_terminals
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(terminated)
                
                time_step += 1
                current_ep_reward += reward
                
                # Update PPO agent
                if time_step % update_timestep == 0:
                    print(f"Updating Policy at timestep {time_step}...")
                    agent.update()
                    agent.reset_hidden() # Reset after update since we cleared buffer history
                    
                # Save Model
                if time_step % save_model_freq == 0:
                    print(f"Saving model at timestep {time_step}...")
                    torch.save(agent.policy.state_dict(), "ppo_agent.pth")
                    
                if terminated or truncated:
                    break
            
            i_episode += 1
            print(f"Episode {i_episode} finished. Reward: {current_ep_reward:.2f}. Total Timesteps: {time_step}")
            
    except KeyboardInterrupt:
        print("Stopping training...")
        torch.save(agent.policy.state_dict(), "ppo_agent_interrupted.pth")
    finally:
        bot.stop()
        env.close()

if __name__ == "__main__":
    main()
