import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

def main():
    save_path = "minigrid_model"
    os.makedirs(save_path, exist_ok=True)

    # 环境：复杂的四房间导航
    env = gym.make("MiniGrid-FourRooms-v0", render_mode="rgb_array")
    env = FlatObsWrapper(env)
    env = Monitor(env)

    # 增加网络宽度以应对更复杂的决策
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003
    )

    print("开始训练导航版 (FourRooms)...")
    model.learn(total_timesteps=500000)

    model.save(os.path.join(save_path, "ppo_minigrid_four_rooms"))
    env.close()

if __name__ == "__main__":
    main()