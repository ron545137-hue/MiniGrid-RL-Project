import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os


def main():
    save_path = "minigrid_model"
    os.makedirs(save_path, exist_ok=True)

    # 环境：简单的 8x8 空房间
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
    env = FlatObsWrapper(env)
    env = Monitor(env)

    # 使用基础的 MLP 策略
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    print("开始训练基础版 (Empty-8x8)...")
    model.learn(total_timesteps=100000)

    model.save(os.path.join(save_path, "ppo_minigrid_empty_8x8"))
    env.close()


if __name__ == "__main__":
    main()