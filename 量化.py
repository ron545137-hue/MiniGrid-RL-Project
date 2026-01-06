import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import torch as th
from torch import nn


# === 1. 定义我们之前的 CNN 特征提取器 (实验组) ===
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def train_model(model_type, log_dir, total_steps=300000):
    env_id = "MiniGrid-DoorKey-6x6-v0"

    # 根据模型类型选择不同的 Observation Wrapper
    if model_type == "MLP_Baseline":
        # 基准组：扁平化输入，丢失空间结构
        env = gym.make(env_id, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        policy_type = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))  # 简单的网络
        print(f"\n[开始训练] 基准组: MLP (FlatObs) - {env_id}")

    elif model_type == "CNN_Ours":
        # 实验组：图像输入，保留空间结构
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        policy_type = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        print(f"\n[开始训练] 实验组: CNN (ImgObs) - {env_id}")

    # 使用 Monitor 记录详细数据用于画图
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    model = PPO(
        policy_type,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=None
    )

    model.learn(total_timesteps=total_steps)
    env.close()
    print(f"[{model_type}] 训练完成！日志已保存至 {log_dir}")


def main():
    # 训练步数：为了快速看对比结果，先设为 30万步。
    # 如果想看完美效果，建议设为 50万 - 80万步。
    steps = 300000

    # 1. 训练基准模型 (MLP)
    train_model("MLP_Baseline", "logs/mlp_baseline", total_steps=steps)

    # 2. 训练我们的优化模型 (CNN)
    train_model("CNN_Ours", "logs/cnn_ours", total_steps=steps)

    print("\n所有对比实验已完成！请运行 plot_results.py 生成图表。")


if __name__ == "__main__":
    main()