import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper  # 注意：这里改用图像包装器
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import os


# 自定义 CNN 特征提取器，适配 7x7 的小地图
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


def main():
    save_path = "minigrid_model"
    os.makedirs(save_path, exist_ok=True)

    # 环境：需要拿钥匙开门的 DoorKey 环境
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = ImgObsWrapper(env)  # 关键：保留图像信息
    env = Monitor(env)

    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, ent_coef=0.01)

    print("开始训练逻辑版 (DoorKey)...")
    model.learn(total_timesteps=300000)

    model.save(os.path.join(save_path, "ppo_minigrid_doorkey_cnn"))
    env.close()


if __name__ == "__main__":
    main()