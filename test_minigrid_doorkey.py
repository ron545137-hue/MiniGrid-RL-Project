import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import torch as th
from torch import nn
from gymnasium.wrappers import RecordVideo


# 必须包含与训练时完全一致的特征提取器定义
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
    model_path = "minigrid_model/ppo_minigrid_doorkey_cnn.zip"
    video_folder = "minigrid_videos_hd"

    if not os.path.exists(model_path):
        print("错误: 找不到模型文件。")
        return

    # 1. 创建环境并提高 tile_size
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array", tile_size=64)

    # 2. 添加视频录制包装器
    env = RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: True,
        name_prefix="doorkey-hd"
    )

    env = ImgObsWrapper(env)

    # 3. 加载模型
    print("正在加载模型")
    model = PPO.load(model_path, env=env)

    # 4. 运行演示
    for episode in range(3):
        obs, info = env.reset()
        terminated = False
        truncated = False

        print(f"正在录制第 {episode + 1} 局 (高清模式)...")

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

    # 5. 关闭环境
    env.close()
    print(f"\n高清录制完成！视频已保存至: {os.path.abspath(video_folder)}")


if __name__ == "__main__":
    main()