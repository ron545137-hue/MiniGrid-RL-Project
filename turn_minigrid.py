import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import torch as th
from torch import nn


# 自定义特征提取器，专门处理 MiniGrid 的 7x7 图像
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # 针对 7x7 的输入，我们使用更小的卷积核 (如 3x3)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算卷积后的输出维度
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def main():
    # 1. 创建模型保存文件夹
    save_path = "minigrid_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. 调整环境：先从 6x6 开始
    env_id = "MiniGrid-DoorKey-6x6-v0"
    env = gym.make(env_id, render_mode="rgb_array")

    # 3. 包装环境
    env = ImgObsWrapper(env)
    env = Monitor(env)

    # 4. 配置 PPO 算法
    # 使用自定义的特征提取器替代默认的 NatureCNN
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./minigrid_tensorboard/"
    )

    # 5. 开始训练过程
    total_steps = 1000000
    print(f"优化版训练：正在挑战 DoorKey ({env_id})")
    print(f"使用自定义 CNN 适配 7x7 输入，目标步数: {total_steps}")

    model.learn(total_timesteps=total_steps)

    print("训练完成！")

    # 6. 保存最终模型
    model_name = "ppo_minigrid_doorkey_cnn"
    model.save(os.path.join(save_path, model_name))
    print(f"模型已保存至: {save_path}/{model_name}.zip")

    # 7. 评估
    print("正在评估训练成果...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"评估结果 - 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()