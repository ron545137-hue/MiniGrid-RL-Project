import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os


def main():
    # 1. 指定进阶版模型路径
    # Path to the advanced model file
    model_path = "minigrid_model/ppo_minigrid_pro.zip"
    video_folder = "minigrid_pro_videos"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}。")
        print("请确保完成训练并生成该文件。")
        return

    # 2. 创建进阶环境
    # Creating the MultiRoom environment with high-definition settings
    env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode="rgb_array", tile_size=64)

    # 3. 添加视频录制包装器
    env = RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: True,
        name_prefix="multiroom-pro-demo"
    )

    # 4. 包装观测值
    # Standardize the observation space to match training
    env = FlatObsWrapper(env)

    # 5. 加载进阶模型
    # Load the trained PPO model
    print("正在加载进阶版模型并开始高清录制...")
    model = PPO.load(model_path)

    # 6. 运行演示并录制
    # Record 5 episodes of the agent navigating multiple rooms
    for i in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0

        print(f"正在录制第 {i + 1} 次进阶任务（多房间导航）...")

        while not (terminated or truncated):
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # 安全限制：防止在复杂地形中陷入死循环
            if step_count > 500:
                break

        print(f"第 {i + 1} 次录制完成！步数: {step_count}, 得分: {total_reward:.2f}")

    # 7. 关闭环境并保存视频
    # Close and finalize the video recording
    env.close()
    print(f"\n录制结束。演示视频已保存至: {os.path.abspath(video_folder)}")


if __name__ == "__main__":
    main()