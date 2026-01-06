import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os


def main():
    # 1. 指定模型路径
    # Set the path to the trained model file
    model_path = "minigrid_model/ppo_minigrid_empty_8x8.zip"
    video_folder = "minigrid_empty_videos"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请先运行训练脚本。")
        return

    # 2. 创建环境
    # 使用 rgb_array 模式进行高清视频录制，设置 tile_size=64 提升画质
    # Create the environment with rgb_array mode for video recording
    env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array", tile_size=64)

    # 3. 添加视频录制包装器
    # Add RecordVideo wrapper to save MP4 files
    env = RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: True,
        name_prefix="empty-8x8-demo"
    )

    # 4. 包装观测值
    # Use FlatObsWrapper to match the training configuration
    env = FlatObsWrapper(env)

    # 5. 加载模型
    # Load the trained PPO model
    print("正在加载模型并准备录制...")
    model = PPO.load(model_path)

    # 6. 运行演示并录制
    # Run the demonstration for 5 episodes
    for i in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        step_count = 0

        print(f"正在录制第 {i + 1} 次任务...")

        while not (terminated or truncated):
            # 使用模型预测动作
            # Predict the best action using the loaded model
            action, _ = model.predict(obs, deterministic=True)
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        print(f"第 {i + 1} 次录制完成！耗时 {step_count} 步。")

    # 7. 关闭环境
    # Close the environment to finalize video saving
    env.close()
    print(f"\n录制结束。视频已保存至: {os.path.abspath(video_folder)}")


if __name__ == "__main__":
    main()