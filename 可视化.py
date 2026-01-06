import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def read_monitor_files(log_dir):
    all_data = []
    # 读取 stable_baselines3 生成的 monitor.csv
    monitor_file = os.path.join(log_dir, "monitor.csv")

    if not os.path.exists(monitor_file):
        print(f"警告: 找不到日志文件 {monitor_file}")
        return None

    # 跳过前两行元数据
    df = pd.read_csv(monitor_file, skiprows=1)
    return df


def moving_average(values, window):
    """ 平滑曲线，让趋势更明显 """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def main():
    # 定义实验组名称和对应的颜色
    experiments = {
        "MLP Baseline (FlatObs)": {"path": "logs/mlp_baseline", "color": "gray", "style": "--"},
        "CNN Pro (ImgObs)": {"path": "logs/cnn_ours", "color": "red", "style": "-"}
    }

    plt.figure(figsize=(10, 6))

    for label, config in experiments.items():
        df = read_monitor_files(config["path"])

        if df is not None and not df.empty:
            # 计算累积步数
            # 'l' 列是 episode length (步数), 'r' 列是 reward (奖励)
            cumulative_steps = np.cumsum(df['l'])
            rewards = df['r']

            # 对奖励进行平滑处理 (窗口大小 50)
            window_size = 50
            if len(rewards) > window_size:
                smooth_rewards = moving_average(rewards, window_size)
                # 调整 x 轴长度以匹配平滑后的 y 轴
                smooth_steps = cumulative_steps[window_size - 1:]

                plt.plot(smooth_steps, smooth_rewards,
                         label=label,
                         color=config["color"],
                         linestyle=config["style"],
                         linewidth=2)
            else:
                print(f"数据量不足以进行平滑处理: {label}")

    plt.title("MiniGrid-DoorKey Training Performance: MLP vs CNN", fontsize=14)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Average Reward (Smoothed)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存图表
    plt.savefig("result_comparison.png", dpi=300)
    print("图表已保存为 'result_comparison.png'，请打开查看！")
    plt.show()


if __name__ == "__main__":
    main()