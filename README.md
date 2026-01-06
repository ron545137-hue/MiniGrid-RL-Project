MiniGrid Reinforcement Learning Project

本项目研究了 PPO 算法在稀疏奖励环境（MiniGrid）下的表现，重点探讨了感知架构（CNN vs MLP）对复杂逻辑任务（DoorKey）的影响。

实验结论

通过对比实验发现，自定义的轻量级 CNN 特征提取器能有效保留网格空间结构，在 30 万步内成功解决了需要逻辑交互的任务，而传统的 MLP 架构在相同条件下难以收敛。

环境要求

Python 3.10+

gymnasium

minigrid

stable-baselines3

moviepy (用于视频录制)

代码介绍：

生成图表: 运行 python plot_results.py

查看演示: 运行 python test_minigrid_hd.py

训练过程 ：
Level 1: Basic Navigation (Empty Room)

Focus: Understanding the basic PPO training loop and reward signals.

Observation: Flat observation.

Level 2: Logical Reasoning (DoorKey)

Focus: Solving causal dependencies (Pick up key -> Open door).

Innovation: Custom CNN feature extractor was designed to handle 7x7 spatial information.

Level 3: Long-term Planning (Multi-Room)

Focus: Navigating through multiple connected compartments.

Scale: Increased training steps to 500k to ensure stable policy convergence.