import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import cwt, morlet

# 假设你的 EEG 数据形状为 (N, 64, 128)
N, C, T = 100, 64, 128  # 示例数据大小
eeg_data = np.random.randn(N, C, T)  # 生成随机EEG数据

# 定义小波参数
scales = np.arange(3, 105)  # 频率尺度
wavelet = morlet  # 使用 Morlet 小波

# 计算 CWT 并绘制时频图
def compute_cwt(signal):
    return cwt(signal, wavelet, scales)

# 选择一个 trial 和一个通道示例绘制
example_trial = 0
example_channel = 0

signal = eeg_data[example_trial, example_channel, :]
cwt_matrix = compute_cwt(signal)

# 绘制时频图
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(cwt_matrix), aspect='auto', cmap='jet', extent=[0, T, scales[-1], scales[0]])
plt.colorbar(label='Magnitude')
plt.xlabel("Time (samples)")
plt.ylabel("Frequency Scales")
plt.title(f"CWT Time-Frequency Map (Trial {example_trial}, Channel {example_channel})")
plt.show()