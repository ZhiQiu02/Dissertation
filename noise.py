import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimSun'
# 生成正弦函数数据
def generate_sine(amplitude, period, noise_level, num_points):
    x = np.linspace(0, period * 2 * np.pi, num_points)  # 生成x轴数据
    y = amplitude * np.sin(x)  # 生成正弦波数据
    y_noise = noise_level * np.random.normal(size=num_points)  # 生成高斯噪声
    y_with_noise = y + y_noise  # 信号叠加噪声
    return x, y, y_noise, y_with_noise

# 参数设置
amplitude = 5
period = 200
noise_level = 5
num_points = 200

# 生成数据
x, y, y_noise, y_with_noise = generate_sine(amplitude, period, noise_level, num_points)

# 绘制正弦函数图
plt.figure(figsize=(4, 4))
plt.plot(x, y, label='正弦信号', color='blue')
plt.xlabel('时间 (s)')
plt.ylabel('振幅')
plt.legend()
plt.show()

# 绘制高斯噪声图
plt.figure(figsize=(4, 4))
plt.plot(x, y_noise, label='高斯噪声', color='green')
plt.xlabel('时间 (s)')
plt.ylabel('振幅')
plt.legend()
plt.show()

# 绘制信号叠加噪声图
plt.figure(figsize=(4, 4))
plt.plot(x, y_with_noise, label='融合信号', color='red')
plt.xlabel('时间 (s)')
plt.ylabel('振幅')
plt.legend()
plt.show()