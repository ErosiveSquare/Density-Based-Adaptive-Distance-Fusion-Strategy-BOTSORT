import numpy as np

def exp_rect(x, y, x0, y0, w, h, k=5):
    """
    基于指数衰减的快速近似
    - k: 控制衰减速度（越大则过渡越陡峭）
    """
    dx = np.abs(x - x0) - w/2
    dy = np.abs(y - y0) - h/2
    d = np.sqrt((np.maximum(dx, 0)**2) + (np.maximum(dy, 0)**2))
    return np.exp(-k * d)

def edge_gaussian_rect(x, y, x0, y0, w, h, sigma):
    """
    高斯衰减矩形函数（边缘密度固定为0.9）
    - sigma: 控制衰减宽度（越大衰减越平缓）
    - 自动保证矩形边缘密度为0.9，内部为1，外部按高斯衰减
    """
    # 计算到矩形边界的符号距离
    sigma = sigma*(np.minimum(w,h))
    dx = np.abs(x - x0) - w / 2
    dy = np.abs(y - y0) - h / 2

    # 判断点是否在矩形外部（包括边缘）
    outside_mask = (dx >= 0) | (dy >= 0)

    # 计算外部区域的衰减距离
    d_squared = (np.maximum(dx, 0) ** 2) + (np.maximum(dy, 0) ** 2)

    # 高斯衰减计算
    decay = np.exp(-d_squared / (2 * sigma ** 2))

    # 组合结果：内部1，边缘和外部0.9*衰减
    return np.where(outside_mask, 0.90 * decay, 1.0)


def smooth_gaussian_rect(x, y, x0, y0, w, h, sigma):
    """
    矩形内外均平滑处理的高斯衰减函数
    - sigma: 控制衰减宽度（越大衰减越平缓）
    - 内部从1平滑过渡到边缘0.9，外部从0.9按高斯衰减
    """
    # 计算到矩形边界的符号距离
    sigma = sigma * (np.minimum(w, h))
    dx = np.abs(x - x0) - w / 2
    dy = np.abs(y - y0) - h / 2

    # 计算内部区域的衰减距离（负值表示内部）
    d_internal = np.sqrt(np.maximum(-dx, 0) ** 2 + np.maximum(-dy, 0) ** 2)

    # 计算外部区域的衰减距离（正值表示外部）
    d_external = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)

    # 内部平滑处理：从1过渡到0.9
    internal_decay = 1 - 0.1 * np.exp(-d_internal ** 2 / (2 * sigma ** 2))

    # 外部平滑处理：从0.9按高斯衰减
    external_decay = 0.9 * np.exp(-d_external ** 2 / (2 * sigma ** 2))

    # 组合结果：内部平滑 + 外部衰减
    return np.where((dx < 0) & (dy < 0), internal_decay, external_decay)


def conserved_smooth_rect(x, y, x0, y0, w, h, k):
    """
    保积指数衰减矩形函数：矩形外函数积分等于矩形面积，k控制扩散程度
    - k: 衰减速度参数（越大衰减越快，扩散范围越小）
    """
    k/=(w*h)
    # 计算到矩形边界的符号距离
    dx = np.abs(x - x0) - w / 2
    dy = np.abs(y - y0) - h / 2

    # 计算到矩形边界的欧氏距离（外部区域）
    d = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)

    # 计算归一化因子C（确保外部积分=矩形面积）
    S = 6.18*(w * h)
    denominator = 2 * (k * (w + h) + np.pi)
    C = (S * k ** 2) / np.where(denominator != 0, denominator, 1e-10)  # 避免除以零

    # 组合结果：内部为0，外部按C*exp(-k*d)衰减
    return np.where((dx <= 0) & (dy <= 0), 1, C * np.exp(-k * d))



import matplotlib.pyplot as plt

# 参数设置
x0, y0 = 0, 0
w, h = 10, 24
sigma = 0.5
k = 1

# 生成网格
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)

# 计算三种方法的输出
Z_analytic_approx = smooth_gaussian_rect(X, Y, x0, y0, w, h, 1)
Z_scipy = edge_gaussian_rect(X, Y, x0, y0, w, h, 1)
Z_fast = conserved_smooth_rect(X, Y, x0, y0, w, h, k)

# 绘制结果s
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].contourf(X, Y, Z_analytic_approx, levels=50, cmap='viridis')
ax[0].set_title("Full_gaussian")
ax[1].contourf(X, Y, Z_scipy, levels=50, cmap='viridis')
ax[1].set_title("Edge_gaussian")
ax[2].contourf(X, Y, Z_fast, levels=50, cmap='viridis')
ax[2].set_title("conserved_smooth_rect")
plt.show()