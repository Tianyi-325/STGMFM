import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 不直接使用，但必须导入以启用3D绘图
import mne
import matplotlib

# 加载 montage
montage = mne.channels.make_standard_montage("standard_1020")
positions = montage.get_positions()["ch_pos"]

# 获取坐标和名称
x, y, z, names = [], [], [], []
for name, coord in positions.items():
    x.append(coord[0])
    y.append(coord[1])
    z.append(coord[2])
    names.append(name)

# 创建 3D 图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="blue", s=60)

# 标注电极名
for i in range(len(names)):
    ax.text(x[i], y[i], z[i], names[i], fontsize=8)

ax.set_title("EEG 10-20 System Electrodes (3D)")
plt.tight_layout()

# 显示为独立窗口
plt.show()


matplotlib.use("Qt5Agg")  # 或 TkAgg、WXAgg、GTK3Agg，看你系统支持哪个
