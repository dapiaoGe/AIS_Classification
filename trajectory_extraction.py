import os
from collections import Counter
import time

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


##### 提取AIS数据
def extract_trajectories(file_path):
    """
    从CSV文件中提取轨迹数据、标签和MMSI。

    参数:
    file_path (str): CSV文件的路径。

    返回:
    X (list): 包含每艘船的轨迹特征列表。
    Y (list): 包含每艘船的类别标签列表。
    MMSIs (list): 包含每艘船的MMSI列表。
    """

    # 读取数据
    df = pd.read_csv(file_path)

    # 定义类别映射
    ship_type_mapping = {
        'Cargo': 0,
        'Fishing': 1,
        'Tanker': 2,
        'Passenger': 3,
        'Military': 4
    }

    # 应用类别映射
    df['Ship type'] = df['Ship type'].map(ship_type_mapping)

    # 提取特征
    features = ['Latitude', 'Longitude', 'SOG', 'COG', 'Heading', 'Width', 'Length', 'Latitude_change',
                'Longitude_change']

    # 创建空列表来存储轨迹、标签和MMSI
    trajectories = []
    labels = []
    MMSIs = []

    # 将数据分组为每艘船的轨迹
    for mmsi, group in df.groupby('MMSI'):
        # 对于每艘船，按时间顺序排列数据
        sorted_group = group.sort_values(by='# Timestamp')

        # 构建轨迹
        trajectory = sorted_group[features].values.tolist()

        # 添加轨迹到列表
        trajectories.append(trajectory)

        # 获取该船的类别作为标签
        label = sorted_group['Ship type'].iloc[0]  # 使用第一个观测的类别作为标签
        labels.append(label)

        # 保存MMSI
        MMSIs.append(mmsi)

    return trajectories, labels, MMSIs


##### 提取AIS数据和图像
# def save_or_load_trajectory_image(latitudes, longitudes, mmsi, image_save_path, image_size=(224, 224)):
#     """
#     生成轨迹图像或者从已有图像中加载，如果图像已存在，则不重新生成。
#
#     参数:
#     latitudes (pd.Series): 船的纬度数据。
#     longitudes (pd.Series): 船的经度数据。
#     mmsi (int): 船的MMSI号，用于命名图像文件。
#     image_save_path (str): 图像保存路径。
#     image_size (tuple): 图像大小。
#
#     返回:
#     image_array (ndarray): 图像的数值矩阵，用于模型输入。
#     """
#     # 创建图像保存目录
#     if not os.path.exists(image_save_path):
#         os.makedirs(image_save_path)
#
#     # 图像文件名
#     image_filename = os.path.join(image_save_path, f'{int(mmsi)}.jpg')
#
#     # 检查图像文件是否已经存在
#     if os.path.exists(image_filename):
#         # 如果图像已经存在，则加载图像
#         image = Image.open(image_filename).resize(image_size)
#         image_array = np.array(image)
#     else:
#         # 否则生成并保存轨迹图像
#         plt.figure(figsize=(8, 6))
#         plt.plot(longitudes, latitudes, marker='o', markersize=2, linestyle='-', color='blue')
#         plt.title(f'Ship MMSI: {mmsi}')
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
#
#         # 保存图像
#         plt.savefig(image_filename)
#         plt.close()
#
#         # 加载保存后的图像并调整大小
#         image = Image.open(image_filename).resize(image_size)
#         image_array = np.array(image)
#
#     return image_array

# 初始化全局变量来跟踪每个MMSI的文件序号
mmsi_counters = {}


def save_or_load_trajectory_image(latitudes, longitudes, mmsi, image_save_path, image_size=(224, 224)):
    # 确保保存目录存在
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    # 获取当前MMSI的计数器，如果没有则初始化为0
    if mmsi not in mmsi_counters:
        mmsi_counters[mmsi] = 0
        # 检查已存在的最大编号
        existing_files = [f for f in os.listdir(image_save_path) if
                          f.startswith(f'{int(mmsi)}_') and f.endswith('.jpg')]
        if existing_files:
            max_counter = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
            mmsi_counters[mmsi] = max_counter + 1

    # 增加计数器并创建文件名
    while True:
        file_counter = mmsi_counters[mmsi]
        image_filename = os.path.join(image_save_path, f'{int(mmsi)}_{file_counter}.jpg')

        # 如果文件不存在，则生成新的图像并保存
        if not os.path.isfile(image_filename):
            plt.figure(figsize=(8, 6))
            plt.plot(longitudes, latitudes, marker='o', markersize=2, linestyle='-', color='blue')
            # 关闭坐标轴，从而移除xlabel和ylabel
            plt.axis('off')
            plt.savefig(image_filename)
            plt.close()
            break  # 文件已保存，跳出循环

        # 如果文件存在，递增计数器并尝试下一个编号
        mmsi_counters[mmsi] += 1

    # 加载刚刚保存或已存在的图像并调整大小
    image = Image.open(image_filename).resize(image_size)
    image_array = np.array(image)

    return image_array

########################## 速度集成到图像
def save_or_load_trajectory_image_v(latitudes, longitudes, speeds, mmsi, image_save_path,
                                  image_size=(224, 224),
                                  global_min_speed=0, global_max_speed=20,
                                  cmap_name='viridis'):
    """
    生成或加载包含速度信息的轨迹图像，使用统一的颜色映射。

    参数:
        latitudes (list or array): 纬度列表。
        longitudes (list or array): 经度列表。
        speeds (list or array): 速度列表，与经纬度对应。
        mmsi (int or str): MMSI标识符。
        image_save_path (str): 图像保存路径。
        image_size (tuple): 图像大小，默认为(224, 224)。
        global_min_speed (float): 全局最小速度，用于统一归一化。
        global_max_speed (float): 全局最大速度，用于统一归一化。
        cmap_name (str): 颜色映射名称，默认为'viridis'。

    返回:
        numpy.ndarray: 调整大小后的图像数组。
    """
    # 确保保存目录存在
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    # 获取当前MMSI的计数器，如果没有则初始化为0
    if mmsi not in mmsi_counters:
        mmsi_counters[mmsi] = 0
        # 检查已存在的最大编号
        existing_files = [f for f in os.listdir(image_save_path) if
                          f.startswith(f'{int(mmsi)}_') and f.endswith('.jpg')]
        if existing_files:
            max_counter = max([int(f.split('_')[1].split('.')[0]) for f in existing_files])
            mmsi_counters[mmsi] = max_counter + 1

    # 增加计数器并创建文件名
    while True:
        file_counter = mmsi_counters[mmsi]
        image_filename = os.path.join(image_save_path, f'{int(mmsi)}_{file_counter}.jpg')

        # 如果文件不存在，则生成新的图像并保存
        if not os.path.isfile(image_filename):
            plt.figure(figsize=(8, 6))

            # 准备轨迹点和速度
            points = np.array([longitudes, latitudes]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 使用全局最小和最大速度进行归一化
            norm = plt.Normalize(global_min_speed, global_max_speed)

            cmap = plt.get_cmap(cmap_name)  # 选择颜色映射，如'jet'或'viridis'
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(speeds)
            lc.set_linewidth(2)

            # 添加LineCollection到图中
            plt.gca().add_collection(lc)

            # 添加颜色条
            cbar = plt.colorbar(lc, label='速度 (单位)')  # 根据实际速度单位修改标签

            # 设置坐标范围
            plt.xlim(min(longitudes) - 0.01, max(longitudes) + 0.01)
            plt.ylim(min(latitudes) - 0.01, max(latitudes) + 0.01)

            # 移除坐标轴
            plt.axis('off')

            # 保存图像，去除多余的边缘
            plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            break  # 文件已保存，跳出循环

        # 如果文件存在，递增计数器并尝试下一个编号
        mmsi_counters[mmsi] += 1

    # 加载刚刚保存或已存在的图像并调整大小
    image = Image.open(image_filename).resize(image_size)
    image_array = np.array(image)

    return image_array


def extract_trajectories_and_images(file_path, image_save_path, image_size=(224, 224)):
    """
    从CSV文件中提取轨迹数据、标签和MMSI，同时生成或加载每艘船的轨迹图像。

    参数:
    file_path (str): CSV文件的路径。
    image_save_path (str): 图像保存路径。
    image_size (tuple): 图像大小，用于CNN模型输入。

    返回:
    X (list): 包含每艘船的轨迹特征列表。
    Y (list): 包含每艘船的类别标签列表。
    MMSIs (list): 包含每艘船的MMSI列表。
    image_features (list): 包含每艘船的图像特征数组列表。
    """

    # 读取数据
    df = pd.read_csv(file_path)

    # 定义类别映射
    ship_type_mapping = {
        'Cargo': 0,
        'Fishing': 1,
        'Tanker': 2,
        'Passenger': 3
    }

    # 应用类别映射
    df['Ship type'] = df['Ship type'].map(ship_type_mapping)

    # 提取特征
    features = ['Latitude', 'Longitude', 'SOG', 'COG', 'Heading', 'Width', 'Length', 'Latitude_change','Longitude_change']

    # 创建空列表来存储轨迹、标签、MMSI 和图像特征
    trajectories = []
    labels = []
    MMSIs = []
    image_features = []

    # 将数据分组为每艘船的轨迹
    for mmsi, group in df.groupby('MMSI'):
        # 对于每艘船，按时间顺序排列数据
        sorted_group = group.sort_values(by='# Timestamp')

        # 构建轨迹
        trajectory = sorted_group[features].values.tolist()

        # 添加轨迹到列表
        trajectories.append(trajectory)

        # 获取该船的类别作为标签
        label = sorted_group['Ship type'].iloc[0]  # 使用第一个观测的类别作为标签
        labels.append(label)

        # 保存MMSI
        MMSIs.append(mmsi)

        # 生成或加载'Latitude'和'Longitude'的轨迹图像，并获取图像特征
        latitudes = sorted_group['Latitude']
        longitudes = sorted_group['Longitude']
        speeds = sorted_group['SOG']
        image_array = save_or_load_trajectory_image(latitudes, longitudes, mmsi, image_save_path, image_size)
        #image_array = save_or_load_trajectory_image_v(latitudes, longitudes,speeds, mmsi, image_save_path, image_size)           # 集成速度

        # 将图像特征加入列表
        image_features.append(image_array)

    return trajectories, labels, MMSIs, image_features

def calculate_class_distribution(Y):
    """
    计算并打印类别的数量和比例。

    :param Y: 标签列表
    """
    # 统计各类别的数量
    counter = Counter(Y)

    # 计算总样本数量
    total_samples = sum(counter.values())

    # 打印各类别比例
    print("Class distribution:")
    for class_label, count in counter.items():
        proportion = count / total_samples
        print(f"Class {class_label}: Count = {count}, Proportion = {proportion:.2%}")
