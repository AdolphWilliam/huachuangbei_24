import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体和样式
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set(style="whitegrid", font='Microsoft YaHei')  # 设置Seaborn样式和字体

# 1. 加载数据集
# df = pd.read_excel('指标化数据.xlsx', sheet_name=1)
df = pd.read_excel('模拟数据.xlsx')

# 特征和目标
X = df[['品牌意识综合指标', '产品偏好综合指标', '消费心理综合指标', 
    '性别', '年龄', '受教育水平', '职业', '月收入']]
y = df['购买频率']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# 绘制肘部法图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), wcss, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('簇数', fontsize=14)
plt.ylabel('簇内平方和', fontsize=14)
plt.title('肘部法 - 确定最佳聚类数', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 根据肘部法确定8簇
kmeans = KMeans(n_clusters=8, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 根据每个簇的购买频率均值对簇进行排序
cluster_order = df.groupby('Cluster').mean()['购买频率'].sort_values(ascending=False).index

# 选择排名前三的簇
top_clusters = cluster_order[:3]

# 为每个簇绘制堆积柱形图（每个图为一个子图）
colors = ['#85c1e9', '#4575b4','#f6e0b3', '#a7e7a7', '#f9c291']
for i, cluster in enumerate(top_clusters):
    # 获取每个簇的数据
    cluster_data = df[df['Cluster'] == cluster]

    # 计算百分比频数
    gender_freq = cluster_data['性别'].value_counts(normalize=True) * 100
    age_freq = cluster_data['年龄'].value_counts(normalize=True) * 100
    education_freq = cluster_data['受教育水平'].value_counts(normalize=True) * 100
    job_freq = cluster_data['职业'].value_counts(normalize=True) * 100
    income_freq = cluster_data['月收入'].value_counts(normalize=True) * 100

    # 创建频数数据表
    data = pd.DataFrame({
        '性别': gender_freq,
        '年龄': age_freq,
        '受教育水平': education_freq,
        '职业': job_freq,
        '月收入': income_freq
    }).fillna(0)

    # 创建子图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制堆积柱形图
    data.T.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(data.columns)])
    
    # 设置图表标题和标签
    ax.set_title(f"群体 {cluster} - 购买频率均值: {df[df['Cluster'] == cluster]['购买频率'].mean():.2f}", fontsize=16)
    ax.set_ylabel('百分比 (%)', fontsize=14)
    ax.set_xlabel('特征类别', fontsize=14)
    ax.legend(title="类别", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # 显示网格和调整布局
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()  # 优化布局
    plt.show()

