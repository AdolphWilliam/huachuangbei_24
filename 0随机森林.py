import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文显示和字体
plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载数据集
# data = pd.read_excel('指标化数据.xlsx', sheet_name=1)
data = pd.read_excel('模拟数据.xlsx')

# 定义因变量和自变量
X = data[["品牌意识综合指标", "产品偏好综合指标", 
    "消费心理综合指标", "性别","年龄", "受教育水平", 
    "职业", "月收入"]]

y = data["购买频率"]

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# 2. 构建随机森林回归模型
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=0
)

# 训练模型
rf_regressor.fit(X_train, y_train)

# 对测试集进行预测
y_pred_rf = rf_regressor.predict(X_test)

# 评估模型
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 打印回归报告
print("随机森林回归模型评估报告")
print("----------------------------")
print(f"均方误差 (MSE): {mse_rf:.4f}")
print(f"均方根误差 (RMSE): {rmse_rf:.4f}")
print(f"R² 分数: {r2_rf:.4f}")

# 特征重要性
feature_importances_rf = rf_regressor.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

# 3. 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#1f77b4')  # 使用深蓝色
plt.xlabel('特征重要性', fontsize=14)
plt.title('随机森林回归模型特征重要性（从大到小）', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.gca().invert_yaxis()  # 反转y轴，使重要性高的特征显示在顶部
plt.tight_layout()  # 优化布局
plt.show()
