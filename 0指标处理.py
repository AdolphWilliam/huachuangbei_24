import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#导入数据
file_name = '华创杯数分数据.xlsx'
df = pd.read_excel(file_name, skiprows=2)
# print(df.head())


#1.品牌意识PCA处理
#选取品牌意识系列变量
df1=df.iloc[:,7:14]

# 标准化数据
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df1)

# 运用 PCA
pca = PCA(n_components=1)  # 将系列变量将为一维
principal_component1 = pca.fit_transform(standardized_data)

# 添加到原df
df['品牌意识综合指标'] = principal_component1


#2.产品偏好PCA处理
#选取产品偏好系列变量
df2=df.iloc[:,14:25]

# 标准化数据
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df2)

# 运用 PCA
pca = PCA(n_components=1)  # 将系列变量将为一维
principal_component2 = pca.fit_transform(standardized_data)

# 添加到原df
df['产品偏好综合指标'] = principal_component2

#3.消费心理PCA处理
#选取消费心理系列变量
df3=df.iloc[:,25:33]

# 标准化数据
scaler = StandardScaler()
standardized_data = scaler.fit_transform(df3)

# 运用 PCA
pca = PCA(n_components=1)  # 将系列变量将为一维
principal_component3 = pca.fit_transform(standardized_data)

# 添加到原df
df['消费心理综合指标'] = principal_component3
# print(df)

# 导出DataFrame到Excel文件
df.to_excel("指标化数据.xlsx", index=False)
