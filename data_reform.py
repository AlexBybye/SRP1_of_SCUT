import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

# 读取真实数据（假设文件名为preprocessing_cleaned.xlsx）
real_data = pd.read_excel("preprocessing_cleaned.xlsx", sheet_name="Sheet1")

# ------------------------------
# 参数计算（基于37条真实数据）
# ------------------------------
# 1. 性别比例（1=男，2=女）
gender_ratio = real_data["性别："].value_counts(normalize=True).sort_index().values  # [0.5405, 0.4595]

# 2. 总分分布
total_mean = real_data["总分"].mean()
total_std = real_data["总分"].std()
total_min = real_data["总分"].min()
total_max = real_data["总分"].max()

# 3. 用时分布（保留所有异常值）
time_kde = KernelDensity(bandwidth=50, kernel='gaussian')
time_kde.fit(real_data[["所用时间/秒"]])  # 训练核密度估计模型

# 4. 各题项统计参数（38题）
question_cols = real_data.columns[5:-1]  # 从第5列到倒数第二列是题项
question_params = {col: {
    "mean": real_data[col].mean(),
    "std": real_data[col].std(),
    "min": real_data[col].min(),
    "max": real_data[col].max()
} for col in question_cols}

# ------------------------------
# 生成300条模拟数据
# ------------------------------
np.random.seed(42)  # 固定随机种子
n = 300

# 1. 生成基础字段
sim_data = pd.DataFrame({
    "序号": range(1, n+1),
    "所用时间/秒": np.exp(time_kde.sample(n)).flatten().astype(int),
    "总分": np.clip(np.random.normal(total_mean, total_std, n), total_min, total_max).astype(int),
    "我的学科：": 1,  # 固定为1
    "性别：": np.random.choice([1, 2], n, p=gender_ratio),
})

# 2. 生成题项（处理Q7反向题）
for col in question_cols:
    params = question_params[col]
    raw_scores = np.random.normal(params["mean"], params["std"], n)
    
    # 反向题处理：Q7生成后需转换计分方向
    if "不会注意我的感受" in col:
        raw_scores = 6 - raw_scores  # 反向计分（1→5, 2→4, etc.）
    
    scores = np.clip(np.round(raw_scores), params["min"], params["max"]).astype(int)
    sim_data[col] = scores

# 3. 保存数据
sim_data.to_csv("simulated_300_final.csv", index=False)
