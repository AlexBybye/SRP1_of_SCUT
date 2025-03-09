import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('preprocessing_cleaned.csv')

# 按时间行升序排序
data_sorted = data.sort_values(by='所用时间/秒')  

# 绘制时间分布直方图
plt.figure(figsize=(10, 6))
plt.hist(data_sorted['所用时间/秒'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Time Distribution Histogram')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 计算时间的描述性统计
time_stats = data_sorted['所用时间/秒'].describe()
print(time_stats)

# # 根据统计结果确定异常值阈值
# threshold = data_sorted['所用时间/秒'].quantile(0.05)
# print(f"异常值阈值：{threshold}秒")

# # 筛选异常值
# abnormal_data = data_sorted[data_sorted['所用时间/秒'] < threshold]
# print(abnormal_data)