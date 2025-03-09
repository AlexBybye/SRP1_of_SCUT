import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('origin.csv', encoding='utf-8')

# 定义一个函数来判断是否需要删除某一行
def should_delete_row(row):
    # # 检查是否有某个值出现的次数大于等于 20
    value_counts = row.value_counts()
    if value_counts.max() >= 20:
        return True
    
    # 检查是否有连续 8 个相同的值
    current_value = None
    current_streak = 1
    for value in row:
        if value == current_value:
            current_streak += 1
            if current_streak >= 8:
                return True
        else:
            current_value = value
            current_streak = 1
    
    # 如果两个条件都不满足，则返回 False
    return False

# 应用函数到每一行，判断是否需要删除
rows_to_delete = df.apply(should_delete_row, axis=1)

# 删除满足条件的行
df_cleaned = df[~rows_to_delete]

# 将清理后的数据保存到新的 CSV 文件
df_cleaned.to_csv('preprocessing_cleaned.csv', index=False, encoding='utf-8')

print("处理完成，已保存到 preprocessing_cleaned.csv")