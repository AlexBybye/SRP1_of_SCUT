import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载数据
data = pd.read_csv('cleaned_data.csv')

# -------------------------------
# 1. 反馈机制现状分析（按性别分组）
# -------------------------------
teacher_feedback_columns = [
    '在我完成写作作业后，老师会提供反馈意见。',
    '老师会及时提供写作反馈。',
    '老师会根据不同的写作任务调整反馈的重点。',
    '老师把反馈作为我们完成写作任务过程中的必备环节。',
    '老师将有限的写作反馈机会聚焦于我最需要指导/改进的方面。',
    '老师在提供写作反馈的时候不会注意我的感受。',
    '老师鼓励我们向她/他或同学寻求写作反馈意见。',
    '老师在给我们写作反馈的时候，会通过理解和信任来帮助我们。',
    '老师会借助网络与我交流写作的反馈意见。',
    '老师帮助我认识写作反馈是双向互动过程。',
    '老师会强调学生不仅是反馈的接收者，也是反馈的提供者。',
    '老师在教学活动中会指导我积极参与写作反馈（包括:如何寻求反馈、理解反馈或者根据反馈修改作文等）。'
]

# 按性别分组
gender_groups = data.groupby('性别：')

# 绘制按性别分组的教师反馈平均得分
plt.figure(figsize=(12, 8))
for name, group in gender_groups:
    group_mean = group[teacher_feedback_columns].mean()
    group_mean.plot(kind='barh', label=f'性别 {name}', alpha=0.7)
plt.title('教师反馈相关问题的平均得分（按性别）')
plt.xlabel('平均得分')
plt.ylabel('问题')
plt.xlim(1, 5)
plt.legend()
plt.show()

# -------------------------------
# 2. 学生对写作反馈的感知和态度（按性别分组）
# -------------------------------
student_perception_columns = [
    '我会根据收到的反馈改进我的写作。',
    '我会通过自我反思来加强和规范我的写作。',
    '如果有评分标准，我在写作前会通读评分标准。',
    '在做写作作业之前，我会先学习范例或老师推荐的文章。',
    '我会根据老师的指导来计划和管理我的写作。',
    '我会通过使用网上参考资料来解决写作中遇到的一些问题。',
    '我会观察其他学生是如何完成写作任务的，这样我就可以改进我的写作策略。',
    '如果我一开始不能理解收到的写作反馈，我会不断思索，直到我理解为止。',
    '我会判断自己的写作质量是否符合要求，是否需要进一步修改。',
    '我会把老师在课堂上的指导和我的写作作业联系起来，思考我哪里写的不好。',
    '当我阅读写作反馈时，我可以分清哪些是重要信息，哪些是次重要信息。',
    '根据收到的写作反馈，我会发现自己在写作方面存在哪些不足之处，并思考下一步该怎么做。',
    '我会分析来自不同来源的写作反馈，并认识到它们对我提高写作水平的价值。',
    '我会积极寻求同伴的写作反馈。',
    '我会积极寻求老师的写作反馈。',
    '我会和同学讨论并给出写作反馈。',
    '我愿意和别人分享我关于写作作业的想法。',
    '我重视别人的写作反馈。',
    '我期待收到对我的写作作业的反馈。',
    '我喜欢收到关于我的写作作业的有挑战性和建设性的反馈。',
    '当我收到关于我的写作作业的负面反馈时，我会感到沮丧。',
    '当收到老师的写作反馈时，我感觉很好。'
]

plt.figure(figsize=(12, 8))
for name, group in gender_groups:
    group_mean = group[student_perception_columns].mean()
    group_mean.plot(kind='barh', label=f'性别 {name}', alpha=0.7)
plt.title('学生对写作反馈的感知和态度平均得分（按性别）')
plt.xlabel('平均得分')
plt.ylabel('问题')
plt.xlim(1, 5)
plt.legend()
plt.show()

# -------------------------------
# 3. 反馈机制对学生写作能力的影响（按性别分组）
# -------------------------------
feedback_type_columns = [
    '老师会及时提供写作反馈。',
    '老师鼓励我们向她/他或同学寻求写作反馈意见。',
    '我会积极寻求同伴的写作反馈。',
    '我会积极寻求老师的写作反馈。',
    '我会和同学讨论并给出写作反馈。'
]

writing_ability_columns = [
    '我会根据收到的反馈改进我的写作。',
    '我会通过自我反思来加强和规范我的写作。',
    '我会判断自己的写作质量是否符合要求，是否需要进一步修改。',
    '根据收到的写作反馈，我会发现自己在写作方面存在哪些不足之处，并思考下一步该怎么做。',
    '我会分析来自不同来源的写作反馈，并认识到它们对我提高写作水平的价值。'
]

plt.figure(figsize=(10, 6))
for name, group in gender_groups:
    correlations = {}
    for col in feedback_type_columns:
        corr = group[col].corr(group[writing_ability_columns].mean(axis=1))
        correlations[col] = corr
    df_corr = pd.DataFrame.from_dict(correlations, orient='index', columns=[f'相关性（性别 {name}）'])
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'不同类型反馈与写作能力的相关性（性别 {name}）')
    plt.show()

# -------------------------------
# 4. 反馈机制的改进建议（按性别分组）
# -------------------------------
expectation_columns = [
    '我期待收到对我的写作作业的反馈。',
    '我喜欢收到关于我的写作作业的有挑战性和建设性的反馈。',
    '我会积极寻求老师的写作反馈。',
    '我会积极寻求同伴的写作反馈。'
]

plt.figure(figsize=(12, 8))
for name, group in gender_groups:
    group_mean = group[expectation_columns].mean()
    group_mean.plot(kind='barh', label=f'性别 {name}', alpha=0.7)
plt.title('学生对反馈机制的期待平均得分（按性别）')
plt.xlabel('平均得分')
plt.ylabel('期待')
plt.xlim(1, 5)
plt.legend()
plt.show()

# -------------------------------
# 5. 模型构建与预测
# -------------------------------
# 如果没有关于学生反馈后行动的指标，定义为空列表
student_action_columns = []  

# 构建特征集，仅使用反馈类型和性别（取消学科信息）
X = data[feedback_type_columns + ['性别：']]
y = (data['我重视别人的写作反馈.'] >= 4).astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
