import pandas as pd

def convert_xlsx_to_csv(xlsx_file, csv_file, sheet_name=0):
    """
    将 Excel (.xlsx) 文件转换为 CSV 文件

    参数:
    - xlsx_file (str): 输入的 xlsx 文件路径
    - csv_file (str): 输出的 csv 文件路径
    - sheet_name (int 或 str, 可选): 需要转换的工作表名称或索引，默认为第一个工作表(0)

    返回:
    - None: 直接将转换后的 CSV 文件写入到磁盘
    """
    try:
        # 读取指定工作表的内容
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        # 保存为 CSV 格式，不保留索引
        df.to_csv(csv_file, index=False)
        print(f"转换成功：{xlsx_file} -> {csv_file}")
    except Exception as e:
        print(f"转换过程中出现错误: {e}")

if __name__ == '__main__':
    # 示例：设置输入和输出文件路径
    input_file = 'preprocessing.xlsx'
    output_file = 'preprocessing.csv'
    convert_xlsx_to_csv(input_file, output_file)
