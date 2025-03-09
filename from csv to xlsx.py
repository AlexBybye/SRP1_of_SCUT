import pandas as pd

def csv_to_xlsx():
    # 定义CSV文件和XLSX文件的路径
    csv_file = 'preprocessing_cleaned.csv'
    xlsx_file = 'preprocessing_cleaned.xlsx'
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 将数据写入XLSX文件
    df.to_excel(xlsx_file, index=False)
    print(f"CSV文件已成功转换为XLSX文件: {xlsx_file}")

if __name__ == "__main__":
    csv_to_xlsx()