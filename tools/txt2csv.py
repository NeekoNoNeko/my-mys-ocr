import csv

def txt_to_csv(txt_path, csv_path, delimiter='\t', headers=None):
    """
    将固定分隔符的 .txt 文件转换为 .csv 文件

    :param txt_path: 输入的 .txt 文件路径
    :param csv_path: 输出的 .csv 文件路径
    :param delimiter: txt 文件中字段的分隔符，默认是逗号
    :param headers: 可选，表头列表，例如 ['姓名', '年龄', '城市']
    """
    with open(txt_path, 'r', encoding='utf-8') as infile:
        lines = [line.strip() for line in infile if line.strip()]

    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.writer(outfile)

        if headers:
            writer.writerow(headers)

        for line in lines:
            row = line.split(delimiter)
            writer.writerow(row)

# 示例用法
if __name__ == "__main__":
    txt_file = '/root/workspace/data/single_combined/ocr_results.txt'
    csv_file = '/root/workspace/data/single_combined/ocr_results.csv'
    headers = ['图片路径', '预测']  # 如果没有表头，可以设为 None
    txt_to_csv(txt_file, csv_file, delimiter='\t', headers=headers)
    print("转换完成！")