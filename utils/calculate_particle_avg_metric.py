import re
def calculate_avg_metric(file_path):
    """
    计算文件中以 arvg_metric: 开头的行中数字的平均值
    :param file_path: 输入文件路径
    :return: 平均值若无匹配行则返回 None
    """
    values = []
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
            # 匹配以 arvg_metric: 开头且后跟数字的行
                match = re.match(r'arvg_metric:\s*(\d+)', line.strip())
                if match:
                    values.append(int(match.group(1)))  # 提取数字并转为整数[6,7](@ref)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    # 计算平均值避免除零错误
    if not values:
        print("警告未找到符合条件的行")
        return None
    
    average = sum(values) / len(values)  # 使用内置函数计算平均值[1,5](@ref)
    print("len(values)", len(values))
    return average

# 示例调用
file_path = "sme.11LWFA.444.timer.static.out"  # 替换为实际文件路径
print(file_path)
result = calculate_avg_metric(file_path)
if result is not None:
    print(f"平均值: {result:.2f}")

