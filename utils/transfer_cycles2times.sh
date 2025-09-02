#!/bin/bash

# ==============================================================================
# 脚本功能:
# 1. 读取一个包含多个统计数据块的日志文件。
# 2. 对文件中每个统计项（如 "Global Total"）的周期数进行跨块累加。
# 3. 在读取完整个文件后，执行两种计算：
#    a.【累加总时间】: 将累加后的总周期数直接换算成时间（秒）。
#    b.【平均时间】:   将总周期数除以指定的块数得到平均周期，再换算成时间（秒）。
# 4. 生成一个包含上述两种计算结果的、格式化的最终报告。
# ==============================================================================

# --- 配置区 ---
# 要进行平均的数据块总数 (例如，您提到的80个)
NUM_BLOCKS=40
# CPU 频率 (单位: GHz)
FREQUENCY_GHZ="1.55"

# --- 脚本主逻辑 ---

# 检查用户是否提供了输入和输出文件名
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <包含所有数据的输入文件> <最终报告输出文件>"
    echo "示例: ./compute_total_and_avg_time.sh all_stats.log final_report.txt"
    exit 1
fi

# 将命令行参数赋值给可读的变量名
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在。"
    exit 1
fi

echo "正在从 '$INPUT_FILE' 中聚合数据..."
echo "将基于 $NUM_BLOCKS 个数据块进行汇总和平均计算..."

# 使用 awk 完成所有聚合与计算
# awk 的关联数组 `sums` 用于按名称存储每个统计项的累加周期数
awk -v num_blocks="$NUM_BLOCKS" -v freq_ghz="$FREQUENCY_GHZ" '
# 对每一行，检查它包含哪个统计项的关键字，然后将该行的最后一个字段($NF, 即数值)累加到sums数组的对应项中
/Global Total:/    { sums["total"]      += $NF }
/Global precompute:/ { sums["precompute"] += $NF }
/Global cale_time:/  { sums["cale_time"]  += $NF }
/Global sort time:/  { sums["sort"]       += $NF }
/Global reduce time:/ { sums["reduce"]     += $NF }

# END 块在整个文件读取完毕后执行一次
END {
    # --- 准备工作：定义计算常量 ---
    frequency_hz = freq_ghz * 1e9;

    # --- 第一部分：计算【累加总时间】 ---
    # 公式: 总时间(秒) = 总周期数 / 频率(Hz)
    total_time_total      = sums["total"] / frequency_hz;
    total_time_precompute = sums["precompute"] / frequency_hz;
    total_time_cale       = sums["cale_time"] / frequency_hz;
    total_time_sort       = sums["sort"] / frequency_hz;
    total_time_reduce     = sums["reduce"] / frequency_hz;

    # --- 第二部分：计算【平均时间】 ---
    # 公式: 平均时间(秒) = (总周期数 / 数据块数) / 频率(Hz)
    avg_time_total      = (sums["total"] / num_blocks) / frequency_hz;
    avg_time_precompute = (sums["precompute"] / num_blocks) / frequency_hz;
    avg_time_cale       = (sums["cale_time"] / num_blocks) / frequency_hz;
    avg_time_sort       = (sums["sort"] / num_blocks) / frequency_hz;
    avg_time_reduce     = (sums["reduce"] / num_blocks) / frequency_hz;

    # --- 打印【累加总时间】报告 ---
    print "### 累加总时间 (单位: 秒) ###";
    print "=================================================";
    printf "Global Total: %.9f\n", total_time_total*2;
    printf "|  Global precompute: %.9f\n", total_time_precompute*2;
    printf "|  Global cale_time: %.9f\n", total_time_cale*2;
    printf "|  Global sort time: %.9f\n", total_time_sort*2;
    printf "|  Global reduce time: %.9f\n", total_time_reduce*2;
    print "-------------------------------------------------";

    # 打印一个空行用于分隔
    print "";

    # --- 打印【平均时间】报告 ---
    print "### 平均时间 (单位: 秒) ###";
    print "=================================================";
    printf "Global Total: %.9f\n", avg_time_total*2;
    printf "|  Global precompute: %.9f\n", avg_time_precompute*2;
    printf "|  Global cale_time: %.9f\n", avg_time_cale*2;
    printf "|  Global sort time: %.9f\n", avg_time_sort*2;
    printf "|  Global reduce time: %.9f\n", avg_time_reduce*2;
    print "-------------------------------------------------";

    # --- 打印脚注信息 ---
    printf "\n(计算基于 %d 个数据块 和 %.2fGHz 频率)\n", num_blocks*2, freq_ghz;

}' "$INPUT_FILE" > "$OUTPUT_FILE"

# 检查输出文件是否成功生成
if [ -f "$OUTPUT_FILE" ]; then
    echo "----------------------------------------"
    echo "计算完成!"
    echo "最终报告已写入文件: $OUTPUT_FILE"
    echo ""
    echo "--- 最终报告内容预览 ---"
    cat "$OUTPUT_FILE"
else
    echo "错误: 生成报告文件失败。"
    exit 1
fi