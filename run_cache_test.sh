#!/bin/bash

# 当任何命令失败时立即退出
set -e

# --- 配置区 ---
# 定义CSV缓存文件的路径 (与你Python代码中的路径一致)
CACHE_FILE="/workspace/chakra_fx/gemm_collected.csv"

# 定义基础的torchrun命令 (不包含--use_cache)
BASE_CMD="torchrun --nproc-per-node=4 profile_fxgraph.py --fxgraph_actions chakra --exp_tag test_output --job fwbw --model llama"

# --- 测试流程 ---

echo "==========================================================="
echo "📊 STARTING END-TO-END PROFILING TESTS 📊"
echo "==========================================================="

# 1. 清理环境, 确保从一个干净的状态开始
echo "🧹 [Step 1] Cleaning up previous results and cache..."
rm -f "$CACHE_FILE"
rm -rf "test_output/" # 清理上一次的输出目录
echo "Cleanup complete."
echo "-----------------------------------------------------------"


# 2. 测试 use_cache=0 (无缓存, 基准测试)
echo "🚀 [Step 2] Running test with use_cache=0 (Baseline)..."
# 使用 'time' 命令来测量整个命令的执行时间
time $BASE_CMD --use_cache 0
echo "✅ Test with use_cache=0 finished."
echo "-----------------------------------------------------------"


# 3. 测试 use_cache=1 (首次运行, 生成缓存)
# 这一次运行会很慢,因为它需要测量所有算子并写入CSV文件
echo "🚀 [Step 3] Running test with use_cache=1 (First Run - Cache Generation)..."
# 再次清理, 确保这次是独立生成缓存
rm -f "$CACHE_FILE"
rm -rf "test_output/"
time $BASE_CMD --use_cache 1
echo "✅ Test with use_cache=1 (First Run) finished. Cache file should now exist."
echo "-----------------------------------------------------------"


# 4. 测试 use_cache=1 (再次运行, 使用缓存)
# 这次运行应该会非常快, 因为它会从CSV文件中读取所有duration
echo "🚀 [Step 4] Running test with use_cache=1 (Second Run - With Cache Hit)..."
rm -rf "test_output/" # 只清理输出, 不清理缓存文件
time $BASE_CMD --use_cache 1
echo "✅ Test with use_cache=1 (Second Run) finished."
echo "-----------------------------------------------------------"


# 5. 测试 use_cache=2 (跳过测量)
# 这次运行理论上应该是最快的, 因为它既不测量也不读写CSV
echo "🚀 [Step 5] Running test with use_cache=2 (Skip Measurement)..."
rm -rf "test_output/"
time $BASE_CMD --use_cache 2
echo "✅ Test with use_cache=2 finished."
echo "==========================================================="
echo "🎉 All tests completed! 🎉"
echo "==========================================================="