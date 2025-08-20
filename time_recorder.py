# profiler.py
import time


class Timer:
    def __init__(self):
        self._timestamps = {}

    def mark(self, name):
        """记录一个名为name的时间点"""
        self._timestamps[name] = time.perf_counter()
        print(f"[Timer] Marked: {name}       {self._timestamps[name]}")

    def duration(self, start_name, end_name):
        """计算两个时间点之间的时长"""
        if start_name not in self._timestamps or end_name not in self._timestamps:
            print(f"[Timer] Error: Missing mark for {start_name} or {end_name}")
            return None
        return self._timestamps[end_name] - self._timestamps[start_name]

    def display_results(self):
        """根据你的需求，特别定制的报告函数"""
        try:
            # Duration 1: D开始 - B开始
            duration1 = self.duration("D_start", "B_start")

            # Duration 2: 程序结束 - D开始
            duration2 = self.duration("program_end", "D_start")

            print("\n==================== 测量结果 (类版本) ====================")
            print(f"从函数 B 开始到函数 D 开始的耗时: {abs(duration1):.4f} 秒")
            print(f"从函数 D 开始到程序结束的耗时:   {abs(duration2):.4f} 秒")
            print("=========================================================")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!{self._timestamps['D_start']}")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!{self._timestamps['program_end']}")

        except (KeyError, TypeError):
            print("无法计算结果，缺少必要的时间标记。")
            print("需要 'B_start', 'D_start', 'program_end' 标记。")


# 创建一个全局唯一的实例，这样在任何地方导入的都是同一个对象
timer = Timer()
