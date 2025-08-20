# profiler.py
import time


class Timer:
    def __init__(self):
        self._timestamps = {}

    def mark(self, name):
        """Record a timestamp with the given name"""
        self._timestamps[name] = time.perf_counter()
        print(f"[Timer] Marked: {name}")

    def duration(self, start_name, end_name):
        """Calculate the duration between two timestamps"""
        if start_name not in self._timestamps or end_name not in self._timestamps:
            print(f"[Timer] Error: Missing mark for {start_name} or {end_name}")
            return None
        return self._timestamps[end_name] - self._timestamps[start_name]

    def display_results(self):
        """Custom report function based on your requirements"""
        try:
            # Duration 1: D_start - B_start
            duration1 = self.duration("D_start", "B_start")

            # Duration 2: program_end - D_start
            duration2 = self.duration("program_end", "D_start")
            result_path = "/workspace/chakra_fx/test_result"

            text_to_write = f"\n==================== Test Result ====================\nCompiling Time: {abs(duration1):.4f} s\nConversion to Chakra Time:   {abs(duration2):.4f} s\n========================================================="
            with open(result_path, "a", encoding="utf-8") as f:
                f.write(text_to_write)

            print("\n==================== Test Result ====================")
            print(f"Compiling Time: {abs(duration1):.4f} s")
            print(f"Conversion to Chakra Time:   {abs(duration2):.4f} s")
            print("=========================================================")

        except (KeyError, TypeError):
            print("Unable to calculate results, missing required time marks.")
            print("Required marks: 'B_start', 'D_start', 'program_end'.")


# Create a globally unique instance so that the same object is imported everywhere
timer = Timer()
