# %%
print("hello world")
# %%


import matplotlib.pyplot as plt
import numpy as np

# %%

xticks = ["Flint", "Flint-Cache", "Flint-No Duration"]
x = [0.3, 1, 1.7]

y1 = [1.025919763, 2.315008653]
y2 = [1.060189338, 0.044403392]
y3 = [0.986501582, 0.013498418]

y1 = [30.766, 69.4241]
y2 = [31.7937, 1.3316]
y3 = [29.5839, 0.4048]

bar_width = 0.25

plt.bar(x[0], y1[0], bar_width, color="C0")
plt.bar(x[0], y1[1], bar_width, bottom=y1[0], color="C1")

plt.bar(x[1], y2[0], bar_width, color="C0")
plt.bar(x[1], y2[1], bar_width, bottom=y2[0], color="C1")

plt.bar(x[2], y3[0], bar_width, color="C0")
plt.bar(x[2], y3[1], bar_width, bottom=y3[0], color="C1")

plt.bar(0, 0, bar_width, color="C0", label="FX graph capture")
plt.bar(0, 0, bar_width, color="C1", label="Conversion to Chakra graph")

plt.xticks(x, xticks, fontsize=12)
plt.legend(fontsize=12)
plt.xlim(0, 2)
plt.xlabel("Runtime Configurations", fontsize=14)
plt.ylabel("Average Duration (s)", fontsize=14)
plt.savefig("scalability.pdf")
# %%
