import numpy as np
import csv
import glob
import matplotlib.pyplot as plt

files = glob.glob("./scratch/*")

regret = []

for file in range(0, len(files), 2):
    np_file = np.load(files[file])
    regret.append(np_file[-1])

y = np.array(regret)
x = np.arange(0, len(regret))

plt.title("Regret Over Time")

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y, color='navy')

plt.plot(x, a*x+b, color='black', linestyle='--', linewidth=2)

plt.text(0, 4000, 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)

plt.show()
