import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 14
font = {'size': 20}
matplotlib.rc('font', **font)
alpha = 1

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={
    'width_ratios': [0.9, 1.5, 1.3]}, figsize=(28, 5))


# INPUT RATE EXPERIMENT
ir = [10, 100, 1000, 10000]
# embedded
nd4j = [9.95, 95.12, 696.33, 1429.46]
onnx = [9.96, 97.73, 888.47, 6663.73]
tf = [9.96, 96.48, 867.76, 5191.74]
# external
tor_s = [9.99, 99.12, 427.39, 515.06]
tf_s = [9.99, 99.17, 981.86, 2225.36]

ax1.plot(ir, ir, linestyle='--', c='lightgrey', label='ideal', linewidth=3)
ax1.plot(ir, nd4j, marker='o', label='nd4j', alpha=alpha, ls='-', c='#e6194B')
ax1.plot(ir, onnx, marker='o', label='onnx', alpha=alpha, ls='-', c='#f58231')
ax1.plot(ir, tf, marker='o', label='tf', alpha=alpha, ls='-', c='#ffe119')
ax1.plot(ir, tf_s, marker='X', label='tf-s', alpha=alpha, ls='-', c='#3cb44b')
ax1.plot(ir, tor_s, marker='X', label='tor-s',
         alpha=alpha, ls='-', c='#4363d8')

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.legend()

ax1.set_title('Throughput Experiment.')

# fig.text(.5, 0, "CAPTION", ha='left', fontsize=30)

ax1.set_xlabel('Input Rate (req/s)')
ax1.set_ylabel('Average Throughput (req/s)')
ax1.grid(True)


# BATCH SIZE
bs = ['1', '10', '100', '250', '500', '1000']
# embedded
nd4j = [0.77, 2.514, 20.98, 52.96, 126.705, 287.01]
onnx = [0.35, 2.76, 27.83, 72.51, 166.19, 367.56]
tf = [0.51, 2.62, 24.9, 64.71, 147.95, 333.82]

# external
tor_s = [2.25, 18.4, 208.67, 550.77, 1126.47, 2293.89]
tor_sg = [2.26, 18.39, 209.63, 551.061, 1131.104, 2308.83]
tf_s = [0.53, 1.47, 9.64, 22.06, 48.88, 102.8]

X = np.arange(len(bs))
width = 0.17


hatches = itertools.cycle(['/', '+', '//', '-', 'x', '\\', '*', 'o', 'O', '.'])

ax2.bar(X, nd4j, width=width, label='nd4j', hatch=next(
    hatches), edgecolor='black', color='#e6194B')
ax2.bar(X + width, onnx, width=width, label='onnx',
        hatch=next(hatches), edgecolor='black', color='#f58231')
ax2.bar(X + 2 * width, tf, width=width, label='tf',
        hatch=next(hatches), edgecolor='black', color='#ffe119')

ax2.bar(X + 3 * width, tf_s, width=width, label='tf-s',
        hatch=next(hatches), edgecolor='black', color='#3cb44b')
# ax.bar(X + 4 * width, tor_s, width = width, label='tor-s', hatch = next(hatches), edgecolor='black')
ax2.bar(X + 4 * width, tor_sg, width=width, label='tor-s',
        hatch=next(hatches), edgecolor='black', color='#4363d8')

ax2.set_xticks([i + 0.3 for i in X], bs)
ax2.set_yscale('log')
ax2.set_ylim(ymax=10000)
ax2.legend(ncol=3)
ax2.set_title('Latency Experiment.')
ax2.set_xlabel('Batch Size (images/batch)')
ax2.set_ylabel('Average Latency (ms/batch)')
ax2.set_axisbelow(True)
ax2.grid(True)


# SCALABILITY EXPERIMENT
mr = ['1', '2', '4', '8', '16']
# embedded
nd4j = [1508.222, 2352.52, 2642.799, 2727.805, 2185.114]
onnx = [6649.69, 13050.89, 25354.03, 27513.56, 9022.43]
tf = [4486.12, 9241.91, 15145.13, 26748.63, 26793.25]

# external
tor_s = [512.85, 988.46, 1919.98, 2241.75, 2144.26]
tf_s = [2758.23, 4066.09, 6493.42, 8176.73, 8135]

X = np.arange(len(mr))
width = 0.17

hatches = itertools.cycle(['/', '+', '//', '-', 'x', '\\', '*', 'o', 'O', '.'])
ax3.bar(X, nd4j, width=width, label='nd4j', hatch=next(
    hatches), edgecolor='black', color='#e6194B')
ax3.bar(X + width, onnx, width=width, label='onnx',
        hatch=next(hatches), edgecolor='black', color='#f58231')
ax3.bar(X + 2 * width, tf, width=width, label='tf',
        hatch=next(hatches), edgecolor='black', color='#ffe119')

ax3.bar(X + 3 * width, tf_s, width=width, label='tf-s',
        hatch=next(hatches), edgecolor='black', color='#3cb44b')
ax3.bar(X + 4 * width, tor_s, width=width, label='tor-s',
        hatch=next(hatches), edgecolor='black', color='#4363d8')

ax3.set_xticks([i + 0.3 for i in X], mr)

ax3.set_yscale('log')
ax3.set_title('Vertical Scalability Experiment.')
ax3.set_xlabel('Vertical Scalability (#model servers)')
ax3.set_ylabel('Average Throughput (req/s)')
ax3.set_ylim(ymax=70000)
ax3.set_axisbelow(True)
ax3.grid(True)

fig.tight_layout()
plt.savefig("all-plots.png", bbox_inches='tight')
# plt.show()
