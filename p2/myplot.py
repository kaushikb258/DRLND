import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('performance_standard.txt')

b = np.loadtxt('performance_prioritized.txt')


print(a.shape, b.shape)

plt.plot(a[:,0], a[:,1], label='standard')
plt.plot(b[:,0], b[:,1], label='prioritized')
plt.legend(fontsize=15)
plt.xlabel('episode', fontsize=15)
plt.ylabel('episode reward', fontsize=15)
plt.axis([0, 500, 0, 40])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.show()
plt.savefig('rewards.eps')
