import numpy as np
import matplotlib.pyplot as plt
#
# from base_data_two_photo import f_trial1, trial1_stim_index, f_dff, trial_stim_index
# from utils import generate_firing_curve_config, visualize_firing_curves
#
# config = generate_firing_curve_config()
# index = range(198, 208)
# config['mat'] = f_dff[index]
# config['stim_kind'] = 'single'
# config['stim_index'] = trial_stim_index
# visualize_firing_curves(config)

cov = np.array([[2, 1], [1, 2]])
t1 = np.arange(100)
t2 = np.arange(100)
u1 = np.random.multivariate_normal(mean=[1, 2], cov=cov, size=1000)
u2 = np.random.multivariate_normal(mean=[2, 1], cov=cov, size=1000)
x11 = u1[:, 0]
x12 = u1[:, 1]
y11 = u2[:, 0]
y12 = u2[:, 1]
# z1 = x11 - x12
# z2 = y11 - y12
# plt.plot(z1, c='blue')
# plt.plot(z2, c='red')
# plt.show(block=True)
# z1 = np.cumsum(z1)
# z2 = np.cumsum(z2)
# plt.plot(x11, c='blue')
# plt.plot(x12, c='red')
# plt.show(block=True)

plt.scatter(x11, x12, c='b')
plt.scatter(y11, y12, c='r')
plt.show(block=True)

x = np.random.normal(1, 1, 1000)
y = np.random.normal(0, 1, 1000)
z1 = x + y
z2 = x - y
plt.hist(z2)
plt.show(block=True)

z = np.linalg.inv(cov) @ np.array([-1, 1])
plt.plot(np.cumsum(x))
plt.show()
