import math
import scipy
import numpy as np
import matplotlib.pyplot as plt


def construct_m(t, w):
    r = np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t), np.cos(t)]
    ])
    return r @ w / np.sqrt(np.sum(w ** 2))


def cal_r(A, u, t):
    r0 = np.array([0, 0]).reshape(-1, 1)
    r_inf = - np.linalg.inv(A) @ u
    return scipy.linalg.expm(A * t) @ (r0 - r_inf) + r_inf


def cal_r_pert(A, u, t, r_0, r_t):
    r_inf = - np.linalg.inv(A) @ u
    r_c = r_0 + r_t
    return scipy.linalg.expm(A * t) @ (r_c - r_inf) + r_inf


t1 = .02 * np.pi
t2 = t1 + 3 * np.pi / 2
tau1 = 10
tau2 = 2
lam1 = - 1 / tau1
lam2 = - 1 / tau2
Lam = np.diag([lam1, lam2])
g1 = np.array([6, 6]).reshape(-1, 1)
g2 = np.array([5, 7]).reshape(-1, 1)
sig = np.array([
    [20, 10],
    [10, 20],
])
u = (g1 + g2) / 2
wld = np.linalg.inv(sig) @ (g2 - g1)
m1 = construct_m(t1, wld)
m2 = construct_m(t2, wld)
M = np.concatenate([m1, m2], axis=-1)
A = np.linalg.inv(M) @ Lam @ M


t = np.arange(0, 100, .01)
r = np.zeros(shape=(10000, 2))
for idx, item in enumerate(t):
    tmp = cal_r(A, u, item).reshape(-1)
    r[idx] = tmp
r1 = r[:, 0]
r2 = r[:, 1]
plt.plot(r1, c='b', label='1')
plt.plot(r2, c='r', label='2')
plt.legend()
plt.show(block=True)

r_n = np.zeros_like(r)
for idx, item in enumerate(t):
    tmp = cal_r_pert(A, u, item, r[-1].reshape(-1, 1), np.array([0, 10]).reshape(-1, 1)).reshape(-1)
    r_n[idx] = tmp
r_all = np.concatenate([r, r_n], axis=0)
r1 = r_all[:, 0]
r2 = r_all[:, 1]
plt.plot(r1, c='b', label='1')
plt.plot(r2, c='r', label='2')
plt.legend()
plt.show(block=True)

plt.plot(r1, r2)
plt.scatter(r1[9910], r2[9910], c='r')
plt.show(block=True)

mod1 = r_all @ m1
mod2 = r_all @ m2
plt.plot(mod1, c='b', label='1')
plt.plot(mod2, c='r', label='2')
plt.legend()
plt.show(block=True)

mod_input = np.zeros_like(r_all)
mod_input[10000] = np.array([0, 10])
mod1 = mod_input @ m1
mod2 = mod_input @ m2
plt.subplot(211)
plt.plot(mod1, c='b', label='1')
plt.legend()
plt.subplot(212)
plt.plot(mod2, c='r', label='2')
plt.legend()
plt.show(block=True)


""" ode """
def generate_r(t, stim):
    r_r = np.zeros(shape=(len(t), 2))
    for idx, item in enumerate(t):
        if idx == 0:
            continue
        else:
            r_r[idx] = r_r[idx - 1] + (A @ r_r[idx - 1] + stim[idx])
    r_r = np.concatenate([np.zeros(shape=(40, 2)), r_r], axis=0)
    r_new = np.zeros(shape=(104, 2))
    for idx in range(len(r_new)):
        r_new[idx] = r_r[idx: idx+10].mean(axis=0)
    return r_new


t = np.arange(1000000)
stim1 = np.random.multivariate_normal(g1.reshape(-1), sig, len(t))
stim2 = np.random.multivariate_normal(g2.reshape(-1), sig, len(t))
r_1 = generate_r(t, stim1)
r_2 = generate_r(t, stim2)

plt.plot(r_1[:, 0], r_1[:, 1], c='b')
plt.plot(r_2[:, 0], r_2[:, 1], c='r')
plt.scatter(r_1[-1, 0], r_1[-1, 1], c='black')
plt.scatter(r_2[-1, 0], r_2[-1, 1], c='black')
plt.show(block=True)
