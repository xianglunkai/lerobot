import numpy as np
from scipy import sparse
import osqp

# 简单测试数据
T = 10
true = np.sin(np.linspace(0, 2*np.pi, T))
noisy = true + 0.2*np.random.randn(T)
noisy[0] = true[0] + 0.5  # 故意让首端偏离

# 构建差分矩阵
D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T), format='csc')
I = sparse.eye(T, format='csc')
w_data, w_acc = 1.0, 5.0
P = w_data * I + w_acc * (D2.T @ D2) + 1e-6 * I
q = -w_data * noisy

# 固定端点
A_eq = sparse.vstack([sparse.eye(1, T), sparse.eye(1, T, k=T-1)])
l_eq = np.array([noisy[0], noisy[-1]])
u_eq = l_eq.copy()

prob = osqp.OSQP()
prob.setup(P=P, q=q, A=A_eq, l=l_eq, u=u_eq, verbose=False)
res = prob.solve()
x = res.x

print("Original noisy:", noisy)
print("Smoothed:", x)
print("RMSE vs true:", np.sqrt(np.mean((x - true)**2)))