from matplotlib import pyplot as plt
from scipy import linalg as slin
from scipy.integrate import solve_ivp
import numpy as np


M = np.loadtxt("MCKFolder\M.txt")
C = np.loadtxt("MCKFolder\C.txt")
K = np.loadtxt("MCKFolder\K.txt")
N = M.shape[0]
A = np.block([[np.zeros((N,N)), np.eye(N)],
              [-np.linalg.solve(M,K),-np.linalg.solve(M,C)]])

def dxdt(t,z):
    return A @ z
    # only the damped vibration

t_col = (0,10)
init_condition = np.ones(N*2)
pts = 2000
t_eval  = np.linspace(t_col[0],t_col[1],pts)
print(A.shape)
solution = solve_ivp(dxdt, t_col, init_condition, method = "RK45", t_eval = t_eval)

hor_DOFs = np.loadtxt("MCKFolder/Hor_DOFs.txt")
# Find out the horizontal DOFs
x = solution.y[:N//2,:]

[plt.plot(t_eval,x[int(i),:], label = f"{i}th floor displacement") for i in range(x.shape[0])]
plt.legend()
plt.grid()
plt.show()