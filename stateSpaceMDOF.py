from scipy.integrate import solve_ivp
import numpy as np



M = np.array([[2,0],[0,5]])
K = np.array([[4,-4],[-4,10]])
alpha = 0.03
beta = 0.001
C = alpha*M + beta*K 

N  = M.shape[0]

A = np.block([[np.zeros((N,N)), np.eye(N)],
                  [-np.linalg.inv(M) @ K, -np.linalg.inv(M)@C]])
B = np.block([[np.zeros((N,N))],
              [np.linalg.inv(M)]])
T = (0,30)
t_eval = np.linspace(T[0],T[1],1000,endpoint = True)
def gmotion(t):
    gm = np.loadtxt("elCentroNS.txt")
    time = gm[:,0]
    timeSeries = gm[:,1]
    if t <= time[-1]:
       return np.interp(t_eval,time,timeSeries)
    else:
        return 0


  
def dxdt(t,z):
    return A@z + (B@M)@np.ones(())* -gmotion(t)

solution = solve_ivp(dxdt,t_span = T, y0 = (1,0.5,0,0),method = "RK45",t_eval =t_eval)

x  = solution.y[:N,:]
x_dot  = solution.y[N:,:]

from matplotlib import pyplot as plt
[plt.plot(t_eval, x[i,:], label = f"{i+1}th floor displacement") for i in range(N)]
plt.legend()
plt.show()

# print(scipy.linalg.eig(A))