from scipy.integrate import solve_ivp
import numpy as np

mass = 1
k  = 4
omega = np.sqrt(mass/k)
zeta = 0.3
c = 2*zeta*mass*omega
def Force(t):
    return 5
def state(t,x):
    return np.array([x[1], -c*x[1] - k * x[0] + Force(t)])


solution = solve_ivp(state, (0,50), (0.,0), method = "RK45", t_eval=np.linspace(0,50,2000,endpoint=True))


from matplotlib import pyplot as plt

# print(solution.y.shape)
plt.plot(np.linspace(0,50,2000,endpoint= True),solution.y[0,:])
plt.show()

