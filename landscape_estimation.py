import numpy as np
import matplotlib.pyplot as plt
from numba import jit





def E(X,weights,centroid,sigma):
    x = X.reshape(-1,2)
    energy = weights * np.exp([-((np.expand_dims(x[...,0],1)-centroid[...,0])**2 +(np.expand_dims(x[...,1],1)-centroid[...,1])**2) / (2*sigma**2)])
    energy = np.sum(energy,2)
    energy = energy.reshape(X.shape[:2])
    return energy

weights = np.random.uniform(0.1,0.2,3)
centroid = np.random.uniform(1,2,(3,2))
sigma = np.random.uniform(0.2,0.4,3)

x_space = np.linspace(0,3)
X = np.array(np.meshgrid(x_space,x_space,indexing="ij"))
X = X.transpose(1,2,0)

plt.imshow(E(X,weights,centroid,sigma))
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
a = 1
c = 0.5
x1 = -1
x2 = 1

# Generate x values
x = np.linspace(-1.5,1.5, 1000)

# Calculate y values
y = ((x - x1)**2*(x - x2)**2 + c*x)

# Plot the curve
plt.plot(x, y)
# plt.plot(x[1:],y[1:]-y[:-1])

plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()

from scipy.optimize import fsolve,minimize

@jit(nopython=True)
def E(x,x1,x2,a,c):
    return a*((x - x1) ** 2 * (x - x2) ** 2 + c * x)
@jit(nopython=True)
def dE(x,x1,x2,a,c):
    return c + 2 * (x - x1)**2 * (x - x2) + 2 * (x - x1) * (x - x2)**2
@jit(nopython=True)
def dE_sq(x,x1,x2,a, c):
    return a**2 * (c + 2 * (x - x1)**2 * (x - x2) + 2 * (x - x1) * (x - x2)**2)**2


def get_features_from_E(x1,x2,a,c):
    sol = minimize(dE_sq,0.,(x1,x2,a,c))
    sol2 = minimize(dE_sq,x1,(x1,x2,a,c))
    sol3 = minimize(dE_sq,x2,(x1,x2,a,c))
    if (sol.fun < 1e-5)*(sol2.fun < 1e-5)*(sol3.fun < 1e-5):
        E_act = E(sol.x,x1,x2,a,c)[0]
        E1 = E(sol2.x,x1,x2,a,c)[0]
        E2 = E(sol3.x,x1,x2,a,c)[0]
        dx = x2-x1
        return E_act-E1,E_act-E2,dE,dx
    else:
        return np.nan,np.nan,np.nan


T = np.array([[0,0.2,0.4],
              [0.1,0,0.2],
              [0.001,0.05,0]])

transition_start,transition_end = np.where(~np.eye(T.shape[0]).astype(bool))
rates = T[transition_start,transition_end]
def cost(X,Is,Js,rates,temp=1):
    x1,x2,a,c = X[::4],X[1::4],X[2::4],X[3::4]
    res = np.array([get_features_from_E(_x1,_x2,_a,_c) for (_x1,_x2,_a,_c) in zip(x1,x2,a,c)])
    Eact1,Eact2,dE,dx = res[:,0],res[:,1],res[:,2]

    Tab = np.exp(-Eact1/temp)
    Tba = np.exp(-Eact1/temp)

a = 1
c = 0.5
x1 = -1
x2 = 1


Energy = E(x,-1,1,1,0.5)

x1,y1 = x.copy(),np.zeros_like(x)
x2,y2 = x.copy(),x.copy()*0.5

X,Y = np.concatenate((x1,x2)),np.concatenate((y1,y2))

from scipy.interpolate import bisplrep,bisplev

interp = bisplrep(X,Y,np.concatenate((Energy,Energy)))#,kx=5,ky=5)

x_extended = np.linspace(-2,2,100)

plt.imshow(bisplev(x_extended,x_extended,interp))
plt.show()
