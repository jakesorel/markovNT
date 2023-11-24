import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functools import partial

import jax.numpy as jnp
from jax import grad as jgrad
from jax import hessian, jacrev,jacfwd
from jax import grad, vmap
from scipy.integrate import solve_ivp
import time
from jax import jit
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba

@jit
def switch(x):
    return 0.5*(1+jnp.tanh(x))

@jit
def sigmoid_signal(R,S,R0,S0,R_m,S_m,vmax):
    """This acts like an 'and' gate, with higher R_m/S_m toggling sharper boundaries"""
    return switch((R-R0)*R_m)*switch((S-S0)*S_m)*vmax


def make_extent(x_range,y_range,xscale="linear",yscale="linear",center=True):
    if xscale == "log":
        x_range = np.log10(x_range)
    if yscale == "log":
        y_range = np.log10(y_range)
    if center is False:
        extent = [x_range[0],x_range[-1]+x_range[1]-x_range[0],y_range[0],y_range[-1]+y_range[1]-y_range[0]]
    else:
        extent = [x_range[0]-(x_range[1]-x_range[0])/2,x_range[-1]+(x_range[1]-x_range[0])/2,y_range[0]-(y_range[1]-y_range[0])/2,y_range[-1]+(y_range[1]-y_range[0])/2]

    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    return extent,aspect
@jit
def Shh_spacetime(x,t,growth_rate=0.51,C0_0=-23.,C0_grad=27,l_eff0=0.163):
    """
    From Cohen et al. Nat Com, the absolute lengthscale = 20 microns across time
    Reported sizes:
    125µm = E8.5
    200µm = E9.5
    350µm = E10.5

    L = 123e^(0.51*t) ##very approximately, taken from Cohen et al.

    consequently, l_eff = l/L = 20/(123*e^(0.51*t)) = 0.163 e^(-0.51*t)

    very approximately, from Cohen too (eye-balled from the paper):
    125µm -> C0 = 4
    200µm --> 20
    250µm --> 30
    300µm --> 42
    350µm --> 53
    400µm --> 64

    C0 = -23 + 0.22*L
    C0 = -23 + 27*e^(0.51*t)
    """

    C0 = C0_0 + C0_grad*jnp.exp(growth_rate*t)
    l_eff = l_eff0 *jnp.exp(-growth_rate*t)
    return C0*jnp.exp(-x/l_eff)


@jit
def RA_spacetime(x,t,l_0,l_grad,R_max=1.):
    l_eff = (l_0+l_grad*t)
    return switch((x-l_eff)*1000)*R_max

# @jit(nopython=True)
# def dts_matrix(s_mat_flat,t,D_mat):
#     """
#     Implicit ds/dt calculation. s is the number of cells in a given state.
#     """
#     s_mat = s_mat_flat.reshape(D_mat.shape[0],-1)
#     dts = np.zeros_like(s_mat)
#     for i, (s,D) in enumerate(zip(s_mat,D_mat)):
#         dts[i] = np.asfortranarray(D)@np.asfortranarray(s)
#     return dts.ravel()

class NTMarkov:
    def __init__(self,morphogen_params,transition_params,states):
        self.morphogen_params = morphogen_params
        self.states = states
        self.transition_params = transition_params
        self.from_state,self.to_state,self.n_states = [],[],[]
        self.transitions_to_matrix,self.param_matrix = [],[]
        self.generate_transitions()


    def generate_transitions(self):
        """
        transition_params will be in the form

        dictionary: {"A->B":[R0,S0,R_m,S_m,vmax],...}
        """

        self.from_state,self.to_state = ["nm"]*len(self.transition_params),["nm"]*len(self.transition_params)
        for i, key in enumerate(self.transition_params.keys()):
            self.from_state[i],self.to_state[i] = key.split("->")

        # self.states = list(set(list(self.from_state)).union(list(self.to_state)))
        self.n_states = len(self.states)

        self.transitions_to_matrix = np.zeros((len(self.transition_params),self.n_states,self.n_states))

        self.param_matrix = np.zeros((len(self.transition_params),5))

        for i, (key,vals) in enumerate(self.transition_params.items()):
            _from,_to = self.from_state[i],self.to_state[i]
            print(_from,_to,self.states)
            fi,ti = self.states.index(_from),self.states.index(_to)
            self.transitions_to_matrix[i,fi,ti] = 1
            self.param_matrix[i] = vals


@partial(jit,static_argnums=(6,7,8))
def f(t,_y,x,transitions_to_matrix,param_matrix,morphogen_params,n_x,n_states,n_transitions):
    y = _y.reshape(n_x,n_states)
    S = Shh_spacetime(x, t, growth_rate=morphogen_params["growth_rate"], C0_0=morphogen_params["S_C0_0"], C0_grad=morphogen_params["S_C0_grad"], l_eff0=morphogen_params["S_l_eff0"])
    R = RA_spacetime(x, t,l_0=morphogen_params["R_l_0"],l_grad=morphogen_params["R_l_grad"],R_max=morphogen_params["R_max"])

    transition_rates = sigmoid_signal(jnp.expand_dims(R,1),jnp.expand_dims(S,1),*param_matrix.T) ##should give something of shape n_transitions x n_x

    transition_matrix = jnp.matmul(transition_rates,transitions_to_matrix.reshape(n_transitions,n_states*n_states)).reshape(n_x,n_states,n_states)

    D = transition_matrix - jnp.eye(transition_matrix.shape[-1])*jnp.expand_dims(transition_matrix.sum(axis=-1),-1)
    D = D.transpose(0,2,1)

    dty = jnp.row_stack([DD@yy for DD,yy in zip(D,y)])
    return dty.ravel()


morphogen_params = {"growth_rate":0.51, ##roughly speaking the doubling time of the neural tube, 'extracted' from Cohen
                    "S_C0_0":-23.0, #Amplitude of the NT at E0 (setting the amplitude at E8.5), Cohen
                    "S_C0_grad":27.0, #Increase in the amplitude of Shh with time, Cohen
                    "S_l_eff0":0.163, #Sets the initial decay length of the neural tube in rescaled units, Cohen
                    "R_l_0":0.0, ## Position of RA boundary at t0
                    "R_l_grad":0.2, ##Change in position of RA boundary with time
                    "R_max":1.0 ##Maximum RA level
                    }




transition_params = {"U->D":[0.2,0.,10.,1.,30],#[R0,S0,R_m,S_m,vmax]
                     "D->pMN":[0.2,4.,10.,1.,100.0],
                     "pMN->p3":[0.2,10.,10.,1.,100.0]}

states = ["U","D","pMN","p3"]
n_states = len(states)
colors = ["#bcbcbc","#f6b26b","#b20000","#6aa84f"]
n_transitions = len(transition_params)

model = NTMarkov(morphogen_params, transition_params, states)


x = np.linspace(0,0.4,250)
n_x = x.size

s0 = jnp.column_stack([np.ones(n_x),np.zeros(n_x),np.zeros(n_x),np.zeros(n_x)])
_s0 = s0.ravel()

tfin = 3
dt = 0.01
t_eval = np.arange(0,tfin,dt)
t_span = [t_eval[0],t_eval[-1]]

sol = solve_ivp(f,t_span,_s0,method="LSODA",t_eval=t_eval,args=(x,model.transitions_to_matrix,model.param_matrix,model.morphogen_params,n_x,model.n_states,n_transitions))

y_t = sol.y.reshape(n_x,n_states,len(t_eval))

fig, ax = plt.subplots()
for i in range(n_states):
    ax.plot(t_eval,y_t[2, i, :],label=states[i])
ax.legend()
fig.show()


def create_alpha_colormap(hex_color, n_steps=100):
    # Convert hex color to RGBA
    rgba_color = to_rgba(hex_color) # Add alpha = 0

    # Create colormap with n_steps points ranging from 0 to 1
    alphas = np.linspace(0, 1, n_steps)
    colors = np.tile(rgba_color, (n_steps, 1))
    colors[:, 3] = alphas

    colormap = LinearSegmentedColormap.from_list('alpha_colormap', colors)

    return colormap

most_abundant_state = np.argmax(y_t,axis=1)


extent,aspect=make_extent(t_eval,x)
fig, ax = plt.subplots(figsize=(4,4))
for i, (state, color) in enumerate(zip(states,colors)):
    ax.imshow(np.flip((most_abundant_state==i),axis=0),cmap=create_alpha_colormap(color),extent=extent,aspect=aspect,label=states)
ax.set(xlabel="Time (days)",ylabel="x/L")
fig.subplots_adjust(bottom=0.3,left=0.3,top=0.8,right=0.8)
fig.show()

#
# fig, ax = plt.subplots()
# for i, (state, color) in enumerate(zip(states,colors)):
#     ax.imshow(np.flip((y_t[:,i]),axis=0),cmap=create_alpha_colormap(color),extent=extent,aspect=aspect)
# fig.show()

X,T = np.meshgrid(x,t_eval,indexing="ij")

S = Shh_spacetime(X,T, growth_rate=morphogen_params["growth_rate"], C0_0=morphogen_params["S_C0_0"],
                  C0_grad=morphogen_params["S_C0_grad"], l_eff0=morphogen_params["S_l_eff0"])
R = RA_spacetime(X,T, l_0=morphogen_params["R_l_0"], l_grad=morphogen_params["R_l_grad"],
                 R_max=morphogen_params["R_max"])


fig, ax = plt.subplots(1,2,figsize=(8,3))
n_plot=10
for i in range(n_plot):
    ax[0].plot(x,S[:,int(i/n_plot*S.shape[1])],color=plt.cm.viridis(i/n_plot))
    ax[1].plot(x,R[:,int(i/n_plot*R.shape[1])],color=plt.cm.viridis(i/n_plot))
fig.subplots_adjust(bottom=0.3,left=0.3,top=0.8,right=0.8,wspace=0.5)
ax[0].set(xlabel="x/L",ylabel="Shh")
ax[1].set(xlabel="x/L",ylabel="RA")

fig.show()