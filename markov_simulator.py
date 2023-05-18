import numpy as np
from numba import jit,njit,types
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.sparse import coo_matrix
from itertools import combinations,permutations
from scipy.optimize import minimize
import matplotlib
import pandas as pd
import time
import os
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Helvetica Neue')


def mkdir(fldr):
    if not os.path.exists(fldr):
        os.mkdir(fldr)

@jit(nopython=True)
def get_D(T):
    DT = T - np.eye(T.shape[0],dtype=T.dtype)*T.sum(axis=1)
    return DT.T

@jit(nopython=True)
def dts(s,t,D):
    """
    Implicit ds/dt calculation. s is the number of cells in a given state.
    """
    return D@s


# @jit(nopython=True)
def sigmoid_signal(a,b,beta0,beta1,beta2,mn,amp):
    exponent = -(beta1*a + beta2*b - beta0)
    return mn + amp/(1+np.exp(exponent))

# @jit(nopython=True)
def get_signal_transition_rates(a,b,signalling_parameters):
    transition_rates = sigmoid_signal(np.expand_dims(a,1),np.expand_dims(b,1),*np.expand_dims(signalling_parameters,1))
    return transition_rates


@jit(nopython=True)
def dts_matrix(s_mat_flat,t,D_mat):
    """
    Implicit ds/dt calculation. s is the number of cells in a given state.
    """
    s_mat = s_mat_flat.reshape(D_mat.shape[0],-1)
    dts = np.zeros_like(s_mat)
    for i, (s,D) in enumerate(zip(s_mat,D_mat)):
        dts[i] = np.asfortranarray(D)@np.asfortranarray(s)
    return dts.ravel()


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



class Markov:
    def __init__(self,transitions_possible=None,run_params=None):
        self.markov_states = None
        self.transitions_possible = transitions_possible
        assert run_params is not None, "Specify run_params"
        self.run_params = run_params

        self.states,self.state_i,self.state_j = None,None,None
        self.i,self.j = None,None
        self.n_transitions = None
        self.make_transition_matrix_list()
        self.T = None
        self.randomly_fill_transition_matrix()
        self.t_span = None
        self.s0 = None
        self.s_solve = None
        self.specify_simulation_conditions()


    def make_transition_matrix_list(self):
        if "allow_all" in self.run_params:
            if self.run_params["allow_all"]:
                assert "states" in self.run_params
                self.states = self.run_params["states"]
                self.state_i,self.state_j = list(zip(*list(permutations(self.states,2))))
            else:
                assert self.transitions_possible is not None, "Specify transitions_possible"
                self.state_i,self.state_j = list(zip(*[t.split("->") for t in self.transitions_possible]))
                self.states = list(set(self.state_i+self.state_j))
                self.states = sorted(self.states)
                print("%d states detected"%len(self.states))
                print("States are: %s"%self.states)

        self.n_transitions = len(self.state_i)
        self.i = np.array([self.states.index(i) for i in self.state_i])
        self.j = np.array([self.states.index(i) for i in self.state_j])

    def randomly_fill_transition_matrix(self):
        accept = False
        k = 0
        while (accept is False) and (k<100):
            transition_rates = np.random.random(self.n_transitions)*self.run_params["init_mult"]
            T = coo_matrix((transition_rates,(self.i,self.j)),shape = (len(self.states),len(self.states)))
            T = T.toarray()
            if (T.sum(axis=1)<1).all():
                accept = True
            else:
                k += 1
        assert (T.sum(axis=1)<1).all(), "init_mult is too high"
        self.T = T

    def specify_simulation_conditions(self):
        self.t_span = np.arange(0,self.run_params["tfin"],self.run_params["dt"])
        assert self.run_params["initial_state"] in self.states, "initial_state is not in the list of states"
        self.s0 = np.zeros(len(self.states),dtype=np.float64)
        self.s0[self.states.index(self.run_params["initial_state"])] = 1.0

    def simulate(self):
        D = get_D(self.T)
        self.s_solve = odeint(dts, self.s0, self.t_span, args=(D,))

    def plot_simulation(self,ax):
        ax.spines[['right', 'top']].set_visible(False)
        for i, state in enumerate(self.states):
            ax.plot(self.t_span,self.s_solve[:,i],label=state)
        ax.legend()

    def plot_network(self,ax,colours=None):
        G = nx.DiGraph()
        G.add_edges_from(list(zip(self.state_i,self.state_j)))
        node_colours = [colours[self.states.index(key)] for key in list(G.nodes)]
        pos = nx.spring_layout(G)  # Set the layout algorithm
        nx.draw(G, pos, with_labels=True,ax=ax,node_color=colours)



class Markov_signalling:
    """
    Each signal is proposed to follow some saturating kinetics on fate transitions.
    Take the functional form of a logistic function. Assume that signals act independently i.e. there's no cross terms
    """
    def __init__(self,transitions_possible=None,run_params=None,signalling_params = None):
        self.markov = Markov(transitions_possible,run_params)
        self.signalling_params = signalling_params
        self.set_initial_signalling_params()
        self.signalling_parameters = None
        self.s0 = None
        self.T = None
        self.D = None
        self.s_solve = None
        self.final_vals = None
        self.final_vals_grid = None
        self.set_initial_signalling_params()
        self.make_transition_matrices()
        self.transition_matrix_grid = None
        self.make_transition_matrices_grid()


    def set_initial_signalling_params(self):
        signalling_parameters = []
        for i in range(5):
            lim = self.signalling_params["lims"][i]
            signalling_parameters.append(np.random.uniform(lim[0],lim[1],len(self.markov.i)))
        self.signalling_parameters = np.array(signalling_parameters)

    def make_transition_matrices(self):
        transition_rates = get_signal_transition_rates(self.signalling_params["a"],self.signalling_params["b"], self.signalling_parameters)
        self.T = np.array([coo_matrix((transition_rate, (self.markov.i, self.markov.j)), shape=(len(self.markov.states), len(self.markov.states))).toarray() for transition_rate in transition_rates])
        self.T /= self.T.sum(axis=2).max()
        self.D = np.array([get_D(Ti) for Ti in self.T])

    def simulate(self):
        self.s0 = np.zeros((len(self.D) * len(self.markov.states)))
        i_init = self.markov.states.index(self.markov.run_params["initial_state"])
        self.s0[i_init::len(self.markov.states)] = 1.0
        self.s_solve = odeint(dts_matrix, self.s0, self.markov.t_span, args=(self.D,))

    def get_terminal_state(self):
        self.final_vals = [self.s_solve[-1,i::len(self.markov.states)] for i in range(len(self.markov.states))]


    def get_terminal_state_grid(self):
        self.get_terminal_state()
        a_unique,b_unique = np.unique(self.signalling_params["a"]),np.unique(self.signalling_params["b"])
        a_mat,b_mat = np.meshgrid(a_unique,b_unique,indexing="ij")
        self.index_mat = np.zeros(a_mat.shape,dtype=np.int64)
        for ai, a in enumerate(a_unique):
            for bi,b in enumerate(b_unique):
                self.index_mat[ai,bi] = np.nonzero((self.signalling_params["a"] == a)*(self.signalling_params["b"]==b))[0]
        self.final_vals_grid = dict(zip(self.markov.states,[np.array([fv[i] for i in self.index_mat.ravel()]).reshape(self.index_mat.shape) for fv in self.final_vals]))

    def make_transition_matrices_grid(self):
        a_unique, b_unique = np.unique(self.signalling_params["a"]), np.unique(self.signalling_params["b"])
        a_mat, b_mat = np.meshgrid(a_unique, b_unique, indexing="ij")
        self.index_mat = np.zeros(a_mat.shape, dtype=np.int64)
        for ai, a in enumerate(a_unique):
            for bi, b in enumerate(b_unique):
                self.index_mat[ai, bi] = \
                np.nonzero((self.signalling_params["a"] == a) * (self.signalling_params["b"] == b))[0]
        self.transition_matrix_grid = np.array([np.array([flat_mat[i] for i in self.index_mat.ravel()]).reshape(self.index_mat.shape) for flat_mat in self.T.reshape(self.T.shape[0],-1).T]).reshape((self.T[0].shape)+(self.index_mat.shape))


class Markov_fit:
    def __init__(self,data_file=None,results_folder=None,init_parameter_lims=None,state_names=None,data_names=None,transitions_possible=None, run_params=None,opt_params=None,plot_params=None):
        assert data_file is not None
        assert init_parameter_lims is not None
        self.plot_params = plot_params
        self.results_folder = results_folder
        mkdir(results_folder)
        self.current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(time.time()))
        self.session_folder = results_folder + "/" + self.current_time
        mkdir(self.session_folder)
        mkdir(self.session_folder + "/plots")
        mkdir(self.session_folder + "/plots/fits")
        mkdir(self.session_folder + "/plots/transitions")
        mkdir(self.session_folder + "/plots/graphs")
        mkdir(self.session_folder + "/results")

        self.state_names = state_names
        self.data_names = data_names
        self.dictionary = dict(zip(self.state_names, self.data_names))

        self.df = pd.read_csv(data_file)
        self.signalling_params = {'lims': init_parameter_lims,  # amp, amplitude of the response at saturating signal.
                                    'a': self.df["a"].values, 'b': self.df["b"].values}
        self.transitions_possible, self.run_params = transitions_possible, run_params
        self.mrkvS = Markov_signalling(transitions_possible, run_params, self.signalling_params)


        if opt_params is None:
            opt_params = {"minimizer_params":{"maxiter":10000},##dial this for increasing precision.
                          "n_iter":8}
        self.opt_params = opt_params
        self.solid_colours = None
        self.get_solid_colours()
        self.plot_network()
        self.results = None
        self.mrkvSs, self.sP_opts,self.tfin_opts = None,None,None
        self.true_final_vals_grid = None
        self.make_data_grids()
        self.proportions_by_data_names = None

    def get_solid_colours(self):
        self.solid_colours = []
        for state in self.mrkvS.markov.states:
            if state in self.plot_params["colour_dict"]:
                self.solid_colours.append(self.plot_params["colour_dict"][state](100))
            else:
                self.solid_colours.append((0.82,0.82,0.82,1.0))

    def plot_network(self):
        fig, ax = plt.subplots(figsize=(4,4))
        self.mrkvS.markov.plot_network(ax,colours=self.solid_colours)
        fig.savefig(self.session_folder + "/plots/graphs/simulated_network.pdf",dpi=300)

    def fit(self):

        mrkvS = Markov_signalling(self.transitions_possible, self.run_params, self.signalling_params)

        def get_cost(X):
            tfin, sP = X[0], X[1:]
            _run_params = self.run_params.copy()
            _run_params["tfin"] = tfin
            mrkvS.signalling_parameters = sP.reshape(mrkvS.signalling_parameters.shape)
            mrkvS.make_transition_matrices()
            mrkvS.run_params = _run_params
            mrkvS.simulate()
            mrkvS.get_terminal_state()
            total_proportions = {}
            for data_name in self.data_names:
                total_proportions[data_name] = np.zeros_like(mrkvS.final_vals[0])
            for state_name in self.state_names:
                total_proportions[self.dictionary[state_name]] += mrkvS.final_vals[mrkvS.markov.states.index(state_name)]
            cost = 0
            for key, val in total_proportions.items():
                cost += (np.abs(val - self.df[key].values / 100) ** 2).sum()
            return cost

        mrkvS.set_initial_signalling_params()
        sP0 = mrkvS.signalling_parameters.ravel()
        X0 = np.zeros(sP0.size + 1)
        X0[0] = self.run_params["tfin"] ##I introduce an additional free parameter being the time of measurement, given the scales of the transitions are arbirary. This is poorly defined in principle but perhaps speeds up optimization?
        X0[1:] = sP0
        res = minimize(get_cost, X0, method="Nelder-Mead", options=self.opt_params["minimizer_params"])
        sP_opt = res.x[1:]
        tfin_opt = res.x[0]
        return mrkvS, sP_opt,tfin_opt,res.fun

    def fit_multiple(self):
        self.results = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self.fit)() for i in range(self.opt_params["n_iter"]))
        self.mrkvSs = [r[0] for r in self.results]
        self.sP_opts = [r[1] for r in self.results]
        self.tfin_opts = [r[2] for r in self.results]
        self.residuals = [r[3] for r in self.results]
        self.collapse_proportions_by_data_names()
        self.save_results()

    def save_results(self):
        for i in range(len(self.mrkvSs)):
            fldr = self.session_folder + "/results/%i"%i
            mkdir(fldr)

            df_sP = pd.DataFrame(self.sP_opts[i].reshape(self.mrkvSs[0].signalling_parameters.shape))
            df_sP.columns = self.mrkvSs[i].markov.transitions_possible
            df_sP.index = ["beta0","beta1","beta2","mn","amp"]
            df_sP.to_csv(fldr + "/signal_params.csv")

            tfin_file = open(fldr + "/tfin.txt","w+")
            tfin_file.write(str(self.tfin_opts[i]))
            tfin_file.close()

            mkdir(fldr + "/final_proportion")
            for nm in self.state_names:
                pd.DataFrame(self.mrkvSs[i].final_vals_grid[nm]).to_csv(fldr + "/final_proportion/%s.csv"%nm)

            mkdir(fldr + "/final_proportion_grouped")

            for nm in self.data_names:
                pd.DataFrame(self.proportions_by_data_names[nm][i]).to_csv(fldr + "/final_proportion_grouped/%s.csv"%nm)
        pd.DataFrame({"index":np.arange(len(self.residuals)),"residual":self.residuals}).to_csv(self.session_folder + "/results/residuals.csv")

    def make_data_grids(self):
        a_unique, b_unique = np.unique(self.signalling_params["a"]), np.unique(self.signalling_params["b"])
        a_mat, b_mat = np.meshgrid(a_unique, b_unique, indexing="ij")
        index_mat = np.zeros(a_mat.shape, dtype=np.int64)
        for ai, a in enumerate(a_unique):
            for bi, b in enumerate(b_unique):
                index_mat[ai, bi] = np.nonzero((self.signalling_params["a"] == a) * (self.signalling_params["b"] == b))[0]
        self.true_final_vals_grid = dict(zip(self.data_names,[np.array([self.df[key].values[i] for i in index_mat.ravel()]).reshape(index_mat.shape)/100 for key in self.data_names]))

    def collapse_proportions_by_data_names(self):
        self.proportions_by_data_names = {}
        for mrkvS in self.mrkvSs:
            mrkvS.get_terminal_state_grid()
        for nm in self.data_names:
            self.proportions_by_data_names[nm] = np.zeros((len(self.mrkvSs),)+self.mrkvSs[0].final_vals_grid[self.state_names[0]].shape)
        for i, mrkvS in enumerate(self.mrkvSs):
            for nm in self.state_names:
                self.proportions_by_data_names[self.dictionary[nm]][i] += mrkvS.final_vals_grid[nm]


    def plot_fits(self):
        a_unique, b_unique = np.unique(self.df["a"].values), np.unique(self.df["b"].values)
        extent, aspect = make_extent(a_unique, b_unique, "linear", "linear")
        vmax = self.plot_params["vmax"]
        for k in range(len(self.mrkvSs)):

            cmaps = [self.plot_params["colour_dict"][nm] for nm in self.data_names]
            fig, ax = plt.subplots(2,len(self.data_names),sharex=True,sharey=True)
            for i, nm in enumerate(self.data_names):
                ax[1,i].imshow(np.flip(self.proportions_by_data_names[nm][k].T,axis=0),vmin=0,vmax=vmax,cmap=cmaps[i],aspect=aspect,extent=extent)
                ax[0,i].imshow(np.flip(self.true_final_vals_grid[nm].T,axis=0),vmin=0,vmax=vmax,cmap=cmaps[i],aspect=aspect,extent=extent)
                ax[0,i].set_title(nm)
            ax[0,0].set(ylabel="Data\nBMP4")
            ax[1,0].set(ylabel="Sim\nBMP4")
            for axx in ax[1]:
                axx.set(xlabel="Activin")
            fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.7,hspace=0.1,wspace=0.1)
            fig.savefig(self.session_folder + "/plots/fits/%d.pdf"%k, dpi=300)

    def plot_transitions(self):
        a_unique, b_unique = np.unique(self.df["a"].values), np.unique(self.df["b"].values)
        extent, aspect = make_extent(a_unique, b_unique, "linear", "linear")

        for k,mrkvS in enumerate(self.mrkvSs):

            fig, ax = plt.subplots(len(mrkvS.markov.states),len(mrkvS.markov.states),sharex=True,sharey=True)
            for i in range(len(mrkvS.markov.states)):
                for j in range(len(mrkvS.markov.states)):
                    ax[i,j].imshow(np.flip(mrkvS.transition_matrix_grid[i,j].T,axis=0),vmin=0,vmax=mrkvS.transition_matrix_grid.max(),extent=extent,aspect=aspect,cmap=plt.cm.Greens)
                    ax[-1,j].set(xlabel="BMP\n"+mrkvS.markov.states[j])
                ax[i,0].set(ylabel=mrkvS.markov.states[i]+"\nActivin")
            # fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.7,hspace=0.1,wspace=0.1)

            fig.savefig(self.session_folder + "/plots/transitions/%i.pdf"%k,dpi=300)

#
# transitions_possible = ("P->A","P->M","P->E","M->E","P->X","P->Y")
# run_params = {"initial_state":"P",
#                "dt":0.1,
#                "tfin":3,
#                "allow_all":False,
#                "init_mult":0.1,
#               "states":["P","A","E","M","X"]}
#
# plot_params = {"colour_dict":{"A":plt.cm.Oranges,
#                "M":plt.cm.Reds,
#                "E":plt.cm.Blues},
#                "vmax":0.7}
#
# opt_params = {"minimizer_params": {"maxiter": 100},  ##dial this for increasing precision.
#               "n_iter": 8}
#
#
# init_parameter_lims= [(0, 5),  # Beta0, These are the bounds of the initial guess of the parameters.
#                       (0, 0.1),  # beta1, dependency on a
#                       (0, 0.1),  # beta2, dependency on b
#                       (0.05, 0.2),  # mn, minimal value
#                       (0, 0.4)]  # amp, amplitude of the response at saturating signal.
#
# state_names = ["A","E","M"] ##these are the names of the states that you want to compare to the data
# data_names = ["A","E","M"] ##these are the corresponding names of the columns in your data sheet.
#
# ##^^ This is set up such that you can assign multiple states to a given data column
# ## for example, state_names = ["A","M","ME"]; data_names = ["A","M","M"]
#
# markov_fit = Markov_fit(data_file="data/proportions_data.csv",
#                         results_folder="results",
#                         init_parameter_lims=init_parameter_lims,
#                         state_names=state_names,
#                         data_names=data_names,
#                         transitions_possible=transitions_possible,
#                         run_params=run_params,
#                         opt_params=opt_params,
#                         plot_params=plot_params)
# markov_fit.fit_multiple()
# markov_fit.plot_fits()
# markov_fit.plot_transitions()

