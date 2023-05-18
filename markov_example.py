from markov_simulator import Markov_fit
import numpy as np
import matplotlib.pyplot as plt

transitions_possible = ("P->A","P->M","P->E","M->E","P->X")
run_params = {"initial_state":"P", ##This is the state that all of the cells are initially found in
               "dt":0.1, ##set your time discretisation
               "tfin":3, ##set the final time (although this dynamically changes in the simulation)
               "allow_all":False,##you can over-ride the above specification of which transitions are possible or not, saying that all transitions are possible.
               "init_mult":0.1, ##Some unnecessary parameter used in initialisation, basically preventing the rates going too high
              "states":["P","A","E","M","X"]## in the case of having 'allow_all' be True, then you need to specify what are the states
              }

##Set the plot parameters, specifying the colours of your various (measurable) variables. Can use whichever mpl colourmap you want.
plot_params = {"colour_dict":{"A":plt.cm.Oranges,
               "M":plt.cm.Reds,
               "E":plt.cm.Blues},
               "vmax":0.7 ##this caps the highest proportion plotted.
               }

opt_params = {"minimizer_params": {"maxiter": 1000},  ##dial this for increasing precision.
              "n_iter": 8 ##The number of random initialisations you would like to run. This all runs in parallel.
              }


init_parameter_lims= [(0, 5),  # Beta0, These are the bounds of the initial guess of the parameters.
                      (0, 0.1),  # beta1, dependency on a
                      (0, 0.1),  # beta2, dependency on b
                      (0.05, 0.2),  # mn, minimal value
                      (0, 0.4)]  # amp, amplitude of the response at saturating signal.

state_names = ["A","E","M"] ##these are the names of the states that you want to compare to the data
data_names = ["A","E","M"] ##these are the corresponding names of the columns in your data sheet.

##^^ This is set up such that you can assign multiple states to a given data column
## for example, state_names = ["A","M","ME"]; data_names = ["A","M","M"]

markov_fit = Markov_fit(data_file="data/proportions_data.csv",
                        results_folder="results",
                        init_parameter_lims=init_parameter_lims,
                        state_names=state_names,
                        data_names=data_names,
                        transitions_possible=transitions_possible,
                        run_params=run_params,
                        opt_params=opt_params,
                        plot_params=plot_params)
markov_fit.fit_multiple()
markov_fit.plot_fits()
markov_fit.plot_transitions()
