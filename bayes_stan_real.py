
import argparse
import pystan
import pickle
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, action= 'store', dest='model_file', 
	help= 'Name of pickle file containing stan model')

parser.add_argument('-if', '--infile', type=str, action='store', dest='infile',
	help='Name of kinetic data file')

parser.add_argument('-pf', '--params_file',type=str, action='store', dest='params_file',
	help='Name of file containing parameter values')

parser.add_argument('-of', '--outfile',type=str,action='store',dest='outfile',
	help='Name of file to pickle ')

parser.add_argument('-n','--n_steps',type = int,action='store',dest='n_steps')

parser.add_argument('-w','--warmup',type = int, action='store',dest='n_warmup')

args = parser.parse_args()

#######################################################################################

# Load input file
from modules import read_amylofit_data
input_data = read_amylofit_data.read_data(args.infile)

# Read parameters to initialise simulation with
from modules import read_params
init_guesses, model_parameters = read_params.read_params_mcmc(args.params_file)

# Load stan model
with open(args.model_file,'rb') as file:
	sm = pickle.load(file)

# Set up parameters for Monte Carlo
model_data = {**model_parameters,**input_data,'sigma_xp':0.2}

# Run HMC to generate samples from posterior
fit = sm.sampling(data=model_data,warmup=args.n_warmup, iter=args.n_steps, chains=4, init=[init_guesses]*4)

with open('pickles/{}.pkl'.format(args.outfile),'wb') as file:
	pickle.dump(fit,file)
	
