# import required packages
import numpy as np
import pandas as pd
import sys
import argparse
import bayesian_utility_fcts
from scipy import optimize as opt

###################################
## User hardcoded input
min_MSE = 1
max_iter= 25
Lagrange_decrease_factor = 0.5 # lagrangian gets multiplied by this value at each iteration; should be <1.
flat_prior = True # if True target distribution is flat, if false, the cell target distribution will be used
max_half_time = 240*60 # from 1000 h
min_half_time = 0.01*60 # to 0.6 min
washing_time = 4
cell_doubling_time = 'inf'
# inf for myo; 20*60 otherwise
# Good input for entropy weighting: 10^7 (for 0.5 decrease factor)
# 40 bins: no point in intermediate number of basin hops. E.g. 1
# 20-30 bins: maybe have a few basinhops . But can play around to optimize
###################################
min_k = np.log(2)/max_half_time
max_k = np.log(2)/min_half_time
if cell_doubling_time == 'inf':
	cell_division_rate = 0
else:
	cell_division_rate = np.log(2)/cell_doubling_time

###################################
## Reading in command line variables
full_description = 'This program reads in data from pulse chase experiments and uses a maximum entropy based MAP algorithm to determine the distribution of intra-cellular half times of proteins.'

parser = argparse.ArgumentParser(description=full_description)

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument('-f', '--filename', type=str, help='Datafile to load.', required=True)
requiredNamed.add_argument('-o', '--output_file', type=str, help='File to save analysis results to.', required=True)
parser.add_argument('-plotfit', '--plot_fit_checker', action='store_true', dest='plot_fit_checker', help='Flag for plotting fits. If this is not given, no plot will be saved.')
requiredNamed.add_argument('-bh', '--basinhops', type=int, help='Number of basinhops.', required=True)
requiredNamed.add_argument('-N', '--N_bins', type=int, help='Number of bins.', required=True)
requiredNamed.add_argument('-L', '--Lagrange_mult', type=float, help='Weighing of the entropy contribution.', required=True)



args = parser.parse_args()
for arg in vars(args):
	print('{}: {}'.format(arg,getattr(args,arg)))

N_bins = args.N_bins
basinhops_MAP = args.basinhops
Lagrange_mult = args.Lagrange_mult

errors_and_warnings = 'Summary of all errors and warnings:' # string for printing to terminal
errors_and_warnings_dict = dict() # dict to add into parameter output

##################################
## reading in data ##
data = bayesian_utility_fcts.load_data(args.filename, use_first_column_as_row_name=False, use_first_row_as_headers=True, delimiter=',') # reads data and puts it into array
indices = list(data.index)
initial_size = len(data) # used only if verbose
noise = np.sqrt(np.array(data['noise']))
data['chase_time'] = data['chase_time'] + washing_time # the washing time needs to be added onto the experimentally recorded chase time (which has zero when recarding starts)


# generate the bins
log_space_bins_boundaries = np.linspace(np.log(min_k),np.log(max_k),N_bins+1)
constant_model_para = {'bin_boundaries': np.exp(log_space_bins_boundaries), 'cell_division_rate':cell_division_rate}
print('Bin boundaries at:', constant_model_para['bin_boundaries'])

# parameters of the probability distributions (i.e. the prior) of the model parameters
# these are the parameters defining the priors
if flat_prior:
	# flat pior
	target_distr = [1/N_bins]*N_bins
	# use above parameters to choose starting point for MC/MAP somewhere near max of prior
	initial_parameters = [1/N_bins]*N_bins

else:
	# cell paper prior
	cell_prior=pd.read_csv('cellprior_35bins_240to0p01.csv', header=0)
	target_distr=cell_prior['cell_prior'].tolist()
	# use above parameters to choose starting point for MC/MAP somewhere near max of prior
	initial_parameters = cell_prior['cell_prior'].tolist()
	
print('Target distribution:',target_distr)

user_parameters = {'noise': noise, 'Lagrange_mult': Lagrange_mult, 'constrain_weight':Lagrange_mult, 'target_distr':target_distr}



# initialize output file
output = pd.DataFrame({'bin_boundaries':[i for i in constant_model_para['bin_boundaries']]+[0,0,0,0,0]})
output['bin_bounds_in_min'] = [1/i*np.log(2) for i in constant_model_para['bin_boundaries']]+[0,0,0,0,0]
output.index = ['bin{}'.format(i) for i in range(N_bins+1)]+['sum_of_weights','entroy_weighting','MSE','entropy','posterior']


########################################
## Data fitting/analysis
data_fits_MAP = dict() # dict that stores the fitted parameters and the fitting function to be used in the plotting
parameters_MAP = dict() # array that stores the parameters,
print('\n***Performing maximum a posteriori fit.')
# initialize the model function
model_class = bayesian_utility_fcts.cell_turnover(constant_model_para)

check=min_MSE+1
N_iter = 0
while check > min_MSE and N_iter < max_iter: # adjusts the Lagrangian (i.e. weighing of the entropy) to give desired error
	# adjusting Lagrangian
	user_parameters['Lagrange_mult']=user_parameters['Lagrange_mult']*Lagrange_decrease_factor
	print('Lagrange multiplier:',user_parameters['Lagrange_mult'])

	#define constraints if needed
	constraints = model_class.constraints

	#initialize posterior probability with correct hypothesis (i.e. model_function), prior, likelihood and data
	#without jacobian
#	post = bayesian_utility_fcts.posterior('prior_max_entropy', user_parameters, 'likelihood_given_noise_vector', model_class.relative_signal, data[['pulse_time','chase_time']], data['measurement'], log_prob = True)
	#with jacobian
	post = bayesian_utility_fcts.posterior('prior_max_entropy', user_parameters, 'likelihood_given_noise_vector', model_class.relative_signal_synth, data[['pulse_time','chase_time']], data['measurement'], log_prob = True, jacobian = model_class.jacobian_relative_signal)

	#do standard MAP
	fitted_all_info = bayesian_utility_fcts.MAP_fitting(post.neg_total_prob_num_input, initial_parameters, basinhops_MAP, logstep=False)

	#do constrained MAP
#	fitted_all_info = bayesian_utility_fcts.MAP_fitting_constrained(post.neg_total_prob_num_input, post.total_prob_jacobian, initial_parameters, basinhops_MAP, constraints, logstep=False)

#	fitted_all_info = opt.minimize(post.neg_total_prob_num_input, initial_parameters,  method='Nelder-Mead')
#	print(fitted_all_info)

	para_fit_raw = abs(fitted_all_info['x'])
	para_fit = para_fit_raw/np.sum(para_fit_raw)
	initial_parameters = para_fit
	MSE = -post.log_likelihood_given_noise_vector(para_fit)/len(data['pulse_time'])
	print('MSE',MSE)

	#Plotting data and fits
	model = {'function':model_class.relative_signal_synth, 'parameters':para_fit}
	bayesian_utility_fcts.plot_data_and_fits_mulit_t_input({'myo':data}, y_name='measurement', save_to=args.output_file.replace('.csv','')+ '_fit_{}.png'.format(N_iter), model={'myo':model}, x_name='chase_time', times_to_eval_model={'myo':data})

	output['iteration_{}'.format(N_iter)] = [val for val in para_fit] + [0,np.sum(para_fit),user_parameters['Lagrange_mult'],MSE,post.log_prior_max_entropy(para_fit)/user_parameters['Lagrange_mult'], post.neg_total_prob_num_input(para_fit,None)]#['sum_of_weights','entroy_weighting','MSE','prior','posterior']

	N_iter += 1


#model = {'function':model_class.relative_signal, 'parameters':para_fit}
#model = {'function':model_class.test, 'parameters':para_fit}

########################################
## Saving
print(output)
output.to_csv(args.output_file, index = True)

# # function
# data['model'] = model['function'](model['parameters'],data)
# data.to_csv(args.filename.replace('.csv','_fits.csv'), index=False)


# ########################################
# ## Saving
# print('Plotting data and fits.')
# bayesian_utility_fcts.plot_data_and_fits_mulit_t_input({'blah':data}, y_name='measurement', save_to=args.output_file.replace('.csv', '_fit.png'), model={'blah':model}, x_name='chase_time', times_to_eval_model={'blah':data})
# #bayesian_utility_fcts.plot_data_and_fits_mulit_t_input({'slow': data[data['pulse_time']==1200], 'medium': data[data['pulse_time']==120],'fast':data[data['pulse_time']==20]}, y_name='measurement', save_to=args.output_file.replace('.csv', '_fit.png'), model={'slow':model,'medium':model,'fast':model}, x_name='chase_time', times_to_eval_model={'slow': data[data['pulse_time']==1200], 'medium': data[data['pulse_time']==120],'fast':data[data['pulse_time']==20]})
