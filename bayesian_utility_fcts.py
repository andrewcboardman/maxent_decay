# import required packages
import numpy as np
import pandas as pd
import sys 
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy import optimize as opt
from scipy import stats, special
## A collection of functions to be used by various other scripts

#TODO:
class rearrange_model_input:
	'''
	changes input for model functions from taking (t, p1, p2, p3, p4) to taking 
	1.) If name argument is given:
		({'p1':p1, 'p2':p2, 'p4':p4}, {'p3':p3,'time':t}) 
		where p1,p2,p4 are parameters to be varied and p3 is constant
		and names argument contains the names of the parameters in the correct order i.e. ['p1','p2','p3','p4']
		the parameter names can be whatever, but the x values need to be named 'time'
	2.) If name argument is not given:
		([p1,p2,p3,p4],t); mostly for backwards compatibility
	'''
	def __init__(self, model_to_rearrange, names=None):
		self.model_to_rearrange = model_to_rearrange
		self.names = names

	def rearranged_model(self,para,t_and_const):
		if self.names:
			para_input = list()
			for i in self.names:
				if i in para.keys():	
					para_input.append(para[i])
				elif i in t_and_const.keys():
					para_input.append(t_and_const[i])
			para_input = [t_and_const['time']]+para_input # combine into one list, to have correct input format
		else: # in this case we assume that all parameters are to be varied and are provided as a list (for backwards compatibility)
			para_input = [t_and_const]+para
		return self.model_to_rearrange(*para_input) # takes a list of values e.g. (t, mo, P0, M0, kn, nc, kp, koff)
	
def pastel(colour, weight=2.4): # just needed for plotting - ignore
    #Convert colour into a nice pastel shade
    rgb = np.asarray(colorConverter.to_rgb(colour))
    # scale colour
    maxc = max(rgb)
    if maxc < 1.0 and maxc > 0:
        # scale colour
        scale = 1.0 / maxc
        rgb = rgb * scale
    # now decrease saturation
    total = rgb.sum()
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [c + (x * (1.0-c)) for c in rgb]

    return rgb


def load_data(data_file, use_first_column_as_row_name=False, use_first_row_as_headers=True, delimiter=','): #UNCHANGED
	'''
	Loads a csv file, returns a pandas array; effectively just makes syntax more intuitive and includes loading of file. 
	If use_first_column_as_row_name is true, the first column will be used as the pandas index.
	If use_first_row_as_headers is true, the first row will be used as column names.
	'''

	open_file = open(data_file)

	if use_first_column_as_row_name:
		if use_first_row_as_headers:
			data = pd.read_csv(open_file, index_col=0, delimiter=delimiter)
		else:
			data = pd.read_csv(open_file, index_col=0,header=None, delimiter=delimiter)
	else:
		if use_first_row_as_headers:
			data = pd.read_csv(open_file, delimiter=delimiter)
		else:
			data = pd.read_csv(open_file,header=None, delimiter=delimiter)

	open_file.flush()
	open_file.close()
	return data

### Least squares Fitting

class MyTakeStep(object): #UNCHANGED
	'''
	changes step taking algorithm for MC bit of basinhopping 
	to sample parameters in log space rather than linear space
	'''
	def __init__(self, step_size=1.0):
    		self.step_size = step_size
    		self.step = 0
	def __call__(self, x):
		self.step += 1
		r = (self.step_size*np.random.rand(*x.shape)+1.0)**(np.random.choice([-1,1], size=x.shape))
		a = r*x
		return a

def straight_line(para, times): # for testing only
	'''
	The function to be fitted in this case straight line. 
	'''
	a = abs(para[0])
	b = para[1]
	return a + b*times

def straight_line_individual_args(times, a, b): # for testing only
	'''
	The function to be fitted in this case straight line. 
	'''
	return a + b*times

class cell_turnover:
	def __init__(self, const_para):
		self.sigma = const_para['bin_boundaries']
		self.theta = const_para['cell_division_rate']
		self.constraints = ({'type': 'eq','fun' : lambda h: np.array([np.sum(h)-1]),'jac' : lambda h: np.ones(len(h))})

	def relative_signal(self, para, times): # this is what the measurement yields. signal is normalised to 0 at time 0
		t = np.array(times['chase_time'])
		tau = np.array(times['pulse_time'])
		h = para
		M = 0 # total intracell protein
		for j in range(len(h)):
			temp = (special.expi(-(t)*(self.theta+self.sigma[j+1]))-special.expi(-(t)*(self.theta+self.sigma[j]))-special.expi(-(t+tau)*(self.theta+self.sigma[j+1]))+special.expi(-(t+tau)*(self.theta+self.sigma[j])))*np.exp(self.theta*(t+tau))*abs(h[j])/np.log(10)
			M += temp
		rel_sig = np.zeros_like(M)
		# This next bit takes care of the fact that the data is composed of 3 datasets with different pulse times and they need to be normalised separately 
		rel_sig[:10] = 100*(1-M[:10]/M[0])
		rel_sig[10:20] = 100*(1-M[10:20]/M[10])
		rel_sig[20:] = 100*(1-M[20:]/M[20])
		return rel_sig

	def relative_signal_synth(self, para, times): # this is what the measurement yields. signal is normalised to 0 at time 0
		t = np.array(times['chase_time'])
		tau = np.array(times['pulse_time'])
		h = para
		M = 0 # total intracell protein
		if self.theta == 0:
			for j in range(len(h)):
				temp = (-(t)*special.expi(-(t)*(self.sigma[j+1]))+(t)*special.expi(-(t)*(self.sigma[j]))+(t+tau)*special.expi(-(t+tau)*(self.sigma[j+1]))-(t+tau)*special.expi(-(t+tau)*(self.sigma[j]))+np.exp(-(t+tau)*(self.sigma[j+1]))/(self.sigma[j+1])-np.exp(-(t)*(self.sigma[j+1]))/(self.sigma[j+1])-np.exp(-(t+tau)*(self.sigma[j]))/(self.sigma[j])+np.exp(-(t)*(self.sigma[j]))/(self.sigma[j]) )*abs(h[j])/np.log(10)
				M += temp
		else:
			for j in range(len(h)):
				temp = (-special.expi(-(t)*(self.theta+self.sigma[j+1]))+special.expi(-(t)*(self.theta+self.sigma[j]))+special.expi(-(t+tau)*(self.theta+self.sigma[j+1]))-special.expi(-(t+tau)*(self.theta+self.sigma[j]))-np.exp(-(t+tau)*self.theta)*special.expi(-(t+tau)*self.sigma[j+1])+np.exp(-(t)*self.theta)*special.expi(-(t)*self.sigma[j+1])+np.exp(-(t+tau)*self.theta)*special.expi(-(t+tau)*self.sigma[j])-np.exp(-(t)*self.theta)*special.expi(-(t)*self.sigma[j]))*np.exp(self.theta*(t+tau))*abs(h[j])/(self.theta*np.log(10))
				M += temp
		rel_sig = np.zeros_like(M)
		# This next bit takes care of the fact that the data is composed of 3 datasets with different pulse times and they need to be normalised separately 
		rel_sig[:10] = 100*(1-M[:10]/M[0])
		rel_sig[10:20] = 100*(1-M[10:20]/M[10])
		rel_sig[20:] = 100*(1-M[20:]/M[20])
		return rel_sig

	def jacobian_relative_signal(self, para, times): # vector of the derivatives of the above fucntion wrt to each of the parameters to be minimised
		t = np.array(times['chase_time'])
		tau = np.array(times['pulse_time'])
		h = para

		dM_dhi = [] # derivative of total intracell protein wrt each parameter
		for j in range(len(h)):
			dM_dhi.append((special.expi(-(t)*(self.theta+self.sigma[j+1]))-special.expi(-(t)*(self.theta+self.sigma[j]))-special.expi(-(t+tau)*(self.theta+self.sigma[j+1]))+special.expi(-(t+tau)*(self.theta+self.sigma[j])))*np.exp(self.theta*(t+tau))/np.log(10))
		
		M = 0 # total intracell protein
		for j in range(len(h)):
			temp = (special.expi(-(t)*(self.theta+self.sigma[j+1]))-special.expi(-(t)*(self.theta+self.sigma[j]))-special.expi(-(t+tau)*(self.theta+self.sigma[j+1]))+special.expi(-(t+tau)*(self.theta+self.sigma[j])))*np.exp(self.theta*(t+tau))*abs(h[j])/np.log(10)
			M += temp

		rel_sig_dhi = []
		for dM_dhi_single in dM_dhi:
			temp = np.zeros_like(j)
			# This next bit takes care of the fact that the data is composed of 3 datasets with different pulse times and they need to be normalised separately 
			temp[:10] = -100/M[0]*(dM_dhi_single[:10]-M[:10]/M[0]*dM_dhi_single[0])
			temp[10:20] = -100/M[10]*(dM_dhi_single[10:20]-M[10:20]/M[10]*dM_dhi_single[10])
			temp[20:] = -100/M[20]*(dM_dhi_single[20:]-M[20:]/M[20]*dM_dhi_single[20])
			rel_sig_dhi.append(temp)

		return rel_sig_dhi

	def test(self, para, times):
		t = np.array(times['chase_time'])
		h = para
		return t*h[0] + h[1]	



def mean_square_diff(para, data, times, model_function): #UNCHANGED
	'''
	This function is minimised in the fitting.
	Evaluates the sum of squared differences of data and the model_function given.
	The model function can be anything here that takes para and times as input.
	'''
	diff=np.mean((model_function(para, times)-data)**2)
	return diff

def fitting(model_function, para_initial_guess, data, times, basinhops, logstep=False): # UNCHANGED
	'''
	The actual fitting algorithm.
	Finds the global minimum of the function mean_square_diff.
	If logstep is true it samples logspace for the fitting parameters.
	'''
	if logstep:
		mytakestep = MyTakeStep()
		fit=opt.basinhopping(mean_square_diff, para_initial_guess, take_step=mytakestep, minimizer_kwargs={'method':'Nelder-Mead', 'args':(data, times, model_function)}, niter=basinhops, disp=True, interval=10)
	else:
		fit=opt.basinhopping(mean_square_diff, para_initial_guess, minimizer_kwargs={'method':'Nelder-Mead', 'args':(data, times, model_function)}, niter=basinhops, disp=True, interval=10)
	return fit

#### Bayesian fitting

class posterior:
	# Generalize this to take more known parameters (e.g. monomer conc)
	'''
	The posterior distribution is made up of the likelihood function and the prior. 
	If the option log_prob is set to true, probabilities will instead be calculated in logspace and also returned as log(probability)
	using logs may be necessary to avoid numerical problems but if log porbabilities are used this needs to be taken into account in the MC
	The likelihood function will likely (haha) remain the same, the prior will probably change as we get more and more data.
	This is again highlighted below by an UNCHANGED flag for the ones that should remain the same.
	'''

	def __init__(self, prior_name, prior_user_para, likelihood_name, model_function, data_x, data_y, log_prob = False, jacobian = None):
		self.prior_user_para = prior_user_para
		self.model_fct = model_function
		self.x = data_x # this is time points and any additional parameters that are not to be varied in Bayesian inference
		self.y = data_y
		self.log_prob = log_prob
		if jacobian:
			self.model_fct_jacobian = jacobian
			self.prior_jacobian = eval('self.jacobian_log_'+str(prior_name))
			self.likelihood_jacobian = eval('self.jacobian_log_'+str(likelihood_name))
		else:
			print('Warning: no Jacobian provided; only algorithms that don\'t require Jacobians will work.')
		if self.log_prob:
			self.prior = eval('self.log_'+str(prior_name))
			self.likelihood = eval('self.log_'+str(likelihood_name))
		else:
			self.prior = eval('self.'+str(prior_name)) # this assigns the function to prior that is called by prior_name
			self.likelihood = eval('self.'+str(likelihood_name)) # this assigns the function to prior that is called by prior_name
		# we want the output of posterior to be a single function that takes on argument, the parameters, so this is the neatest way of doing it. Otherwis we would have to separately intialize the prior and the likelihood and then mess around combining them

	def likelihood_homoscedastic_gauss_noise(self, para): # UNCHANGED
		sig = para['sig_likelihood']
		return np.exp(-1/(2*sig**2)*np.sum((self.y-self.model_fct(para, self.x))**2))/(2*np.pi*sig**2)**(len(self.y)/2)

	def log_likelihood_homoscedastic_gauss_noise(self, para): # UNCHANGED
		sig = para['sig_likelihood']
		return -1/(2*sig**2)*np.sum((self.y-self.model_fct(para, self.x))**2) - (len(self.y)/2) * np.log(2*np.pi*sig**2)

	def likelihood_given_noise_vector(self, para):
		noise = self.prior_user_para['noise'] # the standard deviation
			# we use the input input when class is initialised, rather than the argument of the function, because this noise is part of the data and should not be varied in the fitting process. It needs to hav as many entries as the data, i.e. sel.x and sef.y
		return np.exp(-np.sum((self.y-self.model_fct(para, self.x))**2/noise**2))

	def log_likelihood_given_noise_vector(self, para):
		noise = self.prior_user_para['noise']
		return -np.sum((self.y-self.model_fct(para, self.x))**2/noise**2)

	def jacobian_log_likelihood_given_noise_vector(self, para):
		noise = self.prior_user_para['noise']
		jacobian = []
		model_derivatives = self.model_fct_jacobian(para, self.x)
		for dmodel_dxi in model_derivatives:
			2*np.sum((self.y-self.model_fct(para, self.x))/noise**2)*dmodel_dxi

	def prior_straight_line(self, para):
		return stats.norm.pdf(para['intercept'], self.prior_user_para['intercept_avg'], self.prior_user_para['intercept_scale']) *stats.norm.pdf(para['slope'], self.prior_user_para['slope_avg'], self.prior_user_para['slope_scale']) 

	def prior_gen_hatfct(self, para):
		'''
		Evaluates the prior probability given the parameters para (which are to be varied)
		and the (hyper-)parameters user_para (which are user defined) both are dictionaries of values.
		para has entries: 
			sig - the standard deviation of the noise
			tf1 - time of decrease (fall time) in channel 1
			tf2 - time of decrease (fall time) in channel 2
			tr1 - time of increase (rise time) in channel 1
			tr2 - time of increase (rise time) in channel 2
			kf1 - slope of decrease in channel 1
			kf2 - slope of decrease in channel 2
			kr1 - slope of increase in channel 1
			kr2 - slope of increase in channel 2

		user_para has entries: ***update****
			tf2_shift - average shift of tf2 relative to tf1
			tr_shift - average shift of rise time relative to fall time (same for both channels)
			sig_a, sig_loc, sig_scale - parameters determining the inverse gamma distribution of sig
			tf_loc, tf_scale - parameters determining the normal distribution of tf1
			tf_rel_scale - the standard deviation of the second channel fall time about the first channel fall time
			flowrate - the flowrate
			tr_shift - the shift of the rise time relative to the fall time, given for a flowrate of 1
			tr_rel_scale - the standard deviation of the rise time about the shifted value from the fall time
			kr_min, kf_min
			kr_max, kf_max
			kr_scale, kf_scale


		In this case we have the following:
		The probabilities of the amplitudes are flat (i.e. don't feature in the prior).
		The pdf for the variance of the noise, sig, is independent of the other parameters.
		The rise times and slopes in general depend on each other, see below for details.
		***!!!!!maybe split this up into fcts that calculate the individual probabilities; if you are doing MC, only varying one parameter that's a huge waste of computation calculating everything each step!!!!!
		'''
		sig_prob = 	stats.invgamma.pdf(para['sig_likelihood'],self.prior_user_para['sig_a'], self.prior_user_para['sig_loc'], self.prior_user_para['sig_scale']) # the distribution of the noise, an inverse gamma fct
		# stats.norm.pdf(para['sig_likelihood'], self.prior_user_para['sig_loc'], self.prior_user_para['sig_scale'])
		
		tf1_prob = stats.norm.pdf(para['tf1'], self.prior_user_para['tf_loc'], self.prior_user_para['tf_scale']) # the fall time, a normal distribtuion about a predetermined value 
		
		tf2_prob = stats.norm.pdf(para['tf2'], para['tf1'], self.prior_user_para['tf_rel_scale']) # the fall time in the second channel is simply a normal distr about the value of the fall time in the first channel
		
		tr1_prob = stats.norm.pdf(para['tr1'], para['tf1']-self.prior_user_para['tr_shift']/self.prior_user_para['flowrate'], self.prior_user_para['tr_rel_scale']) # dsitribution of the rise time, given the fall time, is determined by the flowrate and a shift value from the fall time ***Gamma pdf moght be more suitable here!!!!!!!
		
		tr2_prob = stats.norm.pdf(para['tr2'], para['tf2']-self.prior_user_para['tr_shift']/self.prior_user_para['flowrate'], self.prior_user_para['tr_rel_scale']) # as above

		kr1_prob = stats.norm.pdf(np.log(para['kr1']), self.prior_user_para['log_kr_loc'], self.prior_user_para['log_kr_scale'])
		#1/2/(self.prior_user_para['kr_max']-self.prior_user_para['kr_min'])*(special.erf((para['kr1'] - self.prior_user_para['kr_min'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])) - special.erf((para['kr1'] - self.prior_user_para['kr_max'])/(np.sqrt(2)*self.prior_user_para['kr_scale']))) # this comes from having a normal disttribtuion with mean mu and stdev kr_scale for kr, where the mean mu has a uniform distribution on kr_min to mr_max, then integrate out mu 
		#*** this is a problem numerically as it's often close to 0

		kr2_prob = stats.norm.pdf(np.log(para['kr2']), self.prior_user_para['log_kr_loc'], self.prior_user_para['log_kr_scale'])
		#1/2/(self.prior_user_para['kr_max']-self.prior_user_para['kr_min'])*(special.erf((para['kr2'] - self.prior_user_para['kr_min'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])) - special.erf((para['kr2'] - self.prior_user_para['kr_max'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])))

		kf1_prob = stats.norm.pdf(np.log(para['kf1']), self.prior_user_para['log_kf_loc'], self.prior_user_para['log_kf_scale'])
		#1/2/(self.prior_user_para['kf_max']-self.prior_user_para['kf_min'])*(special.erf((para['kf1'] - self.prior_user_para['kf_min'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])) - special.erf((para['kf1'] - self.prior_user_para['kf_max'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])))

		kf2_prob = stats.norm.pdf(np.log(para['kf1']), self.prior_user_para['log_kf_loc'], self.prior_user_para['log_kf_scale'])
		#1/2/(self.prior_user_para['kf_max']-self.prior_user_para['kf_min'])*(special.erf((para['kf2'] - self.prior_user_para['kf_min'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])) - special.erf((para['kf2'] - self.prior_user_para['kf_max'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])))

		#Question is where do we pick the parameters of the distributions?-they are dimensions in the MC- Do we marginalize over them? -only at the end when MC is done- are they also varied fro the MC?-yes- => What if we marginalised over them during the MC, i.e. rather than use p(a,b) = p(a|b)p(b), varying a and b in MC just use p(a) = integral(p(a|b)p(b) db); makes each individual MC step longer, but MC has fewer dimensions => see e.g. collapsed Gibbs sampler

		prior_prob = sig_prob*tf1_prob*tr1_prob*tf2_prob*tr2_prob*kr1_prob*kr2_prob*kf1_prob*kf2_prob
		#print('prior_prob',prior_prob)
		return prior_prob

	def log_prior_gen_hatfct(self, para):
		'''
		see prior_gen_hatfct
		'''
		sig_prob = 	stats.invgamma.pdf(para['sig_likelihood'],self.prior_user_para['sig_a'], self.prior_user_para['sig_loc'], self.prior_user_para['sig_scale']) # the distribution of the noise, an inverse gamma fct
		# stats.norm.pdf(para['sig_likelihood'], self.prior_user_para['sig_loc'], self.prior_user_para['sig_scale'])
		
		tf1_prob = stats.norm.pdf(para['tf1'], self.prior_user_para['tf_loc'], self.prior_user_para['tf_scale']) # the fall time, a normal distribtuion about a predetermined value 
		
		tf2_prob = stats.norm.pdf(para['tf2'], para['tf1'], self.prior_user_para['tf_rel_scale']) # the fall time in the second channel is simply a normal distr about the value of the fall time in the first channel
		
		tr1_prob = stats.norm.pdf(para['tr1'], para['tf1']-self.prior_user_para['tr_shift']/self.prior_user_para['flowrate'], self.prior_user_para['tr_rel_scale']) # dsitribution of the rise time, given the fall time, is determined by the flowrate and a shift value from the fall time ***Gamma pdf moght be more suitable here!!!!!!!
		
		tr2_prob = stats.norm.pdf(para['tr2'], para['tf2']-self.prior_user_para['tr_shift']/self.prior_user_para['flowrate'], self.prior_user_para['tr_rel_scale']) # as above

		kr1_prob = stats.norm.pdf(np.log(para['kr1']), self.prior_user_para['log_kr_loc'], self.prior_user_para['log_kr_scale'])
		#1/2/(self.prior_user_para['kr_max']-self.prior_user_para['kr_min'])*(special.erf((para['kr1'] - self.prior_user_para['kr_min'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])) - special.erf((para['kr1'] - self.prior_user_para['kr_max'])/(np.sqrt(2)*self.prior_user_para['kr_scale']))) # this comes from having a normal disttribtuion with mean mu and stdev kr_scale for kr, where the mean mu has a uniform distribution on kr_min to mr_max, then integrate out mu 
		#*** this is a problem numerically as it's often close to 0

		kr2_prob = stats.norm.pdf(np.log(para['kr2']), self.prior_user_para['log_kr_loc'], self.prior_user_para['log_kr_scale'])
		#1/2/(self.prior_user_para['kr_max']-self.prior_user_para['kr_min'])*(special.erf((para['kr2'] - self.prior_user_para['kr_min'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])) - special.erf((para['kr2'] - self.prior_user_para['kr_max'])/(np.sqrt(2)*self.prior_user_para['kr_scale'])))

		kf1_prob = stats.norm.pdf(np.log(para['kf1']), self.prior_user_para['log_kf_loc'], self.prior_user_para['log_kf_scale'])
		#1/2/(self.prior_user_para['kf_max']-self.prior_user_para['kf_min'])*(special.erf((para['kf1'] - self.prior_user_para['kf_min'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])) - special.erf((para['kf1'] - self.prior_user_para['kf_max'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])))

		kf2_prob = stats.norm.pdf(np.log(para['kf1']), self.prior_user_para['log_kf_loc'], self.prior_user_para['log_kf_scale'])
		#1/2/(self.prior_user_para['kf_max']-self.prior_user_para['kf_min'])*(special.erf((para['kf2'] - self.prior_user_para['kf_min'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])) - special.erf((para['kf2'] - self.prior_user_para['kf_max'])/(np.sqrt(2)*self.prior_user_para['kf_scale'])))
		
		#Question is where do we pick the parameters of the distributions?-they are dimensions in the MC- Do we marginalize over them? -only at the end when MC is done- are they also varied fro the MC?-yes- => What if we marginalised over them during the MC, i.e. rather than use p(a,b) = p(a|b)p(b), varying a and b in MC just use p(a) = integral(p(a|b)p(b) db); makes each individual MC step longer, but MC has fewer dimensions => see e.g. collapsed Gibbs sampler

		#print('kr1_prob',kr1_prob)
		#print('kf1_prob',kf1_prob)
		#print('tr1_prob',tr1_prob)
		#print('tf1_prob',tf1_prob)
		#print('tr2_prob',tr2_prob)
		#print('tf2_prob',tf2_prob)
		#for i,j in [('tr1_prob',tr1_prob),('tf1_prob',tf1_prob),('tr2_prob',tr2_prob),('tf2_prob',tf2_prob),('sig',sig_prob)]:
		#	if j==0:
		#		print('WARNING: zero probability encountered in {}'.format(i))

		log_prior_prob = np.log(sig_prob) + np.log(tf1_prob) + np.log(tr1_prob) + np.log(tf2_prob) + np.log(tr2_prob) + np.log(kr1_prob) + np.log(kr2_prob) + np.log(kf1_prob) + np.log(kf2_prob)

		return log_prior_prob

	def prior_flat(self, para): # UNCHANGED
		return 1

	def log_prior_flat(self, para): # UNCHANGED
		return 0

	def prior_max_entropy(self, para):
		h = para/np.sum(para)
		N = len(h)
		target_distr = self.prior_user_para['target_distr']
		if not len(target_distr)==N:
			print('Error: target distribution for prior and initial guess need to have the same number of bins. Terminating.')
			exit()
		lam = self.prior_user_para['Lagrange_mult']
		entropy = np.sum(h - target_distr - h*np.log(h/target_distr))
		return Exp[lam*entropy]

	def log_prior_max_entropy(self, para):
		h = para/np.sum(para)
		N = len(h)
		target_distr = self.prior_user_para['target_distr']
		if not len(target_distr)==N:
			print('Error: target distribution for prior and initial guess need to have the same number of bins. Terminating.')
			exit()
		lam = self.prior_user_para['Lagrange_mult']
		return lam*np.sum(h - target_distr - h*np.log(h/target_distr))

	def jacobian_log_prior_max_entropy(self, para):
		h = para/np.sum(para)
		N = len(h)
		target_distr = self.prior_user_para['target_distr']
		lam = self.prior_user_para['Lagrange_mult']
		return lam*np.log(h/target_distr)

	def total_prob(self, para): # UNCHANGED
		'''
		the total probability, prior and lieklihood combined
		'''
		if self.log_prob:
			return self.prior(para)+self.likelihood(para)
		else:
			return self.prior(para)*self.likelihood(para)

	def neg_total_prob_num_input(self, para_values, para_names):
		'''
		NEGATIVE of total_prob; this takes a list of numbers and a list of names as input and then recombines them into a dictionary
		this is for use in minimisation algorithms that usually require input in the form of purely numerical lists and we will want to maximise the prob
		'''
		if para_names:	# names are given for the parameters, im[plying that the model function needs named parameters as a dict
			para = dict()
			for i,j in zip(para_names, para_values):
				para[i] = abs(j)
		else:			# no parameter names are given implying that the model function needs parameters as a list/vector
			para = abs(para_values)

		if self.log_prob:
			return -self.prior(para)-self.likelihood(para)
		else:
			return -self.prior(para)*self.likelihood(para)

	def total_prob_jacobian(self,para_values,para_names):
		if para_names:	# names are given for the parameters, im[plying that the model function needs named parameters as a dict
			para = dict()
			for i,j in zip(para_names, para_values):
				para[i] = abs(j)
		else:			# no parameter names are given implying that the model function needs parameters as a list/vector
			para = abs(para_values)

		if self.log_prob:
			return -self.prior_jacobian(para)-self.likelihood_jacobian(para)
		else:
			print('ERROR: General Jacobian not yet implemented for non-log likelihoods and priors. Terminating...')
			exit()


class MC:
	# Generalize to have parameters that are not being varied in MC
	'''
	This is the class responsible for running the Monte Carlo algorithm. 
	It takes a probability distribution function as an input, which for Bayesian inference will be the posterior.
	Both the step_proposer (chooses a new set of parameters) and the step taker (decides whether the new parameters should be accepted or not) might change in the future, if we can gain some porcessing speed from changing them
	'''
	def __init__(self, step_proposer_name, step_taker_name, prob_fct,log_prob = False):
		self.prob_fct = prob_fct
		self.log_prob = log_prob
		self.step_proposer = eval('self.'+str(step_proposer_name))
		if self.log_prob:
			self.step_taker = eval('self.log_prob_'+str(step_taker_name))
		else:
			self.step_taker = eval('self.'+str(step_taker_name))

	def propose_new_MC_parameters_random_walk(self, para, step_size):
		# ads a random value, drawn from normal distr with stdev epsilon, to the parameters
		new = {}
		for key, val in para.items():
			new[key] = float(val*(1 + np.random.normal(0,step_size,1)))
		return new 	

	def propose_new_MC_parameters_random_walk_log_space(self, para, step_size):
		# ads a random value, drawn from normal distr with stdev epsilon, to the parameters
		new = {}
		for key, val in para.items():
			new[key] = float(np.exp(np.log(val)+ np.random.normal(0,step_size,1)))
		return new 	

	def take_MC_step_Metropolis(self, para_to_vary, step_size):
		ran = np.random.rand(1)

		new_para = self.step_proposer(para_to_vary, step_size)
		new_prob = self.prob_fct(new_para)
		prob_ratio = new_prob/self.current_prob

		if ran < min(1,prob_ratio):
			self.current_prob = new_prob # we save the porbability associated with the current parameters to the class, so it can be used in the next iteration of the MC without having to evaluate the function again
			return 1, new_para # first output is 1 if accepted 0 if not accepted
		else:
			return 0, para_to_vary

	def log_prob_take_MC_step_Metropolis(self, para_to_vary, step_size):
		ran = np.random.rand(1)

		new_para = self.step_proposer(para_to_vary, step_size)
		new_prob = self.prob_fct(new_para)
		prob_ratio = np.exp(new_prob-self.current_prob)
		if ran < min(1,prob_ratio):
			self.current_prob = new_prob # we save the porbability associated with the current parameters to the class, so it can be used in the next iteration of the MC without having to evaluate the function again
			return 1, new_para # first output is 1 if accepted 0 if not accepted
		else:
			return 0, para_to_vary


	def MC_iterator(self,initial_parameters, N_iter=1e5, step_size=0.03, sample_every=1, update_step_proposer_every = 100): # UNCHANGED
		'''
		if update_step_proposer_every is not None, a PI controller is used to update step_size, which is a parameter controlling the step_proposer, such that the acceptance approaches an "optimal" value; if a different step taker is used, this may no longer be necessary (e.g. in a Gibbs sampler)
		sample_every determines how often the parameters are saved to the trajectory
		'''

		#!!!! decompose into runs and rearrange parameters accordingly; total_prob is for a single run
		# what we mean by this is our model is only in terms of one dependent variable, e.g. time,
		# if other dependent variables, e.g. monomer concentration, are changed that's a different run
		#!!! Why not generalize this to have N independent variables? can we do this already? 
		# each argument in the function is either a float or an array, if it's an array it ouputs an array, i.e. computes it for every element
		# all we need is a smart way of creating these input parameters.

		para_trajectory = {}
		for key, val in initial_parameters.items():
			para_trajectory[key] = [val]

		if update_step_proposer_every:
			optimal_ratio = 0.234
			# setting the PID values based on Ziegler-Nichols method
			#kosc = 0.1# value of porportional gain at which error starts to oscillate, with intergral and derivative gains set to 0
			#Tosc = 8# period of oscillations above
			#controller = PI_controller(optimal_ratio, 0.6*kosc, 1.2*kosc/Tosc, 0.075*kosc*Tosc )
			# setting the PID values based on trial and error
			controller = PI_controller(optimal_ratio, step_size, step_size/500, D=step_size/50)

		accepted = 0
		para_new = initial_parameters
		self.current_prob = self.prob_fct(para_new)
		for i in range(int(N_iter)):
			para_old = para_new
			accepted_temp, para_new = self.step_taker(para_old, step_size)
			accepted += accepted_temp
			if i % sample_every==0:
				for j in para_trajectory.keys():
					para_trajectory[j].append(para_new[j])
				

			if update_step_proposer_every:
				if i % update_step_proposer_every == 0:
					acceptance_ratio = accepted/update_step_proposer_every
					print('Iteration {} of {}. Current step_size: {} Current acceptance: {}%'.format(i, N_iter, step_size, 100* acceptance_ratio))
					step_size = abs(step_size + controller.update(acceptance_ratio)) # adjust step_size so we have the correct ratio
					accepted = 0 # reset the accepted number
		return para_trajectory

def MAP_fitting(neg_prob_fct_num_input, para_initial_guess, basinhops, logstep=False, additional_args=None): # UNCHANGED
	'''
	Similar to a least squares fit, only now this finds the global minimum of (neg_prob_fct_num_input)
	this should be the negative of the probability function (or its log), that takes a numerical list as the first input
	additional inputs to neg_prob_fct_num_input should be provided as a tuple in additional_args
	If logstep is true it samples logspace for the fitting parameters.
	'''
	if logstep:
		mytakestep = MyTakeStep()
		fit=opt.basinhopping(neg_prob_fct_num_input, para_initial_guess, take_step=mytakestep, minimizer_kwargs={'method':'Nelder-Mead', 'args':additional_args}, niter=basinhops, disp=True, interval=10)
	else:
		fit=opt.basinhopping(neg_prob_fct_num_input, para_initial_guess, minimizer_kwargs={'method':'Nelder-Mead', 'args':additional_args}, niter=basinhops, disp=True, interval=10)
	return fit

def MAP_fitting_constrained(neg_prob_fct_num_input, jacobian_neg_prob, para_initial_guess, basinhops, constraints, logstep=False, additional_args=None):
	'''
	Similar to a least squares fit, only now this finds the global minimum of (neg_prob_fct_num_input)
	this should be the negative of the probability function (or its log), that takes a numerical list as the first input
	additional inputs to neg_prob_fct_num_input should be provided as a tuple in additional_args
	If logstep is true it samples logspace for the fitting parameters.
	'''
	if logstep:
		mytakestep = MyTakeStep()
		fit=opt.basinhopping(neg_prob_fct_num_input, para_initial_guess, take_step=mytakestep, minimizer_kwargs={'method':'SLSQP', 'args':additional_args, 'constraints': constraints}, niter=basinhops, disp=True, interval=10)
	else:
		fit=opt.basinhopping(neg_prob_fct_num_input, para_initial_guess, minimizer_kwargs={'method':'SLSQP', 'args':additional_args, 'constraints': constraints}, niter=basinhops, disp=True, interval=10)
	return fit



class PI_controller:
	def __init__(self, target, P, I, D=0):
		self.target = target
		self.P = P
		self.I = I
		self.D = D
		self.acc = 0
		self.prev_err = 0.5
	def update(self, val):
		err = val - self.target
		self.acc += err
		deriv = err-self.prev_err
		self.prev_err = err
		return self.P * err + self.I * self.acc + self.D*deriv


### Plotting

def plot_data_and_fits_single_dataframe(data_sets, model=None, plot_title='', save_to=False, x_name = 'time', plot_style='o', times_to_eval_model=pd.DataFrame([])):
	'''
	Plot data, which should be porvided as pandas array using one common x value column defined by x_name.
	The names of the dictionary entries combined with the column name will be the names of the data_sets.
	The columns to plot can be changed below in x_name and y_name.
	If a model is given, this will be plotted as well; it should be provided in this format
	{<data_set_name>: {'function':<modelfunction>,'parameters':{'a':<parameter_a>,'b':<parameter_b>,...}}, ...}
	where data_set_name should be the same as the column name in the dataframe.
	'''

	x_label = 'time'
	y_label = 'signal'

	fig=plt.figure()
	ax=fig.add_subplot(111)
	data_set_names = data_sets.columns
	N_datasets = len(data_set_names)-1 # not counting the x column
	for ind, val in enumerate(data_set_names):
		if not val==x_name:
			if len(str(val))>18:
				label_temp='...'+str(val)[-15:]
			else:
				label_temp=val
			if label_temp[0]=='_': # for some reason underscore as first character causes bug
				label_temp_list = list(label_temp)
				label_temp_list[0] = ' '
				label_temp = "".join(label_temp_list)

			ax.plot(data_sets[x_name], data_sets[val], plot_style,markersize=7,mec=pastel(plt.get_cmap('spectral')(float(ind)/(N_datasets+1)),weight=2),color=pastel(plt.get_cmap('spectral')(float(ind)/(N_datasets+1)),weight=2.3), label=label_temp)
			ax.set_xlabel(x_label,size='large')
			ax.set_ylabel(y_label,size='large')
			ax.set_title(plot_title,size='large')
			if model:				# if htere is a model function to fit we have to get the minimum and maximum values for x
				xmin = min(data_sets[val][x_name])
				xmax = max(data_sets[val][x_name])
				if times_to_eval_model.empty:
					x = pd.DataFrame(np.linspace(xmin, xmax, 100), columns = ['x'])
				else:
					print('Using timepoints provided by user in plotting.')
					x = pd.DataFrame(times_to_eval_model)
					x.columns = ['x']
				params = model[val]['parameters']
				func = model[val]['function']
				y = func(params, x['x'])
				#print('\n\n ****PLot\n', times_to_eval_model, x,y)
				ax.plot(x, y, '-',color=plt.get_cmap('spectral')(float(ind)/(N_datasets+1)),linewidth=3, label=label_temp+'_fit')

	ax.legend(loc='best', shadow=True,numpoints=1,prop={'size':8}, labelspacing=0.1)

	if save_to:
		print('Saving plot to',save_to)
		fig.savefig(save_to)
	plt.clf()

def plot_data_and_fits(data_sets, model=None, plot_title='', save_to=False, x_name = 'time(s)', y_name = 'normalised_PhotonCounter', plot_style='o', times_to_eval_model=pd.DataFrame([])):
	'''
	Plots a number of data_sets, which should be porvided as pandas arrays in a dictionary.
	The names of the dictionary entries will be the names of the data_sets.
	The columns to plot can be changed below in x_name and y_name.
	If a model is given, this will be plotted as well; it should be provided in this format
	{<data_set_name>: {'function':<modelfunction>,'parameters':{'a':<parameter_a>,'b':<parameter_b>,...}}, ...}
	where data_set_name should be the same as in the dictionary of datasets.
	'''

	x_label = 'time(s)'
	y_label = 'signal'

	fig=plt.figure()
	ax=fig.add_subplot(111)
	for ind, val in enumerate(sorted(data_sets.keys())):
		if len(str(val))>18:
			label_temp='...'+str(val)[-15:]
		else:
			label_temp=val
		if label_temp[0]=='_': # for some reason underscore as first character causes bug
			label_temp_list = list(label_temp)
			label_temp_list[0] = ' '
			label_temp = "".join(label_temp_list)

		ax.plot(data_sets[val][x_name], data_sets[val][y_name], plot_style,markersize=7,mec=pastel(plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),weight=2),color=pastel(plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),weight=2.3), label=label_temp)
		ax.set_xlabel(x_label,size='large')
		ax.set_ylabel(y_label,size='large')
		ax.set_title(plot_title,size='large')
		if model:				# if htere is a model funciton to fit we have to get the minimum and maximum values for x
			xmin = min(data_sets[val][x_name])
			xmax = max(data_sets[val][x_name])
			if times_to_eval_model.empty:
				x = pd.DataFrame(np.linspace(xmin, xmax, 100), columns = ['x'])
			else:
				print('Using timepoints provided by user in plotting.')
				x = pd.DataFrame(times_to_eval_model)
				x.columns = ['x']
			params = model[val]['parameters']
			func = model[val]['function']
			y = func(params, x['x'])
			#print('\n\n ****PLot\n', times_to_eval_model, x,y)
			ax.plot(x, y, '-',color=plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),linewidth=3, label=label_temp+'_fit')

	ax.legend(loc='best', shadow=True,numpoints=1,prop={'size':8}, labelspacing=0.1)

	if save_to:
		print('Saving plot to',save_to)
		fig.savefig(save_to)
	plt.clf()

def plot_data_and_fits_mulit_t_input(data_sets, model=None, plot_title='', save_to=False, x_name = 'time(s)', y_name = 'normalised_PhotonCounter', plot_style='o', times_to_eval_model=None):
	'''
	Plots a number of data_sets, which should be porvided as pandas arrays in a dictionary.
	The names of the dictionary entries will be the names of the data_sets.
	The columns to plot can be changed below in x_name and y_name.
	If a model is given, this will be plotted as well; it should be provided in this format
	{<data_set_name>: {'function':<modelfunction>,'parameters':{'a':<parameter_a>,'b':<parameter_b>,...}}, ...}
	where data_set_name should be the same as in the dictionary of datasets.
	times_to_eval_model can be an array rather than just a vector if the function needs more input
	'''

	x_label = 'time(s)'
	y_label = 'signal'

	fig=plt.figure()
	ax=fig.add_subplot(111)
	for ind, val in enumerate(sorted(data_sets.keys())):
		if len(str(val))>18:
			label_temp='...'+str(val)[-15:]
		else:
			label_temp=val
		if label_temp[0]=='_': # for some reason underscore as first character causes bug
			label_temp_list = list(label_temp)
			label_temp_list[0] = ' '
			label_temp = "".join(label_temp_list)

		ax.plot(data_sets[val][x_name], data_sets[val][y_name], plot_style,markersize=7,mec=pastel(plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),weight=2),color=pastel(plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),weight=2.3), label=label_temp)
		ax.set_xlabel(x_label,size='large')
		ax.set_ylabel(y_label,size='large')
		ax.set_title(plot_title,size='large')
		if model:				# if htere is a model funciton to fit we have to get the minimum and maximum values for x
			xmin = min(data_sets[val][x_name])
			xmax = max(data_sets[val][x_name])
			if not times_to_eval_model:
				x = pd.DataFrame(np.linspace(xmin, xmax, 100), columns = ['x'])
			else:
				print('Using timepoints provided by user in plotting.')
				x = times_to_eval_model[val][x_name].copy()
			params = model[val]['parameters']
			func = model[val]['function']
			y = func(params, times_to_eval_model[val])
			#print('\n\n ****PLot\n', times_to_eval_model, x,y)
			ax.plot(x, y, '-',color=plt.get_cmap('spectral')(float(ind)/(len(data_sets)+1)),linewidth=3, label=label_temp+'_fit')

	ax.legend(loc='best', shadow=True,numpoints=1,prop={'size':8}, labelspacing=0.1)

	if save_to:
		print('Saving plot to',save_to)
		fig.savefig(save_to)
	plt.clf()
