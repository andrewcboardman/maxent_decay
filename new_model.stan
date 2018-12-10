data {
	int N;
	int n_bins;
	vector[N] pulse_time;
	vector[N] chase_time;
	vector[N] measurement;
	vector[N] noise;
	vector[n_bins+1] sigma;
	vector[n_bins] target_distr;
	real theta; 
	real lamda;
}

transformed_data {
	matrix M[N,n_bins];
	for (j in 1:n_bins) {
		for (i in 1:N) {
			M[i,j] = (gamma_q(0.001,(chase_time[i])*(theta+sigma[j+1]))-gamma_q(0.001,(chase_time[i])*(theta+sigma[j]))-gamma_q(0.001,(chase_time[i]+tau)*(theta+sigma[j+1]))+gamma_q(0.001,(chase_time[i]+tau)*(theta+sigma[j])))*exp(theta*(chase_time[i]+tau))*abs(h[j])/log(10);
		}
	}
}

parameters {
	vector[n_bins] h;
}

model {
	matrix M_adj[N,n_bins];
	vector h_norm[n_bins];
	vector rel_sig[N];
	M_adj = M.*abs(h)/log(10);
	//// This next bit takes care of the fact that the data is composed of 3 datasets with different pulse times and they need to be normalised separately 
	rel_sig[:10] = 100*(1-M_adj[:10]/M_adj[1]);
	rel_sig[11:20] = 100*(1-M_adj[11:20]/M_adj[11]);
	rel_sig[21:] = 100*(1-M_adj[21:]/M_adj[21]);
	measurement ~ normal(rel_sig,noise);

	h_norm = h/sum(h);
	entropy = sum(h - target_distr - h*log(h/target_distr));
	target += lamda*entropy;
}

