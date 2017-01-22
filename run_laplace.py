import numpy as np
import scipy.linalg as la
import sys
import time
import pandas as pd
import laplace_cython
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.special
np.set_printoptions(precision=4, linewidth=200)

def symmetrize(X): return X + X.T - np.diag(X.diagonal())

def approx_hessian(fun_g, w0, args, rr=1e-3):

	H = np.eye(w0.shape[0])
	for i in xrange(len(w0)):
		v = H[:,i]
		try:
			_, g2 = fun_g(w0-rr*v, *args)	
			_, g1 = fun_g(w0+rr*v, *args)
		except:
			g1 = optimize.approx_fprime(w0-rr*v, fun_g, 1e-8, *args)
			g2 = optimize.approx_fprime(w0+rr*v, fun_g, 1e-8, *args)			
			
			
		H[:,i] = (g1 - g2) / (2*rr)
	return H


class EWAS_Laplace():
	def __init__(self):
		self.optIter = 0
		self.optimization = False
		self.verbose = False
		pass
		
		
	def ll_binom_laplace(self, hyp, *varargin):
	
		if (np.max(np.abs(hyp[:2])) > 5): return np.inf, np.ones(len(hyp)) + np.inf
		
		try:
			nlZ = laplace_cython.laplace_nlZ(hyp, *varargin)
		except ValueError, e:
			print '#optimization failed with ValueError!'
			print '#message:', e
			print '#hyp:', hyp			
			nlZ = np.inf, np.ones(len(hyp)) + np.inf
			
			
		try: returnGrad = (len(nlZ) > 1)
		except: returnGrad=False
		if returnGrad:
			nlZ, grad = nlZ
		
		if (self.verbose and self.optimization):
			if (self.optIter % 10 == 0):
				t1 = time.time()
				print 'll_binom_laplace optimization iteration %d nlZ: %0.4f  time: %0.2f seconds'%(self.optIter, nlZ, t1-self.iter_t0)
				self.iter_t0 = t1
			self.optIter += 1
			
		if returnGrad: return nlZ, grad
		else: return nlZ


	def optim_ll_binom_laplace(self, hyp0, opt='gpstuff', *opt_args):
		self.optIter = 0
		self.optimization = True
		self.iter_t0 = time.time()
		
		hyp_opt = optimize.minimize(self.ll_binom_laplace, hyp0, args=opt_args, jac=True, method=opt)#, options={'gtol':1e-5, 'ftol':1e-5})#, bounds=[(-5,2) for p in hyp0])
		
		self.optimization = False
		if self.verbose:
			print 'final nlZ: %0.4f'%(hyp_opt.fun)
			print 'final params:', hyp_opt.x
			print
			
		try:				
			return hyp_opt.fun, hyp_opt.x, hyp_opt.hess_inv
		except:
			return hyp_opt.fun, hyp_opt.x, None
		
		
		
		
	
	def testBeta(self, params0_null, params0_alt, kernels, covars_null, covars_alt, y, r, covars_null_fixed, covars_alt_fixed, verbose=True, opt='gpstuff', test='wald', returnF=False, returnGrad=True, inv_tol=-1.0, num_simu=0, ZC_direct=True):
	
		old_verbose = self.verbose
		self.verbose = verbose
		t0_beta = time.time()
		opt_covars = True
		
		params0_alt_orig = params0_alt.copy()
		params0_null_orig = params0_null.copy()
			
		opt_alt = False
		try: 
			beta0_alt = solve_binom(y, r, covars_alt_fixed, return_ll=False)			
			params0_alt[-covars_alt.shape[1]:] = beta0_alt[:covars_alt.shape[1]]
			opt_alt = True
		except: pass
		
		opt_null = False
		if ((test != 'wald') or not opt_alt or (not opt_covars)):
			try:			
				beta0_null = solve_binom(y, r, covars_null_fixed, return_ll=False)
				params0_null[-covars_null.shape[1]:] = beta0_null[:covars_null.shape[1]]
				opt_null = True
			except: opt_covars=True
			
			
		if (opt_null and (not opt_covars)):
			m0 = covars_null.dot(beta0_null)
			opt_args_null = (kernels, np.empty((y.shape[0],0)), m0, y, r, True, returnGrad, False, False, np.zeros(y.shape[0]), inv_tol, num_simu, False, ZC_direct)
			opt_args_alt  = (kernels, covars_alt[:,-1:],        m0, y, r, True, returnGrad, False, False, np.zeros(y.shape[0]), inv_tol, num_simu, False, ZC_direct)
			params0_null = params0_null[:2]
			params0_alt = params0_alt[[0,1,-1]]
		else:
			m0 = np.zeros(y.shape[0])
			opt_args_null = (kernels, covars_null, m0, y, r, True, returnGrad, False, False, np.zeros(y.shape[0]), inv_tol, num_simu, False, ZC_direct)
			opt_args_alt  = (kernels, covars_alt,  m0, y, r, True, returnGrad, False, False, np.zeros(y.shape[0]), inv_tol, num_simu, False, ZC_direct)
			
		nlZ_null = np.inf
		if ((test != 'wald') or not opt_alt):
			try: nlZ_null, params_opt_null, hess_inv_null = self.optim_ll_binom_laplace(params0_null, opt, *opt_args_null)
			except: return np.nan, np.nan
		
		if (test != 'score'):
			params0_alt2 = params0_alt.copy()
			if not opt_alt:
				if returnF: params0_alt2 = params_opt_null.copy()
				else: params0_alt2[:-1] = params_opt_null
			try:
				nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)				
				if np.isclose(nlZ_alt, 0):
					params0_alt2 = params0_alt_orig.copy()
					nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
					
			except:
				opt_args_null=list(opt_args_null); opt_args_null[12] = True; opt_args_null = tuple(opt_args_null)
				opt_args_alt=list(opt_args_alt); opt_args_alt[12] = True; opt_args_alt = tuple(opt_args_alt)
				try:
					nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
					if np.isclose(nlZ_alt, 0):
						params0_alt2 = params0_alt_orig.copy()
						nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
				except:
					return np.nan, np.nan				
			
			#find numerical problems
			if (test=='wald' and np.any(np.abs(params_opt_alt) > 400)):
				try: nlZ_null, params_opt_null, hess_inv_null = self.optim_ll_binom_laplace(params0_null, opt, *opt_args_null)
				except:					
					opt_args_null=list(opt_args_null); opt_args_null[12] = True; opt_args_null = tuple(opt_args_null)
					opt_args_alt=list(opt_args_alt); opt_args_alt[12] = True; opt_args_alt = tuple(opt_args_alt)
					try: nlZ_null, params_opt_null, hess_inv_null = self.optim_ll_binom_laplace(params0_null, opt, *opt_args_null)
					except: return np.nan, np.nan					
				if returnF: params0_alt2 = params_opt_null.copy() 
				else: params0_alt2[:-1] = params_opt_null
				try: nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
				except:
					opt_args_null=list(opt_args_null); opt_args_null[12] = True; opt_args_null = tuple(opt_args_null)
					opt_args_alt=list(opt_args_alt); opt_args_alt[12] = True; opt_args_alt = tuple(opt_args_alt)
					try: nlZ_alt, params_opt_alt, hess_inv_alt = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
					except: return np.nan, np.nan					
				if (np.any(np.abs(params_opt_alt) > 400)): return np.nan, np.nan
				
		#hack for L-BFGS-B
		if (test != 'score' and not returnF):
			try:			
				try:
					nlZ_null2, _ = self.ll_binom_laplace(params_opt_alt[:-1], *opt_args_null)
					if (nlZ_null2 < nlZ_null):
						nlZ_null=nlZ_null2
						params_opt_null = params_opt_alt[:-1]
				except: pass
				
				if (nlZ_alt > nlZ_null):
					params0_alt2[:-1] = params_opt_null
					params0_alt2[-1]=0
					try: nlZ_alt2, params_opt_alt2, hess_inv_alt2 = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)
					except:
						opt_args_null=list(opt_args_null); opt_args_null[12] = True; opt_args_null = tuple(opt_args_null)
						opt_args_alt=list(opt_args_alt); opt_args_alt[12] = True; opt_args_alt = tuple(opt_args_alt)
						try: nlZ_alt2, params_opt_alt2, hess_inv_alt2 = self.optim_ll_binom_laplace(params0_alt2, opt, *opt_args_alt)					
						except: return np.nan, np.nan
					if (nlZ_alt2 < nlZ_alt and nlZ_alt2>0):
						nlZ_alt=nlZ_alt2
						params_opt_alt = params_opt_alt2
						hess_inv_alt = hess_inv_alt2
					
					if (test != 'wald'):
						nlZ_null2, _ = self.ll_binom_laplace(params_opt_alt2[:-1], *opt_args_null)				
						if (nlZ_null2 < nlZ_null):
							nlZ_null=nlZ_null2
							params_opt_null = params_opt_alt2[:-1]
			except: raise
			
		if (test == 'score'):
			params0_final = params0_alt.copy()
			params0_final[:-1] = params_opt_null
			params0_final[-1]=0
			nlZ_null_final, grad_opt_alt = self.ll_binom_laplace(params0_final, *opt_args_alt)
		
		self.verbose = old_verbose
		
		if (test == 'wald'):
		
			if returnF:
				opt_args_alt = list(opt_args_alt)
				opt_args_alt[-2] = True
				opt_args_alt = tuple(opt_args_alt)
				f = laplace_cython.laplace_nlZ(params_opt_alt, *opt_args_alt)
				return f, None		
		
			try:
				#H = approx_hessian(self.ll_binom_laplace, params_opt_alt, opt_args_alt)
				if (False and hess_inv_alt is not None):
					invH_last = hess_inv_alt.todense()[-1,-1]
				else:
					H = approx_hessian(laplace_cython.laplace_nlZ, params_opt_alt, opt_args_alt)
					H_symm = (H+H.T)/2.0
					
					sH, UH = la.eigh(H_symm)
					i_pos = (sH > 1e-10)
					assert np.any(i_pos)					
					invH_last = np.sum(UH[-1,i_pos]**2 / sH[i_pos])
				
				wald_stat = params_opt_alt[-1]**2 / invH_last				
				return wald_stat, None
				
			except:
				return np.nan, None
				
				
		elif (test == 'score'):
				H = approx_hessian(self.ll_binom_laplace, params0_final, opt_args_alt)
				H_symm = (H+H.T)/2.0				
				sH, UH = la.eigh(H_symm)
				i_pos = (sH > 1e-10)
				invH_last = np.sum(UH[-1,i_pos]**2 / sH[i_pos])
				score_stat = grad_opt_alt[-1]**2 / invH_last
				return score_stat, None
			
			
		elif (test == 'lr'):
			return 2*(nlZ_null - nlZ_alt), None
			
		else:
			raise Exception('unknown test')
		
		
		
		
		
		
		
		
def readMacauTest(mcounts_file, counts_file, kernel_file, predictor_file, covars_file=None, correctK=True, kernel2=None):

	#read methelated counts file
	df_mcounts = pd.read_csv(mcounts_file, sep='\t', comment='#', header=0).dropna(axis=1, how='all')	
	yMat = df_mcounts[df_mcounts.columns[1:]].values.astype(np.int)
	snpNames_y = df_mcounts[df_mcounts.columns[0]].values
	
	#read total counts file
	df_rcounts = pd.read_csv(counts_file, sep='\t', comment='#', header=0).dropna(axis=1, how='all')	
	rMat = df_rcounts[df_rcounts.columns[1:]].values.astype(np.int)
	snpNames_r = df_rcounts[df_rcounts.columns[0]].values	
	assert np.all(snpNames_y == snpNames_r)
	
	#remove illegal positions
	bad = np.where(rMat < yMat)
	# if (len(bad[0]) > 0):
		# print '#removing the following illegal sites:', 
		# for s in snpNames_r[bad[0]]: print s,
		# print	
	is_good = np.ones(len(snpNames_r), dtype=np.bool)
	is_good[bad[0]] = False
	snpNames_r = snpNames_r[is_good]
	snpNames_y = snpNames_y[is_good]	
	yMat = yMat[is_good, :]
	rMat = rMat[is_good, :]
	
	#load kernels
	K = np.loadtxt(kernel_file)
	if (kernel2 is not None): K2 = np.loadtxt(kernel2)
	
	#make K a positive-definite matrix
	if correctK:
		print 'kernel is not positive definite...'
		d = np.diag(K).copy()
		factor = 1e-8
		s = la.eigh(K, eigvals_only=True, eigvals=(0,0), check_finite=False)		
		while (s[0] <= 0):
			np.fill_diagonal(K, d*(1+factor))
			factor*=2
			s = la.eigh(K, eigvals_only=True, eigvals=(0,0), check_finite=False)
		if (factor>1e-8): print 'inflated the diagonal of K by %0.5e'%(1+factor)
		K /= np.mean(np.diag(K))
		#np.savetxt('../example/relatedness_n50_fixed.txt', K, delimiter='\t',fmt='%0.4f'); sys.exit(0)
	
	#load predictor
	covToTest = np.loadtxt(predictor_file, dtype=np.float)
	
	#load covariates
	covars=None
	if (covars_file is not None):
		covars = np.loadtxt(covars_file)		
		covars_std = covars.std(axis=0, ddof=1)
		is_const = (covars_std == 0)
		if (np.all(is_const)): covars=None
		else:
			covars = covars[:, ~is_const]		
			covars -= covars.mean(axis=0)
			covars /= covars_std[~is_const]		
		
	
	if (kernel2 is None): return K, yMat, rMat, covToTest, covars, snpNames_y
	else: return (K, K2), yMat, rMat, covToTest, covars, snpNames_y
	
	
	
def solve_binom(y, r, X, return_ll=False):

	#compute beta with IRLS
	prev_beta = np.zeros(X.shape[1])
	dlp_f = np.empty(y.shape[0])
	d2lp_f = np.empty(y.shape[0])
	dummy = np.empty(y.shape[0])
	beta = prev_beta
	prev_ll = np.inf
	ll = 0
	tol = 1e-6
	iter=0
	while (np.abs(ll-prev_ll) > tol):
		iter+=1
		f = X.dot(beta)		
		lp = laplace_cython.likBinomLaplace(y, f, r, dlp_f, d2lp_f, dummy)
		d2lp_f = -d2lp_f
		dlp_beta = dlp_f.dot(X)
		d2lp_beta = -(X.T*d2lp_f).dot(X)		
		L = la.cho_factor(d2lp_beta, overwrite_a=True, check_finite=False)
		beta = prev_beta + la.cho_solve(L, dlp_beta)
		prev_beta = beta
		prev_ll = ll
		ll = lp
	
	if return_ll: return beta, ll
	return beta
	

	
def nll_bb2(bb_rho, covars, y, r, returnBeta=False):
	tol1 = 1e-6
	mudiff=1.0
	n_iter=0
	max_iter=50
	fixed_effects = np.zeros(covars.shape[1])
	while (mudiff>tol1 and n_iter<max_iter):
		n_iter+=1
		Xbeta = covars.dot(fixed_effects)
		mu = r * (1 - 1.0 / (np.exp(Xbeta) + 1))
		M_sqr = mu*(1-mu/r)*(1+(r-1)*bb_rho)	#equivalent to Gamma from the Carat paper
		M_sqr[M_sqr<1e-12] = 1e-12
		D = (mu * (1-mu/r))[:, np.newaxis] * covars
		M_sqr_inv = 1.0/M_sqr
		smallM = (M_sqr<=1e-12)
		M_sqr_inv[smallM] = 1e12		
		temp = D.T * M_sqr_inv
		DT_invOmega_D = temp.dot(D)
		DT_invOmega_yc = temp.dot(y-mu)
		L = la.cho_factor(DT_invOmega_D, overwrite_a=True, check_finite=False)
		fixed_effects += la.cho_solve(L, DT_invOmega_yc, overwrite_b=True, check_finite=False)
		Xbeta_new = covars.dot(fixed_effects)
		mu_new = r * (1 - 1.0 / (np.exp(Xbeta_new) + 1))
		mudiff = np.abs(mu-mu_new).sum()		
	if returnBeta: return fixed_effects
		
	
	sum_ab = (1-bb_rho) / bb_rho
	mu_p = 1.0 / (1 + np.exp(-covars.dot(fixed_effects)))
	alpha = mu_p * sum_ab
	beta = sum_ab - alpha
	ll = np.sum(scipy.special.betaln(y + alpha, r - y + beta) - scipy.special.betaln(alpha, beta))	
	if (np.isnan(ll)): return np.inf
	return -ll	



def perform_ewas_fixed_bb_lr(mcounts, counts, kernel, predictor, covars, verbose=False, out_file=None):
	K, yMat, rMat, covToTest, covars, snpNames = readMacauTest(mcounts, counts, kernel, predictor, covars, correctK=False)	
	covars_null = np.ones((yMat.shape[1],1))
	if (covars is not None): covars_null = np.concatenate((covars_null, covars), axis=1)
	covars_alt = np.concatenate((covars_null, np.row_stack(covToTest)), axis=1)	
	ewas_laplace = EWAS_Laplace()
	use_dispersion = True	
	chi2 = stats.chi2(1)
	U = np.eye(K.shape[0])
	s = np.ones(K.shape[0])
	beta0_null = np.zeros(covars_null.shape[1])
	beta0_alt = np.zeros(covars_alt.shape[1])
	
	#print header
	out_file_h = open(out_file, 'w')
	out_file_h.write('%s\t%s\t%s\t%s\n'%('index', 'id', 'test_stat', 'P-value'))
	
	test_stats = np.zeros(len(snpNames))
	t0_ewas = time.time()
	for i in xrange(len(snpNames)):
		snpName = snpNames[i]
		r_i = rMat[i,:]
		y_i = yMat[i,:]
		
		#exclude individuals with no data
		if (np.sum(r_i>0) < 3): continue			
		y_i = y_i[r_i>0]		
		covars_null_i = covars_null[r_i>0,:]
		covars_alt_i  = covars_alt[r_i>0,:]
		U_i  = U[r_i>0,:]
		r_i = r_i[r_i>0]	
		
		try:
			nll_null = optimize.minimize_scalar(nll_bb2, args=(covars_null_i, y_i, r_i), method='bounded', bounds=(0,1)).fun
			nll_alt  = optimize.minimize_scalar(nll_bb2, args=(covars_alt_i,  y_i, r_i), method='bounded', bounds=(0,1)).fun
			test_stats[i] = 2*(nll_null - nll_alt)
		except:
			test_stats[i] = np.nan
		
		out_file_h.write('%d\t%s\t%0.4f\t%0.5e\n'%(i+1, snpName, test_stats[i], chi2.sf(test_stats[i])))
		
	out_file_h.close()
	print
	print '#total EWAS time: %0.2f minutes'%((time.time()-t0_ewas) / 60.0)
	
	
			
	

def perform_ewas_laplace(mcounts, counts, kernel, predictor, covars, verbose=False, kernel2=None, out_file=None):	
	K, yMat, rMat, covToTest, covars, snpNames = readMacauTest(mcounts, counts, kernel, predictor, covars, correctK=False, kernel2=kernel2)	
	if (kernel2 is not None): K, K2 = K	
	returnGrad = True
	
	opt = 'L-BFGS-B'
	test = 'wald'
	inv_tol = -1 #1e-5
	num_simu = 0

	
	#create kernels
	num_kernels = 2
	if (kernel2 is not None): num_kernels+=1
	kernel_params = np.ones(num_kernels) * np.log(0.5)	
	kernels = np.empty((num_kernels, K.shape[0], K.shape[1]))
	kernels[0,:,:] = K	
	if (kernel2 is not None): kernels[1,:,:] = K2
	kernels[-1,:,:] = np.eye(K.shape[0])
	
	test_stats = np.zeros(len(snpNames))
	covars_null = np.ones((yMat.shape[1],1))
	if (covars is not None): covars_null = np.concatenate((covars_null, covars), axis=1)
	covars_alt = np.concatenate((covars_null, np.row_stack(covToTest)), axis=1)
	
	num_pcs = 0
	s,U = la.eigh(K)
	ind = np.argsort(s)[::-1]
	s = s[ind]
	U = U[:, ind]
	U = U[:, s>0]
	s = s[s>0]	
	covars_null_fixed = np.concatenate((covars_null, U[:, :num_pcs]), axis=1)
	covars_alt_fixed  = np.concatenate((covars_alt,  U[:, :num_pcs]), axis=1)
	
	
	ewas_laplace = EWAS_Laplace()	
	params0_null = np.concatenate((kernel_params, np.zeros(covars_null.shape[1])))
	params0_alt  = np.concatenate((kernel_params, np.zeros(covars_alt.shape[1])))
	
	
	out_file_h = open(out_file, 'w')
	out_file_h.write('%s\t%s\t%s\t%s\t%s\n'%('index', 'id', 'test_stat', 'P-value', 'time'))
	
	chi2 = stats.chi2(1)
	t0_ewas = time.time()
	for i in xrange(len(snpNames)):
		snpName = snpNames[i]
		
		r_i = rMat[i,:]
		y_i = yMat[i,:]
		if (np.sum(r_i>0) < 3): continue		
		is_good = (r_i>0)
		y_i = y_i[is_good]
		covars_null_i = np.asfortranarray(covars_null[is_good, :])
		covars_alt_i = np.asfortranarray(covars_alt[is_good, :])
		covars_null_fixed_i = covars_null_fixed[is_good, :]
		covars_alt_fixed_i = covars_alt_fixed[is_good, :]
		kernels_i = kernels[np.ix_(np.ones(num_kernels, dtype=np.bool), is_good, is_good)]
		r_i = r_i[is_good]
		if 	(np.all(y_i==r_i)): continue
		
		t0_site = time.time()
		test_stats[i], test_info = ewas_laplace.testBeta(params0_null=params0_null, params0_alt=params0_alt, kernels=kernels_i, covars_null=covars_null_i, covars_alt=covars_alt_i, y=y_i, r=r_i, verbose=verbose, opt=opt, covars_null_fixed=covars_null_fixed_i, covars_alt_fixed=covars_alt_fixed_i, test=test, returnGrad=returnGrad, inv_tol=inv_tol, num_simu=num_simu, ZC_direct=False)
		pvalue = chi2.sf(test_stats[i])
		if (pvalue < 1e-6):	#because of some strange bug...
			test_stats[i], test_info = ewas_laplace.testBeta(params0_null=params0_null, params0_alt=params0_alt, kernels=kernels_i, covars_null=covars_null_i, covars_alt=covars_alt_i, y=y_i, r=r_i, verbose=verbose, opt=opt, covars_null_fixed=covars_null_fixed_i, covars_alt_fixed=covars_alt_fixed_i, test=test, returnGrad=returnGrad, inv_tol=inv_tol, num_simu=num_simu, ZC_direct=True)
			pvalue = chi2.sf(test_stats[i])
		if (pvalue < 1e-16): test_stats[i], pvalue = np.nan, np.nan
		out_file_h.write('%d\t%s\t%0.4f\t%0.5e\t%0.2f\n'%(i+1, snpName, test_stats[i], pvalue, time.time()-t0_site))

	out_file_h.close()
	print
	print '#total EWAS time: %0.2f minutes'%((time.time()-t0_ewas) / 60.0)
		

		
		
		
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--mcounts', metavar='mcounts', default=None, required=True, help='mcounts file')
	parser.add_argument('--counts', metavar='counts', default=None, required=True, help='counts file')
	parser.add_argument('--kernel', metavar='kernel', default=None, required=True, help='kernel file')
	parser.add_argument('--kernel2', metavar='kernel2', default=None, help='kernel2 file')
	parser.add_argument('--predictor', metavar='predictor', default=None, required=True, help='predictor file')
	parser.add_argument('--covars', metavar='covars', default=None, help='covariates file')
	parser.add_argument('--verbose', metavar='verbose', type=int, default=0, help='verbosity level')
	parser.add_argument('--test', metavar='test', default='malax')
	
	parser.add_argument('--out', metavar='out', required=True, help='output file')
	args = parser.parse_args()	
	
	if (args.test == 'malax'):
		perform_ewas_laplace(args.mcounts, args.counts, args.kernel, args.predictor, args.covars, args.verbose>0, kernel2=args.kernel2, out_file=args.out)
	elif (args.test == 'bb'):
		perform_ewas_fixed_bb_lr(args.mcounts, args.counts, args.kernel, args.predictor, args.covars, args.verbose>0, out_file=args.out)
	else:
		raise Exception('unknown test')

	
	
	
	
	
	
	
	
	
	
