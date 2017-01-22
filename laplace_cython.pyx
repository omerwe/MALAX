# cython: profile=True


cdef extern from "math.h":
	float exp(float theta)
	float sqrt(float theta)
	float fabs(float theta)
	float log(float theta)
	float erfc(float theta)	
	float floor(float theta)	
	float cos(float theta)	
	float sin(float theta)	
	float log1p(float theta)
	float fmax(float theta1, float theta2)
	
cdef extern from "limits.h":
	int RAND_MAX
	
	
# cdef extern from "boost_invcdf.cpp":
	# double boost_normPPF(double x)
# cdef double boost_ppf(double x): return boost_normPPF(x)	

	
from libc.math cimport M_PI
from libc.stdlib cimport rand

import sys
import scipy.optimize as optimize
import scipy.linalg as la
import numpy as np
cimport numpy as np
cimport cython
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double norm(np.ndarray[double,ndim=1] x):
	cdef int n = x.shape[0], one = 1
	return blas.dnrm2(&n, &x[0], &one)
	
	
	
#Compute the log determinant ldA and the inverse iA of a square nxn matrix
#A = eye(n) + K*diag(W) from its LU decomposition; for negative definite A, we 
#return ldA = Inf. We also return mwiA = -diag(w)/A.
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef logdetA(np.ndarray[double,ndim=2] K, np.ndarray[double,ndim=1] W):
	cdef int n = K.shape[0], i, j, signU=1, detP=1
	cdef double ldA=0, log_abs_u_i, temp
	cdef np.ndarray[double, ndim=1] p = np.empty(n)
	cdef np.ndarray[double, ndim=2] A = np.empty((n,n)), L, U, P, cholSolve1, mwiA = np.empty((n,n)), iA
	
	#A = np.eye(n) + K * W[None,:]
	for i in xrange(n):
		for j in xrange(n):
			A[i,j] = K[i,j]*W[j]
		A[i,i] += 1
	#assert np.allclose(A, np.eye(n) + K * W[None,:]); print 'daaa'
		
	
	(P, L, U) = la.lu(A)
	for i in xrange(n):
		log_abs_u_i = log(abs(U[i,i]))
		ldA += log_abs_u_i				#det(L) = 1 and U triangular => det(A) = det(P)*prod(diag(U))
		if (U[i,i] < 0): signU = -signU
	
	
	#p = P.dot(np.arange(1,n+1))
	for i in xrange(n):
		p[i]=0
		for j in xrange(n): p[i] += P[i,j]*(j+1)
	
	for i in xrange(n):
		if (i+1 != p[i]):
			detP = -detP
			#j = np.where(p==(i+1))[0][0]
			for j in xrange(n):
				if (p[j] == i+1): break
			temp = p[i]
			p[i] = p[j]
			p[j] = temp

	if (signU != detP): ldA = np.nan 		#log becomes complex for negative values, encoded by nan		
	
	cholSolve1 = la.solve_triangular(L, P, lower=True, check_finite=False, overwrite_b=True)
	iA = la.solve_triangular(U, cholSolve1, lower=False, check_finite=False, overwrite_b=True)
	
	#mwiA = -iA * W[:, np.newaxis]
	for i in xrange(n):
		for j in xrange(n):
			mwiA[i,j] = -iA[i,j]*W[i]
	#assert np.allclose(mwiA, -iA * W[:, np.newaxis]); print 'yyyes'

	return ldA, iA, mwiA

	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ccdf(double x):
	cdef double a1 =  0.254829592
	cdef double a2 = -0.284496736
	cdef double a3 =  1.421413741
	cdef double a4 = -1.453152027
	cdef double a5 =  1.061405429
	cdef double p  =  0.3275911

	cdef int xsign = 1
	if (x < 0): xsign = -1        
	x = fabs(x)/sqrt(2.0)
	cdef double t = 1.0/(1.0 + p*x)
	cdef double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)

	return 0.5*(1.0 + xsign*y)
	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_normal():
	cdef double x,u,v,s

	# x = rand() / <float>(RAND_MAX)
	# if (x>0.999): x = boost_ppf(0.999)
	# if (x<0.001): x = boost_ppf(0.001)
	# else: x = boost_ppf(x)
	# return x
	
	
	#box-Muller method
	u = rand() / <float>(RAND_MAX)
	v = rand() / <float>(RAND_MAX)
	x = sqrt(-2*log(u)) * cos(2*M_PI*v)
	
	
	
	# #polar method
	# s=2
	# while (s>=1):
		# u = rand() / <float>(RAND_MAX)
		# v = rand() / <float>(RAND_MAX)
		# s = u**2 + v**2	
	# x = u * sqrt(-2*log(s)/s
	
	return x
	
	
	

	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef chol_inc(np.ndarray[double,ndim=2] K, double tol=1e-12, int rank=-1):
	
	cdef int m = K.shape[0], i, j, k, q, t
	cdef np.ndarray[long, ndim=1] p = np.empty(m, dtype=long)
	cdef np.ndarray[double,ndim=2] L = np.zeros((m,m))
	cdef double sum_diag=0, v, temp
	
	for i in xrange(m):
		p[i]=i
		L[i,0] = K[i, 0]

	for i in xrange(m):
		if (i==rank): break

		for j in xrange(i,m):
			L[j,j] = K[p[j], p[j]]
			for k in xrange(i): L[j,j] -= L[j, k]**2			 
			
		sum_diag=0
		v=L[i,i]; q=i
		for k in xrange(i,m):			
			sum_diag += L[k,k]
			if (L[k,k] > v):
				v = L[k,k]
				q=k
		if (sum_diag <= tol): break
		
		t = p[i]
		p[i] = p[q]
		p[q] = t
		
		for k in xrange(i):
			temp = L[i,k]
			L[i, k] = L[q, k]		
			L[q, k] = temp
		
		v = sqrt(v)
		L[i,i] = v

		#L[i+1:, i] = K[p[i], p[i+1:]] - L[i+1:, :i].dot(L[i,:i])
		#L[i+1:, i] /= v
		for k in xrange(i+1, m):
			L[k,i] = K[p[i], p[k]]
			for j in xrange(i):  L[k,i] -= L[k, j]*L[i,j]				
			L[k, i]  /= v
	
	return L, p, i+1	


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int pcg(np.ndarray[double,ndim=2] A, np.ndarray[double,ndim=1] b, np.ndarray[double, ndim=1] x, 
			np.ndarray[double,ndim=1] xmin, np.ndarray[double,ndim=1] r, np.ndarray[double,ndim=1] p, np.ndarray[double,ndim=1] q,
			double tol=1e-6, int maxit=-1):
			

	cdef int n = A.shape[0], i, one=1
	cdef double n2b = norm(b), oned=1.0, moned=-1.0, zerod=0.0, large=99999999999999999999
	cdef double tolb = tol * n2b                  				#Relative tolerance
	#cdef np.ndarray[double, ndim=1] xmin=np.empty(n), r=np.empty(n), p=np.empty(n), q=np.empty(n)
	cdef int flag=1, imin=0, iter, stag=0, moresteps=0, maxmsteps=5, maxstagsteps=3, ii
	cdef double normr, normr_act, relres, normrmin, rho=1.0, eps=1e-16, pq, alpha, beta, rho1, malpha
	
	if (maxit < 0):
		if (n<20): maxit=n
		else: maxit=20
	
	#xmin = x.copy	
	blas.dcopy(&n, &x[0], &one, &xmin[0], &one)
	
	#r = b - A.dot(x)
	blas.dcopy(&n, &b[0], &one, &r[0], &one)
	blas.dgemv('N', &n, &n, &moned, &A[0,0], &n, &x[0], &one, &oned, &r[0], &one)
	
	
	normr = norm(r) 				                #Norm of residual
	normr_act = normr
	
	if (normr <= tolb):    #Initial guess is a good enough solution
		flag = 0
		relres = normr / n2b
		iter = 0		
		return flag#,relres,iter
	normrmin = normr                  #Norm of minimum residual	
	
	if (n/50.0 < maxmsteps): maxmsteps = <int>(n/50.0)
	if (n-maxit < maxmsteps): maxmsteps = n-maxit	
	
	#loop over maxit iterations (unless convergence or failure)
	for ii in xrange(maxit):
		#z=r	(when there is no preconditioning...)
		
		rho1 = rho
		#rho = r.dot(r)
		rho = norm(r)**2
		
		if (rho==0 or rho>large):
			flag=4; break			
			
		if (ii == 0):
			#p=r.copy()
			blas.dcopy(&n, &r[0], &one, &p[0], &one)
		else:
			beta = rho / rho1
			if (beta==0 or beta>large):
				flag=4; break
			#p = r + beta * p
			for i in xrange(n): p[i] = r[i]+beta*p[i]
			
		#q = A.dot(p)		
		blas.dgemv('N', &n, &n, &oned, &A[0,0], &n, &p[0], &one, &zerod, &q[0], &one)
		
		#pq = p.dot(q)
		pq = blas.ddot(&n, &p[0], &one, &q[0], &one)
		if (pq <= 0 or pq>large):
			flag=4; break
		alpha = rho / pq	
		if (alpha>large):
			flag=4; break
			
		#Check for stagnation of the method
		if (norm(p) * abs(alpha) < eps*norm(x)): stag+=1
		else: stag=0
		
		#x += alpha*p  	           #form new iterate		
		blas.daxpy(&n, &alpha, &p[0], &one, &x[0], &one)
		
		#r -= alpha*q
		malpha = -alpha
		blas.daxpy(&n, &malpha, &q[0], &one, &r[0], &one)
		
		
		normr = norm(r)
		normr_act = normr
		
		#check for convergence
		if (normr <= tolb or stag >= maxstagsteps or moresteps>0):
			#r = b-A.dot(x)
			blas.dcopy(&n, &b[0], &one, &r[0], &one)
			blas.dgemv('N', &n, &n, &moned, &A[0,0], &n, &x[0], &one, &oned, &r[0], &one)
			
			normr_act = norm(r)
			if (normr_act <= tolb):
				flag=0; iter=ii+1; break				
				
			if (stag >= maxstagsteps and moresteps == 0): stag=0
			moresteps+=1
			if (moresteps >= maxmsteps):
				flag=3; iter=ii+1; break
				
		
		if (normr_act < normrmin):      #update minimal norm quantities
			normrmin = normr_act
			#xmin = x.copy()
			blas.dcopy(&n, &x[0], &one, &xmin[0], &one)
			imin = ii+1
			
		if (stag >= maxstagsteps):
			flag=3; break


	#returned solution is first with minimal residual
	if (flag == 0): relres = normr_act / n2b
	else:
		#r = b - A.dot(xmin)
		blas.dcopy(&n, &b[0], &one, &r[0], &one)
		blas.dgemv('N', &n, &n, &moned, &A[0,0], &n, &xmin[0], &one, &oned, &r[0], &one)
		
		if (norm(r) <= normr_act):
			#x = xmin.copy()
			blas.dcopy(&n, &xmin[0], &one, &x[0], &one)
			iter = imin
			relres = norm(r) / n2b
		else:
			iter = ii+1
			relres = normr_act / n2b
			
	return flag#,relres,iter




	


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef laplace_nlZ(np.ndarray[double, ndim=1] params, np.ndarray[double, ndim=3] kernels, np.ndarray[double, ndim=2] covars, np.ndarray[double, ndim=1] m0, np.ndarray[long, ndim=1] y, np.ndarray[long, ndim=1] r, np.uint8_t returnLL, np.uint8_t returnGrad, np.uint8_t returnPost, np.uint8_t returnF,  np.ndarray[double, ndim=1] alpha0, double inv_tol=-1.0, int num_simu=0, np.uint8_t  allow_negW=False, ZC_direct=True):
	"""
	params: The variance components and fixed effects. The first kernels.shape[0] parameters are variance components, and the rest are fixed effects
	kernels: An array of matrices. 
	covars: covariates
	y: Total number of reads
	r: Number of positive reads
	"""

	cdef int i, i2, info, n=y.shape[0], one=1, num_covars=covars.shape[1], n_sqr=y.shape[0]**2
	cdef np.ndarray[double, ndim=3] scaled_kernels = np.zeros((kernels.shape[0], kernels.shape[1], kernels.shape[2]))
	cdef np.ndarray[double, ndim=2] K = np.zeros((kernels.shape[1], kernels.shape[2])), K2=np.empty((n,n), order='C'), L, iA=None
	cdef np.ndarray[double, ndim=1] m=np.empty(n), alpha, f=np.empty(n), W=np.empty(n), sW=np.empty(n), grad, d=np.empty(n), dlp=np.empty(n), d3lp=np.empty(n)
	cdef double lp_sum, factor, nlZ=0.0, oned=1.0, exp_param, sqrt_mean_diag_L=0.0, ldA=0.0
	
	cdef np.uint8_t isWneg, chol_success
	cdef np.uint8_t use_irls = True #(y.shape[0] < 3000)
	#cdef object opt_obj
	
	#compute combined kernel
	for i in range(kernels.shape[0]):
		exp_param = exp(params[i])
		#scaled_kernels[i,:,:] = kernels[i,:,:] * exp(params[i])
		blas.daxpy(&n_sqr, &exp_param, &kernels[i,0,0], &one, &scaled_kernels[i,0,0], &one)
		blas.daxpy(&n_sqr, &exp_param, &kernels[i,0,0], &one, &K[0,0], &one)
	#K = scaled_kernels.sum(axis=0)
	
	#compute mean vector
	if (covars.shape[1] == 0): m = m0
	else: 
		#m = m0 + covars.dot(params[kernels.shape[0]:])
		blas.dcopy(&n, &m0[0], &one, &m[0], &one)
		#covars = np.array(covars, order='F')
		blas.dgemv('N', &n, &num_covars, &oned, &covars[0,0], &n, &params[kernels.shape[0]], &one, &oned, &m[0], &one)
		#assert np.allclose(m, m0 + covars.dot(params[kernels.shape[0]:])); print 'yes'
		
		

	#compute alpha
	if use_irls:
		alpha, f = laplace_irls(m, K, y, r, alpha0, inv_tol=inv_tol)
		if returnF: return f
	else:
		raise Exception('not supported in cython mode...')
		#alpha = opt_onj.x
	#print 'input alpha0:', alpha0
	
	lp_sum = likBinomLaplace(y, f, r, dlp, W, d3lp)
	blas.dcopy(&n, &alpha[0], &one, &alpha0[0], &one)		#alpha0[:] = alpha
	
	
	#sW = np.sqrt(np.abs(W)) * np.sign(W)	
	#isWneg = np.any(W<0)
	isWneg = False
	for i in xrange(n):
		if (W[i] > 0): sW[i] = sqrt(W[i])
		else:		
			sW[i] = sqrt(-W[i])
			isWneg = True
	
	#switch between Cholesky and LU decomposition mode	
	if isWneg:	
		if (not allow_negW): raise Exception('negative W found')		
		ldA, iA, K2 = logdetA(K, W)   #A=eye(n)+K*W is as safe as symmetric B		
		if (ldA != ldA): raise Exception('complex values found')
	else:
		try:
			#K2 = np.eye(n) + (sW[:,np.newaxis]*sW[np.newaxis,:])*K
			for i in xrange(n):
				for i2 in xrange(i,n):
					K2[i,i2] = sW[i]*sW[i2]*K[i,i2]					
					K2[i2,i] = K2[i,i2]
				K2[i,i] += 1
				
			K2 = la.cholesky(K2, lower=True, overwrite_a=True, check_finite=False)
			#lapack.dpotrf('L', &n, &K2[0,0], &n, &info)
			#if (info != 0):
			#	raise Exception('dpotrf failed')
			# print la.cholesky(np.outer(sW,sW)*K + np.eye(n), lower=True)[:10, :10]
			# print
			# print K2[:10, :10]
			# print '--------------------'
			# for i in xrange(n):
				# for i2 in xrange(i,n): K2[i,i2] = 0					
			
		except:
			factor = 1e-8			
			for i in xrange(n):	d[i] = sW[i]*sW[i]*K[i,i] + 1				
			chol_success = False			
			while (factor < 32000):
				factor*=2
				try:
					#L = la.cholesky(K2, overwrite_a=False, check_finite=False) 	#recompute
					for i in xrange(n):
						for i2 in xrange(i,n):
							K2[i,i2] = sW[i]*sW[i2]*K[i,i2]
							#if (i==i2): K2[i,i2] += 1
							K2[i2,i] = K2[i,i2]
						K2[i,i] += 1
						K2[i,i] *= (1+factor)
					lapack.dpotrf('U', &n, &K2[0,0], &n, &info)
					if (info != 0):
						raise Exception('dpotrf failed')					
					chol_success = True					
				except: pass
				if (chol_success): break
			if (not chol_success):
				for i in xrange(n):
					for i2 in xrange(i,n):
						K2[i,i2] = sW[i]*sW[i2]*K[i,i2]						
						K2[i2,i] = K2[i,i2]			
					K2[i,i] += 1
				la.cholesky(K2, overwrite_a=True, check_finite=False) 	#recompute to raise the correct error
			
			###L /= np.sqrt(np.mean(np.diag(K2)))			
			for i in xrange(n): sqrt_mean_diag_L += d[i]
			sqrt_mean_diag_L *= (1+factor) / n
			sqrt_mean_diag_L = sqrt(sqrt_mean_diag_L)
			for i in xrange(n):
				for i2 in xrange(n):
					K2[i,i2] /= sqrt_mean_diag_L
			
			
			
			
		if returnLL:
			if isWneg:				
				for i in xrange(n): nlZ += (f[i]-m[i]) * alpha[i]/2.0
				nlZ -= lp_sum
				nlZ += ldA/2.0
			else:
				#nlZ = alpha.dot(f-m)/2.0 + np.sum(np.log(np.diag(K2))) - lp_sum # ..(f-m)/2 -lp +ln|B|/2			
				for i in xrange(n): nlZ += (f[i]-m[i]) * alpha[i]/2.0 + log(K2[i,i])
				nlZ -= lp_sum
		
	if (not returnGrad and not returnPost): return nlZ
	if returnGrad:
		grad = laplace_grad(scaled_kernels, covars, K, alpha, K2, isWneg, iA, sW, dlp, d3lp, num_simu=num_simu, ZC_direct=ZC_direct)
		#print params, nlZ, grad
		if returnLL: return nlZ, grad
		else: return grad, alpha
		
	if returnPost:
		print 'returnPost'
		return alpha, K2, sW, f
		

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)		
cdef np.ndarray[double, ndim=1] laplace_grad(np.ndarray[double, ndim=3] scaled_kernels, np.ndarray[double, ndim=2] covars, np.ndarray[double, ndim=2] K, np.ndarray[double, ndim=1] alpha, np.ndarray[double, ndim=2] L, np.uint8_t isWneg, np.ndarray[double, ndim=2] iA, np.ndarray[double, ndim=1] sW, np.ndarray[double, ndim=1] dlp, np.ndarray[double, ndim=1] d3lp, int num_simu=0, np.uint8_t ZC_direct=1):

	cdef int n=L.shape[0], n_sqr=L.shape[0]**2, nc=covars.shape[0]*covars.shape[1], j, info, k
	cdef np.ndarray[double, ndim=1] grad = np.empty(scaled_kernels.shape[0] + covars.shape[1])
	cdef np.ndarray[double, ndim=2] Z=np.zeros((n,n), order='C'), C=np.empty((n,n), order='F'), ZC=np.empty((n, covars.shape[1]), order='F'), KZC = np.empty((n, covars.shape[1]),order='F'), covars_temp = np.empty((n, covars.shape[1]), order='F')
	cdef np.ndarray[double, ndim=1] dfhat=np.empty(n), b=np.empty(n), dK_alpha = np.empty(n), Zb=np.empty(n), KZb=np.empty(n), sWb = np.empty(n)
	cdef int i, one=1, num_covars=covars.shape[1]
	cdef double oned=1.0, zerod=0.0, neg_oned=-1.0, gi, tempd
	
	if isWneg:                  		# switch between Cholesky and LU decomposition mode
		#Z = -L		             		# inv(K+inv(W))
		for i in xrange(n):
			for j in xrange(n):
				Z[i,j] = -L[i,j]
		
		#g = np.sum(iA*K, axis=1)/2.0 	# deriv. of ln|B| wrt W; g = diag(inv(inv(K)+diag(W)))/2
		for i in xrange(n):
			gi=0
			for j in xrange(n): gi += iA[i,j]*K[i,j]				
			gi/=2.0
			dfhat[i] = gi*d3lp[i]						# deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
		
	else:	
		for j in xrange(n):			
			for i in xrange(n):	C[i,j] = K[i,j]*sW[i]				
		#C = la.solve_triangular(L, C, lower=True, check_finite=False, overwrite_b=True)					# deriv. of ln|B| wrt W
		blas.dtrsm('L', 'L', 'N', 'N', &n, &n, &oned, &L[0,0], &n, &C[0,0], &n)								# deriv. of ln|B| wrt W		
		#g = (np.diag(K) - np.sum(C**2, axis=0)) / 2.0        												# g = diag(inv(inv(K)+W))/2			
		for i in xrange(n):
			gi = (K[i,i] - blas.dnrm2(&n, &C[0,i], &one)**2) / 2.0
			dfhat[i] = gi*d3lp[i]						# deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
		
		if (num_simu <= 0):
			for j in xrange(n): Z[j,j]=sW[j]
			#Z = la.cho_solve((L, True), np.diag(sW), overwrite_b=True, check_finite=False) * sW[:, np.newaxis]	# sW*inv(B)*sW=inv(K+inv(W))		
			lapack.dpotrs('L', &n, &n, &L[0,0], &n, &Z[0,0], &n, &info)											# sW*inv(B)*sW=inv(K+inv(W))		
			if (info!=0):
				raise Exception('dpotrs failed')
			for j in xrange(n):
				for i in xrange(n):
					Z[i,j] *= sW[i]
					
					
	#dfhat = g*d3lp
	
	#variance components gradient		
	for i in xrange(scaled_kernels.shape[0]):
		#dK = scaled_kernels[i,:,:]
		#b = dK.dot(dlp)
		blas.dgemv('N', &n, &n, &oned, &scaled_kernels[i,0,0], &n, &dlp[0], &one, &zerod, &b[0], &one)
		
		#grad[i]  = np.sum(Z*dK)/2.0 - alpha.dot(dK.dot(alpha))/2.0	 # explicit part			
		blas.dgemv('N', &n, &n, &oned, &scaled_kernels[i,0,0], &n, &alpha[0], &one, &zerod, &dK_alpha[0], &one)	#dK_alpha = scaled_kernels[i].dot(alpha)
		grad[i] = -blas.ddot(&n, &alpha[0], &one, &dK_alpha[0], &one)											#grad[i] = -alpha.dot(dK_alpha)
		
		if (num_simu <= 0):
			grad[i] += blas.ddot(&n_sqr, &Z[0,0], &one, &scaled_kernels[i,0,0], &one)								#grad[i] += np.sum(Z * scaled_kernels[i])		
		else:
			for j in xrange(num_simu):
				for k in xrange(n):	Zb[k] = random_normal()
				#Zb[:] = np.random.randn(n)
				blas.dgemv('N', &n, &n, &oned, &scaled_kernels[i,0,0], &n, &Zb[0], &one, &zerod, &sWb[0], &one)
				for k in xrange(n): sWb[k] *= sW[k]
				lapack.dpotrs('L', &n, &one, &L[0,0], &n, &sWb[0], &n, &info)
				for k in xrange(n): sWb[k] *= sW[k]
				grad[i] += blas.ddot(&n, &Zb[0], &one, &sWb[0], &one) / float(num_simu)
		
		grad[i] /= 2.0
		
		#B = np.eye(n) + (sW[:,np.newaxis]*sW[np.newaxis,:])*K
		#assert np.allclose(la.cho_solve((L, False), np.eye(n)), la.inv(B))		
		#assert np.allclose(K.dot(Z).dot(b), K.dot(sW*la.cho_solve((L, False), sW*b)))		

		###grad[i] -= dfhat.dot(b - K.dot(Z.dot(b)))            		 # implicit part
		
		######### compute Zb ##########
		blas.dgemv('N', &n, &n, &oned, &Z[0,0], &n, &b[0], &one, &zerod, &Zb[0], &one)		#Zb = Z.dot(b)		
		###alternative computation for Zb that doesn't use Z...
		# for j in xrange(n): sWb[j] = b[j]*sW[j]	#sWb = sW * b
		# lapack.dpotrs('L', &n, &one, &L[0,0], &n, &sWb[0], &n, &info)	#swB = inv(B).dot(sWb)
		# if (info != 0):
			# raise Exception('dpotrs failed')
		# for j in xrange(n): Zb[j] = sWb[j] * sW[j]
		# # # #assert np.allclose(Zb, Z.dot(b))
		
		blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &Zb[0], &one, &zerod, &KZb[0], &one)	#KZb = K.dot(Zb)		
		blas.daxpy(&n, &neg_oned, &KZb[0], &one, &b[0], &one)	#b -= KZb
		grad[i] -= blas.ddot(&n, &dfhat[0], &one, &b[0], &one)
		
		

	#fixed effects gradient
	if (covars.shape[1] > 0):
		
		#grad[scaled_kernels.shape[0]:] = -alpha.dot(covars)
		blas.dgemv('T', &n, &num_covars, &neg_oned, &covars[0,0], &n, &alpha[0], &one, &zerod, &grad[scaled_kernels.shape[0]], &one)
		
		################ compute KZC = K.dot(Z.dot(covars)) ##################
		
		#compute ZC
		if (isWneg or ZC_direct):
			blas.dgemm('T', 'N', &n, &num_covars, &n, &oned, &Z[0,0], &n, &covars[0,0], &n, &zerod, &ZC[0,0], &n)	#ZC = Z.dot(covars)
		
		else:
			###alternative computation for ZC that doesn't use Z...
			for j in xrange(num_covars):			
				for i in xrange(n):			
					ZC[i,j] = sW[i]*covars[i,j]	#ZC = np.diag(sW).dot(covars)
			lapack.dpotrs('L', &n, &num_covars, &L[0,0], &n, &ZC[0,0], &n, &info)  #ZC = inv(B).dot(ZC)
			if (info != 0):
				raise Exception('dpotrs failed')
			for j in xrange(num_covars):
				for i in xrange(n):			
					ZC[i,j] *= sW[i]	#ZC = np.diag(sW).dot(covars)
				
		blas.dgemm('N', 'N', &n, &num_covars, &n, &oned, &K[0,0], &n, &ZC[0,0], &n, &zerod, &KZC[0,0], &n)		#KZC = K.dot(ZC)
		#assert np.allclose(KZC, K.dot(Z.dot(covars))); print 'oh yes'
		
		#b = dfhat.dot(covars - KZC)
		blas.dcopy(&nc, &covars[0,0], &one, &covars_temp[0,0], &one)			#covars_temp = covars
		blas.daxpy(&nc, &neg_oned, &KZC[0,0], &one, &covars_temp[0,0], &one)	#covars_temp -= KZC	
		blas.dgemv('T', &n, &num_covars, &oned, &covars_temp[0,0], &n, &dfhat[0], &one, &zerod, &b[0], &one)
				
		#grad[scaled_kernels.shape[0]:] -= b
		blas.daxpy(&num_covars, &neg_oned, &b[0], &one, &grad[scaled_kernels.shape[0]], &one)	#grad[...:] -= b		
		#assert np.allclose(grad[scaled_kernels.shape[0]:], -alpha.dot(covars) - dfhat.dot(covars - K.dot(Z.dot(covars)))); print 'yes!!!!!!'

	
	return grad
	
	
		
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef laplace_irls(np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=2] K, np.ndarray[long, ndim=1] y, np.ndarray[long, ndim=1] r, np.ndarray[double, ndim=1] alpha, double inv_tol=-1.0):
	cdef int n = y.shape[0], n_sqr=y.shape[0]**2
	cdef int maxit = 20
	cdef double Wmin = 0.0
	cdef double tol = 1e-6
	cdef int smin_line = 0
	cdef int smax_line = 2
	cdef int nmax_line = 10
	cdef np.ndarray[double, ndim=1] f = np.empty(n), cholSolve2, x_temp
	cdef np.ndarray[double, ndim=2] B = np.empty((n,n))
	cdef np.ndarray[double, ndim=1] b=np.empty(n), sW=np.empty(n), dalpha=np.empty(n), B2=np.empty(n), dlp=np.empty(n), W=np.empty(n), Kb=np.empty(n)
	cdef np.uint8_t chol_lower = False
	cdef double lp_sum
	cdef int i, i2
	
	cdef int inv_flag, inv_iter
	cdef double inv_relres
	
	cdef double oned = 1.0, zerod=0.0
	cdef int one = 1
	
	cdef double Psi_old = np.inf, Psi_new, s_line
	cdef int it=0, info
	#cdef np.ndarray[double, ndim=2] eyeN = np.eye(K.shape[0])
	
	if (inv_tol > 0):
		x_temp = np.empty(n)
		if (tol < inv_tol): tol = inv_tol
	
	
	#f = K.dot(alpha) + m
	blas.dcopy(&n, &m[0], &one, &f[0], &one)
	blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &alpha[0], &one, &oned, &f[0], &one)	
	
	lp_sum = likBinomLaplace(y, f, r, dlp, W, sW)
	
	Psi_new = psi_fast(0, alpha, alpha, m, K, y, r, sW, Kb)	
	while (Psi_old - Psi_new > tol and it<maxit): #begin Newton			
		Psi_old = Psi_new
		it+=1		
		
		#W[W < Wmin] = Wmin	#limit stepsize
		#b = W*(f-m) + dlp		
		#sW = np.sqrt(W)
		for i in xrange(n):
			if (W[i] < Wmin): W[i] = Wmin
			b[i]=W[i] * (f[i]-m[i]) + dlp[i]	
			sW[i] = sqrt(W[i])		
		
		
		#B = eyeN + (sW[:,np.newaxis]*sW[np.newaxis,:])*K
		for i in xrange(n):
			for i2 in xrange(i,n):
				B[i,i2] = sW[i] * sW[i2] * K[i,i2]
				B[i2,i] = B[i,i2]
			B[i,i] += 1
		
		if (inv_tol<0):
			####L = la.cholesky(B, overwrite_a=False, check_finite=False, lower=chol_lower)   #L'*L=B=eye(n)+sW*K*sW
			lapack.dpotrf('L', &n, &B[0,0], &n, &info)
			if (info != 0):
				raise Exception('dpotrf failed')
		
		
		#Kb = K.dot(b)			
		blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &b[0], &one, &zerod, &Kb[0], &one)	
			
		#B2 = sW * (K.dot(b))
		for i in xrange(n): B2[i] = sW[i] * Kb[i]
		
		if (inv_tol<0):
			####temp = la.cho_solve((L, chol_lower), B2, overwrite_b=False, check_finite=False)
			####cholSolve2 = la.cho_solve((B, chol_lower), B2, overwrite_b=False, check_finite=False)
			lapack.dpotrs('L', &n, &one, &B[0,0], &n, &B2[0], &n, &info)
			if (info != 0):
				raise Exception('dpotrs failed')		
			#####assert np.allclose(B2, cholSolve2); print 'ook'
		else:
			#R, p, k = chol_inc(K, rank=10)
			#x_temp = (la.cho_solve((R, True), y[p]))[np.argsort(p)]
			for i in xrange(n): x_temp[i]=0
			inv_flag = pcg(B, B2, x_temp, tol=inv_tol, xmin=Kb, r=dalpha, p=dlp, q=W)
			#print inv_flag, inv_relres, inv_iter
		
		####assert np.allclose(temp, cholSolve2); print 'triil'
		
		
		#dalpha = b - sW*cholSolve2 - alpha		#Newton dir + line search
		if (inv_tol < 0):
			for i in xrange(n): dalpha[i] = b[i] - sW[i]*B2[i] - alpha[i]
		else:
			for i in xrange(n): dalpha[i] = b[i] - sW[i]*x_temp[i] - alpha[i]
		
		s_line = opt_laplace_step_size(smin_line, smax_line, alpha, dalpha, m, K, y, r, dummy1=sW, dummy2=Kb)
		blas.daxpy(&n, &s_line, &dalpha[0], &one, &alpha[0], &one)	#alpha += s_line*dalpha		
		
		#(Psi_new, f, dlp, W) = laplace_Psi(alpha, m, K, y, r)
		Psi_new = laplace_Psi(alpha, m, K, y, r, f, dlp, W, sW)
	
	return alpha, f
	
	
	
#Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where  f = K.dot(alpha)+m	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double laplace_Psi(np.ndarray[double, ndim=1] alpha, np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=2] K, np.ndarray[long, ndim=1] y, np.ndarray[long, ndim=1] r,
				 np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] dlp, np.ndarray[double, ndim=1] W, np.ndarray[double, ndim=1] dummy):

	cdef int n = alpha.shape[0]	
	cdef int one=1
	cdef double zerod=0.0, oned=1.0, mone=-1.0
	#cdef np.ndarray[double, ndim=1] dpsi=np.empty(n), d2lp=np.empty(n), alpha2=np.empty(n)
	cdef double psi, lp_sum

	# f = K.dot(alpha)+m	
	# psi = alpha.dot(f-m)/2.0 - lp_sum
	# dpsi = K.dot(alpha-dlp)
	# return psi, dpsi, f, alpha, dlp, -d2lp	

						
	# #f = K.dot(alpha) + m
	blas.dcopy(&n, &m[0], &one, &f[0], &one)
	blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &alpha[0], &one, &oned, &f[0], &one)	
	#assert np.allclose(f, K.dot(alpha) + m); print 'yes1'
	
	lp_sum = likBinomLaplace(y, f, r, dlp, W, dummy)
	
	# #psi = alpha.dot(f-m)/2.0 - lp.sum()
	blas.dcopy(&n, &f[0], &one, &dummy[0], &one)			#dummy=f
	blas.daxpy(&n, &mone, &m[0], &one, &dummy[0], &one)		#dummy-=m
	psi = blas.ddot(&n, &alpha[0], &one, &dummy[0], &one)	#psi = alpha.dot(dummy)
	psi /= 2.0
	psi -= lp_sum
	#assert np.allclose(psi, alpha.dot(f-m)/2.0 - lp_sum); print 'yes2'
	
	#dpsi = K.dot(alpha-dlp)
	#blas.dcopy(&n, &alpha[0], &one, &alpha2[0], &one)		#alpha2 = alpha
	#blas.daxpy(&n, &mone, &dlp[0], &one, &alpha2[0], &one)	#alpha2 -= dlp
	#blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &alpha2[0], &one, &zerod, &dpsi[0], &one)	
	#assert np.allclose(dpsi, K.dot(alpha-dlp)); print 'yes3'
	
	return psi
	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef laplace_Psi_line(double s, np.ndarray[double, ndim=1] alpha, np.ndarray[double, ndim=1] dalpha, np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=2] K, np.ndarray[long, ndim=1] y, np.ndarray[long, ndim=1] r, np.uint8_t psi_only=True):
	alpha = alpha + s*dalpha
	cdef np.ndarray[double, ndim=1] f = K.dot(alpha)+m
	cdef np.ndarray[double, ndim=1] dpsi, dlp, d2lp
	cdef double psi, lp_sum	
	lp_sum = likBinomLaplace_fast(y, f, r)
	psi = alpha.dot(f-m)/2.0 - lp_sum
	return psi
	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double likBinomLaplace(np.ndarray[long, ndim=1] y, np.ndarray[double, ndim=1]  f, np.ndarray[long, ndim=1] r,
		np.ndarray[double, ndim=1] dlp, np.ndarray[double, ndim=1] neg_d2lp, np.ndarray[double, ndim=1] d3lp):
	
	cdef int n = y.shape[0]
	cdef int i	
	cdef double log_one_minus_Pi2, sum_lp=0, pi, exp_mfi, pi_denom	
	
	## new fast code  ##
	for i in xrange(y.shape[0]):
		exp_mfi = exp(-f[i])
		pi_denom = 1 + exp_mfi
		pi = 1.0 / (pi_denom)
		if (f[i] < -50): log_one_minus_Pi2 = log(1 - pi)
		else: log_one_minus_Pi2 = -log(pi_denom) - f[i]
		
		if (y[i]==0): sum_lp += r[i]*log_one_minus_Pi2			
		elif (y[i]==r[i]): sum_lp += -y[i] * log(1 + exp_mfi)			
		else: sum_lp += -y[i] * log(1 + exp_mfi) + (r[i]-y[i]) * log_one_minus_Pi2			
		dlp[i] = y[i] - r[i]*pi
		if (f[i] < -50):
			neg_d2lp[i]=0
			d3lp[i]=0
		else:
			neg_d2lp[i] = - (-pi**2 * (pi_denom - 1) * r[i])
			d3lp[i] = -neg_d2lp[i] * (2 * exp(log_one_minus_Pi2) - 1)

	return sum_lp

	
	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double likBinomLaplace_fast(np.ndarray[long, ndim=1] y, np.ndarray[double, ndim=1]  f, np.ndarray[long, ndim=1] r):
	cdef int i	
	cdef double log_one_minus_Pi2, sum_lp=0
	
	for i in xrange(y.shape[0]):
		if (y[i]==0):
			if (f[i] < -50): log_one_minus_Pi2 = log(1 - 1.0/(1+exp(-f[i])))
			else: log_one_minus_Pi2 = -log(1 + exp(-f[i])) -f[i]			
			sum_lp += r[i]*log_one_minus_Pi2
		elif (y[i]==r[i]):
			sum_lp += -y[i] * log(1 + exp(-f[i]))
		else:
			if (f[i] < -50): log_one_minus_Pi2 = log(1 - 1.0/(1+exp(-f[i])))
			else: log_one_minus_Pi2 = -log(1 + exp(-f[i])) -f[i]
			sum_lp += -y[i] * log(1 + exp(-f[i])) + (r[i]-y[i]) * log_one_minus_Pi2
		
	return sum_lp



	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double psi_fast(double x, np.ndarray[double,ndim=1] alpha, np.ndarray[double,ndim=1] dalpha, np.ndarray[double,ndim=1] m, 
						np.ndarray[double,ndim=2] K, np.ndarray[long,ndim=1] y, np.ndarray[long,ndim=1] binom_r,
						np.ndarray[double, ndim=1] f, np.ndarray[double, ndim=1] alpha_temp):
					
	cdef int n = alpha.shape[0]
	#cdef np.ndarray[double, ndim=1] lp=np.empty(n)
	cdef int one=1
	cdef double zerod=0.0, oned=1.0, mone=-1.0
	cdef double fx
	cdef double lp_sum
						
						
	###fx = laplace_Psi_line(x, alpha, dalpha, m, K, y, binom_r)
	#alpha_temp = alpha + x*dalpha
	blas.dcopy(&n, &alpha[0], &one, &alpha_temp[0], &one)
	blas.daxpy(&n, &x, &dalpha[0], &one, &alpha_temp[0], &one)
	#assert np.allclose(alpha_temp, alpha + x*dalpha); print 'yes'
	
	#f = K.dot(alpha_temp) + m
	blas.dcopy(&n, &m[0], &one, &f[0], &one)
	blas.dgemv('N', &n, &n, &oned, &K[0,0], &n, &alpha_temp[0], &one, &oned, &f[0], &one)
	#assert np.allclose(f, K.dot(alpha_temp) + m); print 'yes2'
	
	lp_sum = likBinomLaplace_fast(y, f, binom_r)
	
	#fx = alpha_temp.dot(f-m)/2.0 - lp.sum()
	blas.daxpy(&n, &mone, &m[0], &one, &f[0], &one)	#f -= m
	fx = blas.ddot(&n, &alpha_temp[0], &one, &f[0], &one)	#fx = alpha_temp.dot(f)
	fx /= 2.0
	fx -= lp_sum
	#assert np.allclose(fx, alpha_temp.dot(f)/2.0 - lp.sum()); print 'yes3'						
						
	return fx
	
	
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double opt_laplace_step_size(double x1, double x2, np.ndarray[double,ndim=1] alpha, np.ndarray[double,ndim=1] dalpha, np.ndarray[double,ndim=1] m, 
						np.ndarray[double,ndim=2] K, np.ndarray[long,ndim=1] y, np.ndarray[long,ndim=1] binom_r, np.ndarray[double,ndim=1] dummy1, np.ndarray[double,ndim=1] dummy2):

	cdef int n = alpha.shape[0]
	cdef double xatol=1e-5 
	cdef int maxiter=500
	cdef int flag=0, num=1, si
	cdef double sqrt_eps = sqrt(2.2e-16)
	cdef double golden_mean = 0.5 * (3.0 - sqrt(5.0))
	cdef double a=x1, b=x2
	cdef double fulc = a + golden_mean * (b - a)
	cdef double nfc=fulc, xf=fulc
	cdef double rat = 0.0, e=0.0, x=xf
	cdef double xm = 0.5 * (a + b)
	cdef double tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
	cdef double tol2 = 2.0 * tol1
	cdef np.uint8_t golden
	cdef double r,q,p, diff
	cdef double fx, ffulc, fnfc, fu
	cdef int i
	
	cdef int one=1
	cdef double zerod=0.0, oned=1.0, mone=-1.0
	

	fx = psi_fast(x, alpha, dalpha, m, K, y, binom_r, dummy1, dummy2)
	#assert np.isclose(fx, laplace_Psi_line(x, alpha, dalpha, m, K, y, binom_r)); print 'yes'
	
	ffulc = fnfc = fx

	while (fabs(xf - xm) > (tol2 - 0.5 * (b - a))):
		golden = True
		# Check for parabolic fit
		if fabs(e) > tol1:
			golden = False
			r = (xf - nfc) * (fx - ffulc)
			q = (xf - fulc) * (fx - fnfc)
			p = (xf - fulc) * q - (xf - nfc) * r
			q = 2.0 * (q - r)
			if q > 0.0: p = -p                
			q = fabs(q)
			r = e
			e = rat

			# Check for acceptability of parabola
			if ((fabs(p) < fabs(0.5*q*r)) and (p > q*(a - xf)) and (p < q * (b - xf))):
				rat = (p + 0.0) / q
				x = xf + rat
				if ((x - a) < tol2) or ((b - x) < tol2):
					diff = xm-xf
					si = (diff>=0) - (diff<0)
					rat = tol1 * si
			else: golden = True      # do a golden section step
				

		if golden:  # Do a golden-section step
			if xf >= xm: e = a - xf                
			else: e = b - xf                
			rat = golden_mean*e
		
		si = (rat>=0) - (rat<0)
		x = xf + si * fmax(fabs(rat), tol1)

		fu = psi_fast(x, alpha, dalpha, m, K, y, binom_r, dummy1, dummy2)
		#assert np.allclose(fu, laplace_Psi_line(x, alpha, dalpha, m, K, y, binom_r)); print 'yes22'
		
		
		num += 1        

		if fu <= fx:
			if x >= xf: a = xf                
			else: b = xf                
			fulc, ffulc = nfc, fnfc
			nfc, fnfc = xf, fx
			xf, fx = x, fu
		else:
			if x < xf: a = x                
			else: b = x                
			if (fu <= fnfc) or (nfc == xf):
				fulc, ffulc = nfc, fnfc
				nfc, fnfc = x, fu
			elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
				fulc, ffulc = x, fu

		xm = 0.5 * (a + b)
		tol1 = sqrt_eps * fabs(xf) + xatol / 3.0
		tol2 = 2.0 * tol1

		if num >= maxiter:
			flag = 1
			break

	return xf


	
