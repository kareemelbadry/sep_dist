'''
This document contains some simple utility functions for fitting the separation distributions of wide binaries, following the approach of El-Badry & Rix 2018
'''
from __future__ import division
import numpy as np


def ln_likelihood_double_power_law_with_weights(p1, p2, log_ab, loga_i, contrast, d_pc, 
    loga_min, loga_max):
    '''
    log-likelihood function for a broken power law separation distribution
    
    p1: float; the power-law exponent at loga < log_ab
    p2: float; the power-law exponent at loga > log_ab
    log_ab: float; the log-separation at which there's a break in the power law
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    beta, theta_0 = get_theta_0_and_beta_this_delta_G(delta_G = contrast)
    denominator = analytic_integrand_weights_double_power_law(p1 = p1, p2 = p2, 
        log_ab = log_ab, beta = beta, theta_0 = theta_0, d_pc= d_pc, 
        loga_min = loga_min, loga_max = loga_max)
    phi_i = double_power_law_likelihood(loga_i = loga_i, p1 = p1, p2 = p2, 
        log_ab = log_ab, loga_min = loga_min, loga_max = loga_max)
    return np.sum(np.log(phi_i) - np.log(denominator))
    
def double_power_law_likelihood(loga_i, p1, p2, log_ab, loga_min, loga_max):
    '''
    eq 10, divided by the phi0 term (not normalized)
    
    loga_i: array of floats, each of which is the log-separation of a single binary
    p1: float; the power-law exponent at loga < log_ab
    p2: float; the power-law exponent at loga > log_ab
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    ab = 10**log_ab
    a_i = 10**loga_i
    m = loga_i <= log_ab
    phi_i = ab**(p2 - p1)*a_i**(-p2)
    phi_i[m] = a_i[m]**(-p1)
    return phi_i
    
def analytic_integrand_weights_double_power_law(p1, p2, log_ab, beta, theta_0, d_pc, loga_min,
    loga_max):
    '''
    Helper function for computing the integral in the numerator of Eq 9. 
    see math.pdf for details
    
    p1: float; the power-law exponent at loga < log_ab
    p2: float; the power-law exponent at loga > log_ab
    log_ab: float; the log-separation at which there's a break in the power law
    beta: float, characterizes how steeply f_delta(G) falls off (Eq A1)
    theta_0: float, characterizes how steeply f_delta(G) falls off (Eq A1)
    d_pc: array of floats; distance in pc to each binary. 
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    from scipy.special import hyp2f1
    ab, a_min, a_max = 10**log_ab, 10**loga_min, 10**loga_max
    xbreak, xmin, xmax = ab/(d_pc*theta_0), a_min/(d_pc*theta_0), a_max/(d_pc*theta_0)
    gamma1, gamma2 = 1 + beta - p1, 1 + beta - p2
    I1 = 1/gamma1*(xbreak**gamma1 * hyp2f1(1, gamma1/beta, 1 + gamma1/beta, -xbreak**beta) - 
        xmin**gamma1 * hyp2f1(1, gamma1/beta, 1 + gamma1/beta, -xmin**beta))
    I2 = 1/gamma2*(xmax**gamma2 * hyp2f1(1, gamma2/beta, 1 + gamma2/beta, -xmax**beta) - 
        xbreak**gamma2 * hyp2f1(1, gamma2/beta, 1 + gamma2/beta, -xbreak**beta )) 
    I = (d_pc* theta_0)**(1-p1) * I1 + ab**(p2-p1)* (d_pc* theta_0)**(1-p2)*I2
    return I

def get_theta_0_and_beta_this_delta_G(delta_G):
    '''
    Interpolate on the results found empirically from two-point correlation functions
        to get the parameters of the fitting function describing the sensitivity to a 
        companion at a given magnitude different delta_G
    delta_G: array of floats; the difference in G-band magnitude of the two stars. 
    '''
    tmp = np.load('data/empirical_sensitivity_to_delta_G_fitting_func.npz')
    delta_Gs = tmp['delta_G']
    b_vals = tmp['b']
    theta_0s = tmp['theta_0']
    tmp.close()
    these_b = np.interp(delta_G, delta_Gs, b_vals)
    these_theta0 = np.interp(delta_G, delta_Gs, theta_0s)
    return these_b, these_theta0

def theta_is_within_bounds(theta, theta_bounds):
    '''
    helper function for flat priors or restricted likelihoods. 
    theta: array of floats, e.g. [1, 2, 3]
    theta_bounds: array of constraints, e.g. [[0, None], [0, 3], [None, 5] 
    If there's a None in theta_bounds, the prior will be improper. 
    '''
    for i, param in enumerate(theta):
        this_min, this_max  = theta_bounds[i]
        
        if this_min is None:
            this_min = -np.inf
        if this_max is None:
            this_max = np.inf
        if not (this_min <= param < this_max):
            return False
    return True

def get_good_p0_ball(p0, theta_bounds, nwalkers, r = 0.01):
    '''
    Utility function for initializing MCMC walkers. Returns walkers clustered around a 
    point p0 in parameter space, all of which fall within theta_bounds. 
    
    p0: point in parameter space that we think might have a high probability. e.g. [1, 2, 3]
    theta_bounds: the range of parameter space within which we'll allow the walkers to explore;
        we have flat priors within this region. E.g. [[0, 2], [1, 3], [2, 4]]
    nwalkers: int; number of walkers to initialize
    r: float. Decreasing this makes the walkers more and more clustered around p0
    '''
    num_good_p0 = 0

    ball_of_p0 = []
    while num_good_p0 < nwalkers:
        suggested_p0 = p0 + np.array([r*j*np.random.randn() for j in p0])
        suggested_p0_prob = ln_flat_prior(suggested_p0, theta_bounds = theta_bounds)
        if np.isfinite(suggested_p0_prob):
            ball_of_p0.append(suggested_p0)
            num_good_p0 += 1
    return ball_of_p0
    
def ln_flat_prior(theta, theta_bounds):
    '''
    theta: array of parameters, e.g. [1, 2, 3]
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element. e.g.
        [[0, 2], [1, 3], [2, 6]]
    '''
    if theta_is_within_bounds(theta, theta_bounds):
        return 0
    else: 
        return -np.inf
        
def run_mcmc_double_power_law(p0, theta_bounds, loga_i, contrast, d_pc, 
    loga_min, loga_max, nwalkers = 100, n_steps = 100, burn = 100, nthread = 4):
    '''
    Run MCMC to fit separation distribution
    returns a sampler object
    p0: array of parameters, [p1, p2, log_ab]
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    other parameters describe the MCMC
    
    '''
    import emcee
    ndim = len(p0)
    p0_ball = get_good_p0_ball(p0 = p0, theta_bounds = theta_bounds, nwalkers = nwalkers)

    print('initialized walkers... burning in...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior_double_power_law, 
        args=[theta_bounds, loga_i, contrast, d_pc, loga_min, loga_max], threads = nthread)
    pos, prob, state = sampler.run_mcmc(p0_ball, burn)
    sampler.reset()
    print('completed burn in ...')
    for i, result in enumerate(sampler.sample(pos, iterations = n_steps)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i) / n_steps))
    return sampler
    
def ln_posterior_double_power_law(theta, theta_bounds, loga_i, contrast, d_pc, 
    loga_min = -2, loga_max = 4.5):
    '''
    just ln_prior + ln_likelihood
    p0: array of parameters, [p1, p2, log_ab]
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    lnprior = ln_flat_prior(theta = theta, theta_bounds = theta_bounds)
    if np.isfinite(lnprior):
        p1, p2, log_ab = theta
        lnlikelihood = ln_likelihood_double_power_law_with_weights(p1 = p1, p2 = p2, 
            log_ab = log_ab, loga_i = loga_i, contrast = contrast, d_pc = d_pc, 
            loga_min = loga_min, loga_max = loga_max)
    else:
        lnlikelihood = 0 
    return lnprior + lnlikelihood
    
    
#### functions for a single power law ######


def single_power_law_likelihood(loga_i, p, loga_min, loga_max):
    '''
    likelihood for a single power law, divided by the phi0 term (i.e. not normalized)
    
    loga_i: array of floats, each of which is the log-separation of a single binary
    p: float; the power-law exponent
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    a_i = 10**loga_i
    phi_i = a_i**(-p)
    return phi_i


def analytic_integrand_weights_single_power_law(p, beta, theta_0, d_pc, loga_min,
    loga_max):
    '''
    Helper function for computing the integral in the numerator of Eq 9. 
    see math.pdf for details
    
    p: float; the power-law exponent
    beta: float, characterizes how steeply f_delta(G) falls off (Eq A1)
    theta_0: float, characterizes how steeply f_delta(G) falls off (Eq A1)
    d_pc: array of floats; distance in pc to each binary. 
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    from scipy.special import hyp2f1
    a_min, a_max = 10**loga_min, 10**loga_max
    xmin, xmax = a_min/(d_pc*theta_0), a_max/(d_pc*theta_0)
    gamma = 1 + beta - p
    
    I1 = 1/gamma*(xmax**gamma * hyp2f1(1, gamma/beta, 1 + gamma/beta, -xmax**beta) - 
        xmin**gamma * hyp2f1(1, gamma/beta, 1 + gamma/beta, -xmin**beta))
    I = (d_pc* theta_0)**(1-p) * I1 
    return I

def ln_likelihood_single_power_law_with_weights(p, loga_i, contrast, d_pc, 
    loga_min, loga_max):
    '''
    log-likelihood function for a broken power law separation distribution
    
    p: float; the power-law exponent 
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    beta, theta_0 = get_theta_0_and_beta_this_delta_G(delta_G = contrast)
    denominator = analytic_integrand_weights_single_power_law(p = p,  
        beta = beta, theta_0 = theta_0, d_pc = d_pc, 
        loga_min = loga_min, loga_max = loga_max)
    phi_i = single_power_law_likelihood(loga_i = loga_i, p = p, 
        loga_min = loga_min, loga_max = loga_max)
    return np.sum(np.log(phi_i) - np.log(denominator))
    
def run_mcmc_single_power_law(p0, theta_bounds, loga_i, contrast, d_pc, 
    loga_min, loga_max, nwalkers = 100, n_steps = 100, burn = 100, nthread = 4):
    '''
    Run MCMC to fit separation distribution
    returns a sampler object
    p0: array of parameters, but there is just one free parameter [p] in this case
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    other parameters describe the MCMC
    
    '''
    import emcee
    ndim = len(p0)
    p0_ball = get_good_p0_ball(p0 = p0, theta_bounds = theta_bounds, nwalkers = nwalkers)

    print('initialized walkers... burning in...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior_single_power_law, 
        args=[theta_bounds, loga_i, contrast, d_pc, loga_min, loga_max], threads = nthread)
    pos, prob, state = sampler.run_mcmc(p0_ball, burn)
    sampler.reset()
    print('completed burn in ...')
    for i, result in enumerate(sampler.sample(pos, iterations = n_steps)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i) / n_steps))
    return sampler
    
def ln_posterior_single_power_law(theta, theta_bounds, loga_i, contrast, d_pc, 
    loga_min = -2, loga_max = 4.5):
    '''
    just ln_prior + ln_likelihood
    p0: array of parameters, but there is just one free parameter [p] in this case
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    loga_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    loga_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    loga_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    lnprior = ln_flat_prior(theta = theta, theta_bounds = theta_bounds)
    if np.isfinite(lnprior):
        p = theta[0]
        lnlikelihood = ln_likelihood_single_power_law_with_weights(p = p,
            loga_i = loga_i, contrast = contrast, d_pc = d_pc, 
            loga_min = loga_min, loga_max = loga_max)
    else:
        lnlikelihood = 0 
    return lnprior + lnlikelihood
    
#### functions for an Einasto-like distribution ######

def get_weights_einasto_profile(s0, alpha, beta, theta_0, d_pc, logs_min, 
    logs_max, _weights):
    '''
    Helper function for computing the integral in the numerator of Eq 9. 
        
    pass array to C function. This version is ~6 times faster than the equivalent written in Python
    
    s0 and alpha parameterize the Einasto profile
    beta and theta0 are arrays parameterizing the sensitivity to a companion, for each binary
    d_pc is the distance array for all the binaries
    logs_min and logs_max are the bounds of integration
    _weights is a C object corresponding to the function get_weights_einasto() in integrands.c
    '''
    import ctypes
    beta_double = beta.astype(np.double)
    theta0_double = theta_0.astype(np.double)
    d_pc_double = d_pc.astype(np.double)
    
    weight_array = np.empty(len(beta), dtype = np.double)
    
    _weights(ctypes.c_int(len(beta)), ctypes.c_double(s0), ctypes.c_double(alpha),
        ctypes.c_void_p(weight_array.ctypes.data), ctypes.c_void_p(beta_double.ctypes.data),
        ctypes.c_void_p(theta0_double.ctypes.data), ctypes.c_void_p(d_pc_double.ctypes.data), 
        ctypes.c_double(logs_min), ctypes.c_double(logs_max))

    return weight_array
    
def read_in_C_function_weights(model = 'Einasto'):
    '''
    This returns a function to be passed to get_weights_einasto_profile(), which 
    does the integral in C. 
    '''
    import ctypes
    _weights = ctypes.CDLL('integrands.so')
    if model == 'Einasto':
        return _weights.get_weights_einasto
    else:
        raise ValueError('unknown model')
        
def einasto_likelihood(logs_i, s0, alpha, logs_min, logs_max):
    '''
    likelihood for an einasto distribution function, divided by the phi0 term (i.e. not normalized)
    
    logs_i: array of floats, each of which is the log-separation of a single binary
    s0 and alpha: floats; they parameterize the Einasto profile
    logs_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    logs_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    s_i = 10**logs_i
    phi_i = np.exp(-(s_i/s0)**(alpha))
    return phi_i

def ln_likelihood_einasto_with_weights(s0, alpha, logs_i, contrast, d_pc, 
    logs_min, logs_max):
    '''
    log-likelihood function for an einasto distribution function.
    
    s0 and alpha: floats; they parameterize the Einasto profile
    logs_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    logs_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    logs_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    beta, theta_0 = get_theta_0_and_beta_this_delta_G(delta_G = contrast)
    _weights = read_in_C_function_weights(model = 'Einasto')
    denominator = get_weights_einasto_profile(s0 = s0, alpha = alpha, beta = beta,
        theta_0 = theta_0, d_pc = d_pc, logs_min = logs_min, 
        logs_max = logs_max, _weights = _weights)
    phi_i = einasto_likelihood(logs_i = logs_i, s0 = s0, alpha = alpha, 
        logs_min = logs_min, logs_max = logs_max)
    return np.sum(np.log(phi_i) - np.log(denominator))

    
def ln_posterior_einasto(theta, theta_bounds, logs_i, contrast, d_pc, 
    logs_min = -2, logs_max = 4.5):
    '''
    just ln_prior + ln_likelihood
    p0: array of parameters, [logs0, alpha] in this case
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    logs_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    logs_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    logs_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    '''
    lnprior = ln_flat_prior(theta = theta, theta_bounds = theta_bounds)
    if np.isfinite(lnprior):
        logs0, alpha = theta
        lnlikelihood = ln_likelihood_einasto_with_weights(s0 = 10**logs0, alpha = alpha,
            logs_i = logs_i, contrast = contrast, d_pc = d_pc, 
            logs_min = logs_min, logs_max = logs_max)
    else:
        lnlikelihood = 0 
    return lnprior + lnlikelihood

def run_mcmc_einasto(p0, theta_bounds, logs_i, contrast, d_pc, 
    logs_min, logs_max, nwalkers = 100, n_steps = 100, burn = 100, nthread = 4):
    '''
    Run MCMC to fit separation distribution
    returns a sampler object
    p0: array of parameters, [logs0, alpha] in this case
    theta_bounds: array of the same length, but with a list of length
        two (lower and upper bounds) at each element.
    logs_i: array of floats, each of which is the log-separation of a single binary
    contrast: array of magnitude differences for each binary
    d_pc: array if distances to each binary
    logs_min: float, minimum log-separation of the distribution from which separations 
        are drawn
    logs_max: float, maximum log-separation of the distribution from which separations 
        are drawn    
    other parameters describe the MCMC
    
    '''
    import emcee
    ndim = len(p0)
    p0_ball = get_good_p0_ball(p0 = p0, theta_bounds = theta_bounds, nwalkers = nwalkers)

    print('initialized walkers... burning in...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior_einasto, 
        args=[theta_bounds, logs_i, contrast, d_pc, logs_min, logs_max], threads = nthread)
    pos, prob, state = sampler.run_mcmc(p0_ball, burn)
    sampler.reset()
    print('completed burn in ...')
    for i, result in enumerate(sampler.sample(pos, iterations = n_steps)):
        if (i+1) % 10 == 0:
            print("{0:5.1%}".format(float(i) / n_steps))
    return sampler