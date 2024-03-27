import os
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import norm
import yaml
from datetime import datetime
from multiprocessing import Manager
import argparse

from utils.misc_utils import *
from utils.num_integration_utils import (trapezoid_integration1D,
                                         expectation_trapez_2d,
                                         expectation_3d)
from utils.consts import VALID_LABEL_METHODS, VALID_INTEGRATION_METHODS


def get_misc_error(alpha_perp, alpha_0, alpha_s, data_theta_norm, f_model, bound=5, N=5000):
    """
    Special case of the error in Eq 5.5 for the perfect surrogate case i.e. alpha_surr = 0.
    See Appendix P, "Some useful formulas for binary classification", Equation P.5.

    Calculates the expectation over standard normal G.

    Args:
        alpha_perp: float, alpha_perp
        alpha_0: float, alpha_0
        data_theta_norm: float, norm of the data theta
        f_model: function, label function. Logistic or misspecified.
        bound: float, bound for trapezoidal rule
        N: int, number of sub-intervals

    Returns:
        misc_error: float, the expectation over standard normal G.
    """

    def f_(G, alpha_perp, alpha_0, data_theta_norm):
        q = alpha_0 / np.sqrt(alpha_s ** 2 + alpha_perp ** 2)

        return (2 * f_model(data_theta_norm * G) - 1) * (2 * norm.cdf(q * G) - 1)

    f = lambda G: f_(G, alpha_perp, alpha_0, data_theta_norm)

    expectation_G = trapezoid_integration1D(f, N, bound)

    misc_error = 0.5 - 0.5 * expectation_G

    return misc_error


def get_u_star(y, l, k, mu):
    assert (mu > 0)
    sol = scipy.optimize.root(lambda u: mu * (u - k) - y + 1 / (1 + np.exp(-(u + l))), x0=1)
    u_min = sol.x
    return u_min.item()


def get_obj_u(u, y, l, k, mu):
    return 0.5 * mu * (u - k) ** 2 + np.log(1 + np.exp(u + l)) - y * (u + l)


def compute_current_split_fraction(c, pi, beta_0, beta_perp, n=100, trapz_limit=5):
    def f(G_star, G_s):
        return np.minimum(c * pi(beta_0 * G_star + beta_perp * G_s), 1)

    current_split_fraction = expectation_trapez_2d(f, n, trapz_limit)
    return current_split_fraction


def get_c_opt_expectation(split_pct, beta_0, beta_perp, pi, min_c=0, max_c=1e10):
    f = lambda c: compute_current_split_fraction(c, pi, beta_0, beta_perp) - split_pct / 100
    c_opt = scipy.optimize.bisect(f, min_c, max_c)
    return c_opt


def expectation_fun(G_star, G, G_s, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0, beta_perp, pi, f_model):
    assert mu > 0
    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star + beta_perp * G_s)

    # Intermediate variables for to optimize over u
    k = alpha_perp * G
    l = alpha_0 * G_star + alpha_s * G_s

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = get_obj_u(u_star1, 1, l, k, mu)
    obj_Y0 = get_obj_u(u_star0, 0, l, k, mu)

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def expectation_fun_gradient_alpha_perp(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0,
                                        beta_perp, pi, f_model):
    assert mu > 0, f"mu should be positive, but got {mu}"
    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star + beta_perp * G_s)

    # Intermediate variables for to optimize over u
    k = alpha_perp * G
    l = alpha_0 * G_star + alpha_s * G_s

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = -G * mu * (u_star1 - alpha_perp * G)
    obj_Y0 = -G * mu * (u_star0 - alpha_perp * G)

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def expectation_fun_gradient_alpha_0(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0,
                                     beta_perp, pi, f_model):
    assert mu > 0, f"mu should be positive, but got {mu}"
    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star + beta_perp * G_s)

    # Intermediate variables for to optimize over u
    k = alpha_perp * G
    l = alpha_0 * G_star + alpha_s * G_s

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = -G_star * np.exp(-(u_star1 + alpha_0 * G_star + alpha_s * G_s)) / (
            1 + np.exp(-(u_star1 + alpha_0 * G_star + alpha_s * G_s)))
    obj_Y0 = G_star * np.exp(u_star0 + alpha_0 * G_star + alpha_s * G_s) / (
            1 + np.exp(u_star0 + alpha_0 * G_star + alpha_s * G_s))

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def expectation_fun_gradient_alpha_s(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0,
                                     beta_perp, pi, f_model):
    assert mu > 0, f"mu should be positive, but got {mu}"
    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star + beta_perp * G_s)

    # Intermediate variables for to optimize over u
    k = alpha_perp * G
    l = alpha_0 * G_star + alpha_s * G_s

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = -G_s * np.exp(-(u_star1 + alpha_0 * G_star + alpha_s * G_s)) / (
            1 + np.exp(-(u_star1 + alpha_0 * G_star + alpha_s * G_s)))
    obj_Y0 = G_s * np.exp(u_star0 + alpha_0 * G_star + alpha_s * G_s) / (
            1 + np.exp(u_star0 + alpha_0 * G_star + alpha_s * G_s))

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def compute_alpha_gradient(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, pi,
                           f_model, lambd, delta0, mu, n_quadrature, method, trapz_limit):
    def f_star(G_star, G_s, G):
        return expectation_fun_gradient_alpha_0(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm,
                                                beta_0, beta_perp, pi, f_model)

    def f_s(G_star, G_s, G):
        return expectation_fun_gradient_alpha_s(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm,
                                                beta_0, beta_perp, pi, f_model)

    def f_perp(G_star, G_s, G):
        return expectation_fun_gradient_alpha_perp(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm,
                                                   beta_0, beta_perp, pi, f_model)

    expectation_perp = expectation_3d(f_perp, n_quadrature, method, trapz_limit)
    expectation_star = expectation_3d(f_star, n_quadrature, method, trapz_limit)
    expectation_s = expectation_3d(f_s, n_quadrature, method, trapz_limit)

    grad_perp = -(1 / delta0) * mu * alpha_perp + lambd * alpha_perp + expectation_perp
    grad_star = lambd * alpha_0 + expectation_star
    grad_s = lambd * alpha_s + expectation_s
    grad = [grad_perp.item(), grad_star, grad_s]

    return grad


def expectation_fun_gradient_mu(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0, beta_perp,
                                pi, f_model):
    assert mu > 0, f"mu should be positive, but got {mu}"

    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star + beta_perp * G_s)

    k = alpha_perp * G
    l = alpha_0 * G_star + alpha_s * G_s

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = 0.5 * (u_star1 - alpha_perp * G) ** 2
    obj_Y0 = 0.5 * (u_star0 - alpha_perp * G) ** 2

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def compute_mu_gradient(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, pi, f_model, delta0, mu,
                        n_quadrature, method, trapz_limit):
    """
    Computes the explicit gradient of the objective function with respect to mu.
    """

    def f_mu(G_star, G_s, G):
        return expectation_fun_gradient_mu(G_star, G_s, G, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0,
                                           beta_perp, pi, f_model)

    expectation = expectation_3d(f_mu, n_quadrature, method, trapz_limit)

    grad = -(1 / delta0) * alpha_perp ** 2 / 2 + expectation

    return grad


def compute_exp_composite_gaussian_quadrature(alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0, beta_perp,
                                              pi, f_model, n_quadrature, method, trapz_limit):
    """
    Computes numerical expectation using appropriate quadrature method.
    """

    def f(G_star, G, G_s):
        return expectation_fun(G_star, G, G_s, alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0, beta_perp,
                               pi, f_model)

    # Approximate the expectation using composite Gaussian quadrature
    expectation = expectation_3d(f, n_quadrature, method, trapz_limit)

    return expectation


def ObjMu(mu, alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, delta0, pi, f_model, n_quadrature,
          method, trapz_limit):
    expect = compute_exp_composite_gaussian_quadrature(alpha_perp, alpha_0, alpha_s, mu, data_theta_norm, beta_0,
                                                       beta_perp, pi, f_model, n_quadrature, method, trapz_limit)
    return -0.5 * (1 / delta0) * mu * alpha_perp ** 2 + expect


@timer
def get_mu_star(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, delta0, pi, f_model,
                n_quadrature, trapz_limit, method, mu_init, tol_mu, use_grad_mu, min_bound=1e-15, max_bound=None):
    """
    Solves the optimization problem for mu in Eq 5.4.
    mu_star(\alpha) = argmax_mu L(\alpha, \mu).

    If use_grad_mu is True, uses the explicit gradient of the objective function to solve the optimization problem.
    """

    def obj_mu(mu):
        return ObjMu(mu, alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, delta0, pi, f_model,
                     n_quadrature, method, trapz_limit)

    bounds = [(min_bound, max_bound)]

    if use_grad_mu:
        grad_mu = lambda mu: -compute_mu_gradient(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp,
                                                  pi, f_model, delta0, mu, n_quadrature, method, trapz_limit)
        result = minimize(lambda mu: -obj_mu(mu), mu_init, bounds=bounds, jac=grad_mu, tol=tol_mu)
    else:
        result = minimize(lambda mu: -obj_mu(mu), mu_init, bounds=bounds, tol=tol_mu)

    mu_star = result.x.item()
    return mu_star


def get_pi(alpha_exp, split_pct, beta_0, beta_perp):
    """
    Returns the subselector function \pi as per Eq 6.1.

    Args:
        alpha_exp: float, exponent for the subselector function (\alpha in Eq 6.1).
        split_pct: float, percentage of split (100 * gamma, where gamma is the fraction of data subsampled)
        beta_0: float, surrogate parameter (Eq 5.2)
        beta_perp: float, surrogate parameter (Eq 5.2)

    Returns:
        pi: function, the subselector function \pi as per Eq 6.1.
    """
    # un-normalized sampling scores per datapoint as per Eq 6.1
    # z -> logits of the model <- <theta_surr, x>
    # note: p(1-p) is the hardness score = sigmoid_prime(z)
    # clipping ensures logits are within reasonable range for numerical stability
    unnormalized_pi = lambda z: sigmoid_prime(clip_weights(z)) ** alpha_exp

    # get the right normalization constant so that the number of samples in the split is same as split_pct * N_train
    c_opt = get_c_opt_expectation(split_pct, beta_0, beta_perp, unnormalized_pi)

    # the final selection probability function \pi as per Eq 6.1
    pi = lambda z: min(c_opt * sigmoid_prime(clip_weights(z)) ** alpha_exp, 1)

    return pi


@timer
def solve_gd_alpha(mu_init, alpha_perp_init, alpha_0_init, alpha_s_init,
                   split_pct, lambd, beta_0, beta_perp, data_theta_norm, alpha_exp, delta0, label_method,
                   label_method_a, label_method_b, n_quadrature, trapz_limit, method, lr_alpha,
                   max_iter, tol_alpha, tol_mu, use_grad_mu, decay):
    """
    Solves the optimization problem P.4 in the general case with surrogates using gradient descent.
    In this case, we have three parameters over which we minimize (3D): {alpha_0, alpha_perp, alpha_s} in the saddle
    point calculation in Eq 5.4.
    
    All names follow conventions in the paper.
    
    Args:
        mu_init: float, initial value for mu
        alpha_perp_init: float, initial value for alpha_perp
        alpha_0_init: float, initial value for alpha_0
        alpha_s_init: float, initial value for alpha_s
    
        split_pct: float, percentage of data to be used for training
        lambd: float, regularization parameter
        
        beta_0: float, surrogate parameter (Eq 5.2)
        beta_perp: float, surrogate parameter (Eq 5.2)
        data_theta_norm: float, the norm of the data
        
        alpha_exp: float, exponent for the subselector function (\alpha in Eq 6.1). Positive values imply more
                            weight for hard examples, negative values imply more weight for easy examples.
        delta0: float, N_train / data_p

        label_method: str, method for generating labels (logistic or misspecified), See Section 6.
        label_method_a: float, parameter for label generation under misspecification
        label_method_b: float, parameter for label generation under misspecification
        
        method: str, method for numerical integration
        n_quadrature: int, number of quadrature points
        trapz_limit: float, limit for trapezoidal rule

        lr_alpha: float, learning rate for gradient descent during min over alpha
        max_iter: int, maximum number of iterations
        tol_alpha: float, tolerance for convergence of alpha
        tol_mu: float, tolerance for convergence of mu

        use_grad_mu: bool, whether to use explicit gradient for mu or implicitly calculate it using minimizer
        decay: str, type of learning rate decay (None, 'lin', 'quad')
    """
    assert decay in [False, 'lin', 'quad'], "Invalid decay type"

    # get the selection probability function \pi as per Eq 6.1
    # it takes as input the logits from the model and returns the probability of selection
    pi = get_pi(alpha_exp, split_pct, beta_0, beta_perp)

    # true-probabilities functions, well- or mis-specified probabilities based on the given method
    # f_model(z) = P(y=1|z)
    f_model = f_model_wrapper(label_method, label_method_a, label_method_b)

    # Numerically solve the optimization problem for alpha_perp, alpha_0, alpha_s, mu
    alpha_perp, alpha_0, alpha_s = alpha_perp_init, alpha_0_init, alpha_s_init
    mu = mu_init
    lr_alpha0 = lr_alpha

    # alternate gradient descent for alpha and ascent for mu to find the saddle point of lagrangian
    for i in range(max_iter):
        if decay == 'lin':
            lr_alpha = lr_alpha0 / (i + 1)
        elif decay == 'quad':
            lr_alpha = lr_alpha0 / ((i + 1) ** 2)

        # find optimal mu given alpha
        mu = get_mu_star(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, delta0, pi, f_model,
                         n_quadrature, trapz_limit, method, mu, tol_mu, use_grad_mu)

        # find optimal alpha given mu
        grad_alpha = compute_alpha_gradient(alpha_perp, alpha_0, alpha_s, data_theta_norm, beta_0, beta_perp, pi,
                                            f_model, lambd, delta0, mu, n_quadrature, method, trapz_limit)

        alpha_perp -= lr_alpha * grad_alpha[0]
        alpha_0 -= lr_alpha * grad_alpha[1]
        alpha_s -= lr_alpha * grad_alpha[2]

        if np.max(np.abs(np.array(grad_alpha))) < tol_alpha:
            print(f'CONVERGED AFTER {i} ITER')
            misc_error = get_misc_error(alpha_perp, alpha_0, alpha_s, data_theta_norm, f_model, bound=5, N=5000)
            return alpha_perp, alpha_0, alpha_s, mu, misc_error, 'solve_gd_alpha', i

    print(f'DIVERGED')
    return None, None, None, None, None, None, None


def main(param_grid, opt_params, save_path, queue):
    # setup local worker logger
    worker_configurer(queue)
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.info(f"Starting worker-{os.getpid()}")

    # extract params from param_dict
    data_p = param_grid['data_p']
    data_theta_norm = param_grid['data_theta_norm']
    beta_0 = param_grid['beta_0']
    beta_perp = param_grid['beta_perp']
    lambd = param_grid['lambd']
    alpha_exp = param_grid['alpha_exp']
    split_pct = param_grid['split_pct']
    N_train = param_grid['N_train']
    label_method = param_grid['label_method']
    label_method_a = param_grid['label_method_a']
    label_method_b = param_grid['label_method_b']

    delta0 = N_train / data_p

    # get opt params
    use_grad_mu = opt_params['use_grad_mu']
    method = opt_params['int_method']
    n_quadrature = opt_params['n_quad']
    trapz_limit = opt_params['trapz_limit']

    tol_alpha = float(opt_params['tol_alpha'])
    tol_mu = float(opt_params['tol_mu'])

    max_iter = opt_params['max_iter']

    # these parameters are set based on our observations on the convergence behavior (stability, time to converge etc.)
    # you might have to play around with these values to get the best convergence behavior.
    if lambd == 0.1:
        lr_alpha = 5
        decay = False
    elif lambd == 0.01:
        decay = 'lin'
        lr_alpha = 100
    elif lambd == 0.001:
        lr_alpha = 20
        decay = False
    else:
        lr_alpha = 10
        decay = False

    exp_name = (
        f'theoryLR3D_dataN-{N_train}_dataP-{data_p}_dataThetaNorm-{data_theta_norm}_lambd-{lambd}_'
        f'alphaExp-{alpha_exp}_splitPCT-{split_pct}_beta0-{beta_0}_betaPerp-{beta_perp}_NQ-{n_quadrature}_'
        f'tolAlpha-{tol_alpha}_tolMu-{tol_mu}_method-{method}_trapzLimit-{trapz_limit}_'
        f'label-{label_method}_A-{label_method_a}_B-{label_method_b}_maxIter-{max_iter}.csv'
    )
    logger.info(exp_name)

    # solves the optimization problem
    alpha_perp_init, alpha_0_init, alpha_s_init, mu_init = (opt_params['alpha_perp_init'], opt_params['alpha_0_init'],
                                                            opt_params['alpha_s_init'], opt_params['mu_init'])

    alpha_perp, alpha_0, alpha_s, mu, misc_error, method_solve, it = solve_gd_alpha(mu_init, alpha_perp_init,
                                                                                    alpha_0_init, alpha_s_init,
                                                                                    split_pct, lambd, beta_0, beta_perp,
                                                                                    data_theta_norm, alpha_exp,
                                                                                    delta0, label_method,
                                                                                    label_method_a, label_method_b,
                                                                                    n_quadrature, trapz_limit, method,
                                                                                    lr_alpha, max_iter, tol_alpha,
                                                                                    tol_mu, use_grad_mu, decay)

    # extract and save results
    res = pd.DataFrame(
        {'alpha_perp': [alpha_perp],
         'alpha_0': [alpha_0],
         'alpha_s': [alpha_s],
         'mu': [mu],
         'misc_error': [misc_error],
         'dataThetaNorm': [data_theta_norm],
         'lambd': [lambd],
         'split_pct': [split_pct],
         'alphaExp': [alpha_exp],
         'dataN': [N_train],
         'dataP': [data_p],
         'n_quadrature': [n_quadrature],
         'method': [method],
         'trapzLimit': [trapz_limit],
         'label_method': [label_method],
         'label_method_a': [label_method_a],
         'label_method_b': [label_method_b],
         'method_solve': [method_solve],
         'lr': [lr_alpha],
         'tol_mu': [tol_mu],
         'tol_alpha': [tol_alpha],
         'beta_perp': [beta_perp],
         'beta_0': [beta_0],
         'iter': [it],
         'decay': [decay],
         'max_iter': [max_iter],
         }
    )

    logger.info(f"alpha_perp = {alpha_perp}, alpha_0 = {alpha_0}, mu = {mu}, MiscError = {misc_error}")
    res.to_csv(os.path.join(save_path, exp_name), index=False)

    logger.info(f"Finished worker-{os.getpid()}")


def validate_config(config):
    # assert all param_grid values are lists as we use itertools to create combinations over all possible values
    param_grid = config['param_grid']
    for key in param_grid:
        assert type(param_grid[key]) == list

    for label_method in param_grid['label_method']:
        assert label_method in VALID_LABEL_METHODS, f"Invalid label method: {param_grid['label_method']}; " \
                                                    f"Valid methods: {VALID_LABEL_METHODS}"

    opt_params = config['opt_params']
    assert opt_params['int_method'] in VALID_INTEGRATION_METHODS, f"Invalid integration method: " \
                                                                  f"{opt_params['int_method']}; " \
                                                                  f"Valid methods: {VALID_INTEGRATION_METHODS}"


if __name__ == '__main__':
    # read config file
    parser = argparse.ArgumentParser(description='Run imperfect surrogate theory experiment')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    config_file = parser.parse_args().config

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    validate_config(config)

    param_grid = config['param_grid']
    # assert all param_grid values are lists as we use itertools to create combinations over all possible values
    for key in param_grid:
        assert type(param_grid[key]) == list

    opt_params = config['opt_params']
    n_jobs = config['n_jobs']
    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True)

    # create logger
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path, log_file_name = config['logs']['path'], f"{dt}_{config['logs']['name']}"
    os.makedirs(log_path, exist_ok=True)

    logfile = os.path.join(log_path, log_file_name)
    manager = Manager()
    queue = manager.Queue()
    listener = setup_logger(queue, logfile)

    # run parameter grid
    try:
        run_grid_search(main, param_grid, opt_params, save_path, n_jobs, queue)
    finally:
        listener.stop()
        print("Log file saved at:", logfile)
        print("Done!")
