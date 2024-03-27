import os
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import norm
import yaml
from datetime import datetime
from multiprocessing import Manager
import argparse

from utils.misc_utils import *
from utils.num_integration_utils import trapezoid_integration1D, expectation_2d
from utils.consts import VALID_LABEL_METHODS, VALID_INTEGRATION_METHODS


def get_misc_error(alpha_perp, alpha_0, data_theta_norm, f_model, bound=5, N=5000):
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
        misc_error: float, the expectation over standard normal G
    """

    def f_(G, alpha_perp, alpha_0, data_theta_norm):
        q = alpha_0 / alpha_perp
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
    """
    Objective functon for minimization over variable u in Eq P.4.
    """
    return 0.5 * mu * (u - k) ** 2 + np.log(1 + np.exp(u + l)) - y * (u + l)


def compute_current_split_fraction(c, pi, beta_0, n=5000, bound=5):
    def f(G_star):
        return np.minimum(c * pi(beta_0 * G_star), 1)

    # current_split_fraction = expectation_gauss_hermite_1d(f, n)
    current_split_fraction = trapezoid_integration1D(f, n, bound)

    return current_split_fraction


def get_c_opt_expectation(split_pct, beta_0, pi, min_c=0, max_c=1e10):
    f = lambda c: compute_current_split_fraction(c, pi, beta_0) - split_pct / 100
    c_opt = scipy.optimize.bisect(f, min_c, max_c)
    return c_opt


def expectation_fun(G_star, G, alpha_perp, alpha_0, mu, data_theta_norm, beta_0, pi, f_model):
    assert mu > 0
    f_modelTerm = f_model(data_theta_norm * G_star)
    piTerm = pi(beta_0 * G_star)

    # Intermediate variables for to optimize over u
    k = alpha_perp * G
    l = alpha_0 * G_star

    u_star1 = get_u_star(1, l, k, mu)
    u_star0 = get_u_star(0, l, k, mu)

    obj_Y1 = get_obj_u(u_star1, 1, l, k, mu)
    obj_Y0 = get_obj_u(u_star0, 0, l, k, mu)

    res = f_modelTerm * piTerm * obj_Y1 + (1 - f_modelTerm) * piTerm * obj_Y0

    return res


def compute_exp_composite_gaussian_quadrature(alpha_perp, alpha_0, mu, data_theta_norm, beta_0, pi, f_model,
                                              n_quadrature, method, trapz_limit):
    """
    Computes numerical expectation using appropriate quadrature method.
    """

    def f(G_star, G):
        return expectation_fun(G_star, G, alpha_perp, alpha_0, mu, data_theta_norm, beta_0, pi, f_model)

    # Approximate the expectation using composite Gaussian quadrature
    expectation = expectation_2d(f, n_quadrature, method, trapz_limit)

    return expectation


def obj_mu(mu, alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model, n_quadrature, method, trapz_limit):
    expect = compute_exp_composite_gaussian_quadrature(alpha_perp, alpha_0, mu, data_theta_norm, beta_0, pi,
                                                       f_model, n_quadrature, method, trapz_limit)
    return -0.5 * (1 / delta0) * mu * alpha_perp ** 2 + expect


def get_mu_star(alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model, n_quadrature, trapz_limit,
                method, mu_init, tol=1e-15, lower_bound=1e-15, upper_bound=None):
    """
    Solves the optimization problem for mu in Eq 5.4.
    mu_star(\alpha) = argmax_mu L(\alpha, \mu)
    """

    def obj_mu1d(mu):
        return -obj_mu(mu, alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model, n_quadrature, method,
                       trapz_limit)

    result = minimize(obj_mu1d, mu_init, bounds=[(lower_bound, upper_bound)], tol=tol)
    mu_star = result.x.item()

    return mu_star


def obj_alpha(mu_init, alpha_perp, alpha_0, data_theta_norm, beta_0, pi, f_model, n_quadrature, trapz_limit,
              method, delta0, lambd, tol=1e-15):
    """
    obj_alpha [Eq 5.4]:
    L(\alpha, \mu_star(\alpha)) where \mu_star(\alpha) is the solution of inner-maximization problem
    given a value of \alpha
    """

    # First solve for mu_star(\alpha) = argmax_mu L(\alpha, \mu) numerically
    mu_star = get_mu_star(alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model, n_quadrature,
                          trapz_limit, method, mu_init, tol=tol)

    mu_objective = obj_mu(mu_star, alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model,
                          n_quadrature, method, trapz_limit)

    return mu_objective + (lambd / 2) * (alpha_perp ** 2 + alpha_0 ** 2)


@timer
def get_alpha_star(mu_init, alpha_perp_init, alpha_0_init, data_theta_norm, beta_0, pi, f_model, n_quadrature,
                   trapz_limit, method, delta0, lambd, lower_bound=1e-15, upper_bound=None):
    """
    Solves the optimization problem for alpha_perp, alpha_0, mu in Eq 5.4.
    Returns: alpha_star = argmin_alpha L(\alpha, \mu_star(\alpha))

    obj_alpha:
    L(\alpha, \mu_star(\alpha)) where \mu_star(\alpha) is the solution of inner-maximization problem
    given a value of \alpha
    """

    # need a function of only variables being optimized using scipy.optimize.minimize
    def obj_alpha1d(alpha):
        alpha_perp, alpha_0 = alpha

        return obj_alpha(mu_init, alpha_perp, alpha_0, data_theta_norm, beta_0, pi, f_model, n_quadrature,
                         trapz_limit, method, delta0, lambd)

    x0 = np.asarray([alpha_perp_init, alpha_0_init])

    alpha_perp_bound, alpha_0_bound = (lower_bound, upper_bound), (lower_bound, upper_bound)

    result = minimize(obj_alpha1d, x0, bounds=[alpha_perp_bound, alpha_0_bound])

    alpha_star = result.x
    return alpha_star


def get_pi(alpha_exp, split_pct, beta_0):
    """
    Returns the subselector function \pi as per Eq 6.1.

    Args:
        alpha_exp: float, exponent for the subselector function (\alpha in Eq 6.1).
        split_pct: float, percentage of split (100 * gamma, where gamma is the fraction of data subsampled)
        beta_0: float, surrogate param: norm of the data theta for the perfect surrogate case

    Returns:
        pi: function, the final selection probability function \pi as per Eq 6.1
    """
    # un-normalized sampling scores per datapoint as per Eq 6.1
    # z -> logits of the model <- <theta_surr, x>
    # note: p(1-p) is the hardness score = sigmoid_prime(z)
    # clipping ensures logits are within reasonable range for numerical stability
    unnormalized_pi = lambda z: sigmoid_prime(clip_weights(z)) ** alpha_exp

    # get the right normalization constant so that the number of samples in the split is same as split_pct * N_train
    c_opt = get_c_opt_expectation(split_pct, beta_0, unnormalized_pi)

    # the final selection probability function \pi as per Eq 6.1
    pi = lambda z: min(c_opt * sigmoid_prime(clip_weights(z)) ** alpha_exp, 1)

    return pi


def solve_exact(mu_init, alpha_perp_init, alpha_0_init, split_pct, lambd, data_theta_norm, alpha_exp, delta0,
                label_method, label_method_a, label_method_b, n_quadrature, trapz_limit, method):
    """
    Solves the optimization problem P.4 in the case of perfect surrogate, i.e.
    beta_s=0 (Eq 5.2), alpha_s=0 (Eq L.9).
    In this case, we have only two parameters over which we minimize (2D): {alpha_0, alpha_perp} in the saddle point
    calculation in Eq 5.4.

    All names follow conventions in the paper.

    Args:
        mu_init: float, initial value for mu
        alpha_perp_init: float, initial value for alpha_perp
        alpha_0_init: float, initial value for alpha_0

        split_pct: float, percentage of split (100 * gamma, where gamma is the fraction of data subsampled)
        lambd: float, regularization parameter
        data_theta_norm: float, norm of the data theta

        alpha_exp: float, exponent for the subselector function (\alpha in Eq 6.1). Positive values imply more
                        weight for hard examples, negative values imply more weight for easy examples.
        delta0: float, N_train / data_p

        label_method: str, method for generating labels (logistic or misspecified), See Section 6.
        label_method_a: float, parameter for label generation under misspecification
        label_method_b: float, parameter for label generation under misspecification

        method: str, method for numerical integration
        n_quadrature: int, number of quadrature points
        trapz_limit: float, limit for trapezoidal rule
    """
    # for perfect surrogate, beta_0 = data_theta_norm (see Eq 5.2)
    beta_0 = data_theta_norm

    # get the selection probability function \pi as per Eq 6.1
    # it takes as input the logits from the model and returns the probability of selection
    pi = get_pi(alpha_exp, split_pct, beta_0)

    # true-probabilities functions, well- or mis-specified probabilities based on the given method
    # f_model(z) = P(y=1|z)
    f_model = f_model_wrapper(label_method, label_method_a, label_method_b)

    # Numerically solve the optimization problem for alpha_perp, alpha_0, mu
    alpha = get_alpha_star(mu_init, alpha_perp_init, alpha_0_init, data_theta_norm, beta_0, pi, f_model,
                           n_quadrature, trapz_limit, method, delta0, lambd)
    alpha_perp, alpha_0 = alpha
    mu = get_mu_star(alpha_perp, alpha_0, data_theta_norm, beta_0, delta0, pi, f_model, n_quadrature,
                     trapz_limit, method, mu_init, tol=1e-15)
    misc_error = get_misc_error(alpha_perp, alpha_0, data_theta_norm, f_model, bound=5, N=5000)
    return alpha_perp, alpha_0, mu, misc_error


def main(param_grid, opt_params, save_path, queue):
    # setup local worker logger
    worker_configurer(queue)
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.info(f"Starting worker-{os.getpid()}")

    # extract params from param_dict
    data_p = param_grid['data_p']
    data_theta_norm = param_grid['data_theta_norm']
    lambd = param_grid['lambd']
    alpha_exp = param_grid['alpha_exp']
    split_pct = param_grid['split_pct']
    N_train = param_grid['N_train']
    label_method = param_grid['label_method']
    label_method_a = param_grid['label_method_a']
    label_method_b = param_grid['label_method_b']

    # solve for optimization variables
    delta0 = N_train / data_p

    # get opt params
    alpha_perp_init, alpha_0_init = opt_params['alpha_perp_init'], opt_params['alpha_0_init']
    mu_init = opt_params['mu_init']
    method = opt_params['int_method']
    n_quadrature = opt_params['n_quad']
    trapz_limit = opt_params['trapz_limit']

    exp_name = (
        f'theoryLR2D_dataN-{N_train}_dataP-{data_p}_dataThetaNorm-{data_theta_norm}_lambd-{lambd}_'
        f'alphaExp-{alpha_exp}_splitPCT-{split_pct}_NQ-{n_quadrature}_method-{method}_'
        f'trapzLimit-{trapz_limit}_label-{label_method}_A-{label_method_a}_B-{label_method_b}.csv'
    )

    logger.info(exp_name)

    # solves the optimization problem
    alpha_perp, alpha_0, mu, misc_error = solve_exact(mu_init, alpha_perp_init, alpha_0_init, split_pct, lambd,
                                                      data_theta_norm,
                                                      alpha_exp, delta0,
                                                      label_method, label_method_a, label_method_b,
                                                      n_quadrature, trapz_limit, method)

    # extract and save results
    res = pd.DataFrame(
        {'alpha_perp': [alpha_perp],
         'alpha_0': [alpha_0],
         'mu': [mu],
         'MiscError': [misc_error],
         'dataThetaNorm': [data_theta_norm],
         'lambd': [lambd],
         'split_pct': [split_pct],
         'alpha_exp': [alpha_exp],
         'dataN': [N_train],
         'dataP': [data_p],
         'n_quadrature': [n_quadrature],
         'method': [method],
         'trapzLimit': [trapz_limit],
         'label_method': [label_method],
         'label_method_a': [label_method_a],
         'label_method_b': [label_method_b]
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
    parser = argparse.ArgumentParser(description='Run perfect surrogate theory experiment')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    config_file = parser.parse_args().config

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    validate_config(config)

    param_grid = config['param_grid']

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
