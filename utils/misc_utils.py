import numpy as np
from scipy.optimize import minimize
import scipy
import time
from functools import wraps
from concurrent.futures import ProcessPoolExecutor
import itertools
import logging
from logging import StreamHandler, FileHandler, Formatter
from logging.handlers import QueueHandler, QueueListener


def timer(func):
    @wraps(func)
    def wrapper(*args):
        start_time = time.time()
        retval = func(*args)
        print("the function ends in ", time.time() - start_time, "secs")
        return retval

    return wrapper


def update_exp_params(results_dict, results_dict_experiment):
    for key in results_dict_experiment:
        results_dict[key] += results_dict_experiment[key]

def update_params(results_dict, results_dict_experiment, N):
    for key in results_dict_experiment:
        results_dict[key] += [results_dict_experiment[key]] * N



def f_model_wrapper(method, a=None, b=None,
                    thr=0.5, zeta=0.7):
    """
    Creates the well- and mis-specified probability functions based on the given method.
    Takes as input logits and outputs the corresponding probs.

    E.g. f_model_wrapper(z) = P(y=1|z) = 1 / (1 + exp(-z)) for logistic well-specified model.

    Args:
        method: str, method to generate labels, one of ['logistic', 'misspec', 'misspec2'].
        a: float, parameter for label_method.
        b: float, parameter for label_method.
        thr: float, threshold for 'misspec2' model.
        zeta: float, parameter for 'misspec2' model.

    Returns:
        f: function, probability function.
    """
    assert method in ['logistic', 'misspec', 'misspec2']
    if method == 'logistic':
        return lambda z: 1 / (1 + np.exp(-z))

    elif method == 'misspec':
        assert a is not None and b is not None, "a and b must be provided for misspecified model"
        return lambda z: np.where(z > 0, a, b)

    elif method == 'misspec2':
        assert a is not None and b is not None, "a and b must be provided for misspecified model"

        def f(z, label_method_b, label_method_a):
            if z < -thr:
                return 1 - zeta
            elif (z >= -thr) and (z < 0):
                return label_method_b
            elif (z >= 0) and (z < thr):
                return label_method_a
            else:
                return zeta

        return lambda z: f(z, b, a)
    raise NotImplementedError


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    The derivative of the sigmoid function.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def alpha_exp_wrapper(alpha_exp, rmax=None):
    """
    Exponentiate the hardness scores p(1-p) with the given exponent, Eq 6.1 in the paper. For alpha_exp = -np.inf,
    the function returns a binary function that returns 1 if the hardness score is less than or equal to rmax,
    and 0 otherwise, emulating topK easy samples. For alpha_exp = np.inf, the function returns a binary function that
    returns 1 if the hardness score is greater than or equal to rmax, and 0 otherwise, emulating topK hard samples.
    """
    if alpha_exp == -np.inf:
        return lambda z: (sigmoid_prime(z) <= rmax).astype(int)
    elif alpha_exp == np.inf:
        return lambda z: (sigmoid_prime(z) >= rmax).astype(int)
    else:
        return lambda z: sigmoid_prime(z) ** alpha_exp


def get_c_opt(unnormalized_weights, num_train_samples):
    """
    Compute the optimal normalization constant to ensure the correct expected sub-sampled size while ensuring
    the individual selection probabilities are closest to unnormalized_weights.

    Args:
        unnormalized_weights: np.array, unnormalized weights. Typically, the hardness scores p(1-p).
        num_train_samples: int, number of training samples.

    Returns:
        c_opt: float, optimal normalization constant/
    """
    # initialize
    min_c = 0
    max_c = 1 / np.min(unnormalized_weights)
    f = lambda c: np.sum(np.minimum(c * unnormalized_weights, 1)) - num_train_samples
    c_opt = scipy.optimize.bisect(f, min_c, max_c)

    return c_opt


def clip_weights(x, clip=10):
    x = np.minimum(x, clip)
    x = np.maximum(x, -clip)
    return x


def run_grid_search(func, param_grid, opt_params, save_path, n_jobs, queue):
    """
    Run a grid search over the parameter grid for the given function.

    Args:
        func: function to be called
        param_grid: dictionary of parameter names and values
        opt_params: dictionary of optimization parameters for the function
        n_jobs: number of parallel jobs to run
        queue: multiprocessing queue for logging
    """
    param_combinations = itertools.product(*param_grid.values())

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(func, dict(zip(param_grid.keys(), combination)),
                            opt_params=opt_params,
                            save_path=save_path, queue=queue)
            for combination in param_combinations
        ]
        for future in futures:
            result = future.result()
            print(result)


def setup_logger(queue, logfile):
    """
    Set up the logger to log to both file and stdout, with messages being sent through a multiprocessing queue.
    This setup is for the main process to handle logging messages sent from worker processes.
    """
    # Logger configuration
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    fh = FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler
    ch = StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # Queue listener setup
    listener = QueueListener(queue, fh, ch)
    listener.start()

    return listener


def worker_configurer(queue):
    """
    The worker process configuration, setting up a QueueHandler.
    """
    h = QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

