from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
from subsampler import Subsampler
from utils.data_gen_utils import *
from utils.misc_utils import *
from utils.consts import *
import os
import yaml
from datetime import datetime
from tqdm import tqdm
import argparse

def run_experiment(X, y, theta_surr, theta_0, lambd, sampler, N_samplings, reweight, alpha_exp, split_pct):
    """
    Given a dataset and sampling probabilities does several random samplings and trains a model on the
    subsampled data.

    Args:
        X: dict of train, val, test
        y: dict of train, val, test
        theta_surr: np.array, surrogate model weights of shape (1, p)
        theta_0: np.array, true model weights of shape (1, p)
        lambd: float, regularization parameter
        sampler: Subsampler object
        N_samplings: int, number of random samplings
        reweight: bool, whether to reweight the samples
        alpha_exp: str, type of sampling
        split_pct: float, percentage of data to sample
    """
    # Using the list of keys to create a dictionary with empty lists as values
    results_dict_experiment = {key: [] for key in EXPERIMENT_KEYS}

    if alpha_exp in ['Easy', 'Hard'] or split_pct == 100:
        reweight = False
    total_num_samples = len(y['train'])

    for split_seed in range(N_samplings):
        if split_pct == 100:
            X_select = X['train']
            y_select = y['train']
        else:
            if alpha_exp in ['Easy', 'Hard']:
                train_idx = sampler.sampleTopk_from_unnormalized_pi(split_pct, alpha_exp)
            else:
                assert len(sampler.pi) == total_num_samples
                train_idx = sampler.sample_from_pi(split_seed)
                train_weight = sampler.pi[train_idx]

            X_select = X['train'][train_idx]
            y_select = y['train'][train_idx]
        results_dict_experiment['effectiveN'].append(len(X_select))
        model = LogisticRegression(fit_intercept=False, random_state=42,
                                   max_iter=1000000, C=1 / (lambd * total_num_samples), tol=1e-6, verbose=False)
        if reweight:
            model.fit(X_select, y_select, sample_weight=1 / train_weight)
        else:
            model.fit(X_select, y_select)
        theta = model.coef_
        preds_test = model.predict_proba(X['test'])
        preds_hard_test = model.predict(X['test'])

        preds_val = model.predict_proba(X['val'])
        preds_hard_val = model.predict(X['val'])

        scoreTrain = model.score(X_select, y_select)

        ## Compute grad :
        z = X_select @ theta.T
        h = sigmoid(z).squeeze()
        grad = (1 / (lambd * total_num_samples)) * X_select.T @ (h - y_select) + theta.squeeze()

        alpha_0 = theta @ theta_0.T / np.linalg.norm(theta_0)
        if (theta_0 == theta_surr).all():
            alpha_s = 0
        else:
            theta_surr_star_perp = theta_surr - (
                    theta_surr @ theta_0.T / (np.linalg.norm(theta_0) ** 2)) * theta_0
            alpha_s = theta @ theta_surr_star_perp.T / np.linalg.norm(theta_surr_star_perp)
            alpha_s = alpha_s.item()

        alpha_perp = np.sqrt(np.linalg.norm(theta) ** 2 - alpha_0 ** 2 - alpha_s ** 2)
        mis_error_test = np.mean(preds_hard_test != y['test'])
        mis_error_val = np.mean(preds_hard_val != y['val'])
        testError = log_loss(y['test'], preds_test)
        valError = log_loss(y['val'], preds_val)

        results_dict_experiment['alpha_0'].append(alpha_0.item())
        results_dict_experiment['alpha_s'].append(alpha_s)
        results_dict_experiment['alpha_perp'].append(alpha_perp.item())
        results_dict_experiment['mis_error_test'].append(mis_error_test)
        results_dict_experiment['mis_error_val'].append(mis_error_val)
        results_dict_experiment['log_loss_test'].append(testError)
        results_dict_experiment['log_loss_val'].append(valError)
        results_dict_experiment['acc_train'].append(scoreTrain)
        results_dict_experiment['theta_surr_norm'].append(np.linalg.norm(theta))
        results_dict_experiment['grad_norm'].append(np.linalg.norm(grad))

    return results_dict_experiment


def generate_cube(split_pct_sweep, lambd_sweep, alpha_exp_sweep, data_p,
                  reweight, N_gen, N_samplings_max, lambd_surr, N_train, N_val, N_test, N_surr, data_theta_norm,
                  save_path, label_method, label_method_a=None, label_method_b=None):
    exp_name = (f'syntheticData_SnR-{data_theta_norm}_P{data_p}_method-{label_method}_reweight-{reweight}_'
                f'a-{label_method_a}_b-{label_method_b}_N_train-{N_train}_Nsurr-{N_surr}_lambd_surr-{lambd_surr}_'
                f'{N_gen}sim.csv')

    results_dict = {key: [] for key in EXPERIMENT_KEYS + PARAMETER_KEYS + ['N_samplings']}
    results_dict["label_method"] = label_method
    results_dict['label_method_a'] = label_method_a
    results_dict['label_method_b'] = label_method_b

    for data_seed in tqdm(list(range(N_gen))):
        # do multiple independent synthetic data generations
        X_total, y_total, theta_0 = generate_data(N_train + N_val + N_test + N_surr, data_p, data_theta_norm, data_seed,
                                                  label_method,
                                                  label_method_a, label_method_b)

        X = {'train': X_total[:N_train, :],
             'val': X_total[N_train:N_train + N_val, :],
             'test': X_total[N_train + N_val:N_train + N_val + N_test, :],
             'surr': X_total[N_train + N_val + N_test:, :]}
        y = {'train': y_total[:N_train],
             'val': y_total[N_train:N_train + N_val],
             'test': y_total[N_train + N_val:N_train + N_val + N_test],
             'surr': y_total[N_train + N_val + N_test:]}

        assert len(X['train']) == len(y['train']) == N_train
        assert len(X['val']) == len(y['val']) == N_val
        assert len(X['test']) == len(y['test']) == N_test
        assert len(X['surr']) == len(y['surr']) == N_surr

        if lambd_surr is None:
            # Perfect Surrogate
            theta_surr = theta_0
            surr_score = None
        else:
            # fit a surrogate model
            model_surr = LogisticRegression(fit_intercept=False, random_state=42,
                                            max_iter=1000000, C=1 / (lambd_surr * N_surr), tol=1e-6, verbose=False)
            model_surr.fit(X['surr'], y['surr'])
            theta_surr = model_surr.coef_
            surr_score = model_surr.score(X['test'], y['test'])

        # calculate beta_0 and beta_perp parameters based on the surrogate model parameters
        beta_0 = (theta_surr @ theta_0.T / data_theta_norm).item()
        beta_perp = np.sqrt(np.linalg.norm(theta_surr) ** 2 - beta_0 ** 2)

        # for each subsampling scheme (alpha_exp), and given a regularization parameter (lambd), perform subsampling
        for alpha_exp, lambd in tqdm(list(itertools.product(alpha_exp_sweep, lambd_sweep))):
            if alpha_exp in ['Easy', 'Hard']:
                # topK sampling, alpha_exp is not used, and we pick the topK samples based on the surrogate model scores
                sampler = Subsampler(1, theta_surr)
                N_samplings = 1
            else:
                sampler = Subsampler(alpha_exp, theta_surr)
                N_samplings = N_samplings_max

            # get (unnormalized) sampling probabilities
            sampler.get_unnormalized_pi(X)
            for split_pct in split_pct_sweep:
                if split_pct == 100:
                    # handle the case of taking full-data separately: no subsampling scheme involved
                    continue
                # get normalized sampling probabilities
                sampler.get_pi(split_pct)

                # perform subsampling, train on the sampled data, and evaluate on held-out test data
                results_dict_experiment = run_experiment(X, y, theta_surr, theta_0, lambd, sampler, N_samplings,
                                                         reweight, alpha_exp,
                                                         split_pct)
                update_exp_params(results_dict, results_dict_experiment)

                param_points = {'beta_0': beta_0, 'beta_perp': beta_perp, "surr_score": surr_score,
                                'N_gen': data_seed, 'lambd': lambd, "alpha_exp": alpha_exp,
                                'data_p': data_p, 'split_pct': split_pct, "lambd_surr": lambd_surr}
                update_params(results_dict, param_points, N_samplings)
                results_dict['N_samplings'] += list(range(N_samplings))

        if 100 in split_pct_sweep:
            # handle the case of taking full-data separately: no subsampling scheme involved
            # regularized logistic regression
            split_pct = 100
            N_samplings = 1
            for lambd in lambd_sweep:
                alpha_exp = None
                sampler = None
                results_dict_experiment = run_experiment(X, y, theta_surr, theta_0, lambd, sampler, N_samplings,
                                                         reweight, alpha_exp,
                                                         split_pct)
                update_exp_params(results_dict, results_dict_experiment)

                param_points = {'beta_0': beta_0, 'beta_perp': beta_perp, "surr_score": surr_score,
                                'N_gen': data_seed, 'lambd': lambd, "alpha_exp": alpha_exp,
                                'data_p': data_p, 'split_pct': split_pct, "lambd_surr": lambd_surr}

                update_params(results_dict, param_points, N_samplings)
                results_dict['N_samplings'] += list(range(N_samplings))

    res = pd.DataFrame(results_dict)

    res_path = os.path.join(save_path, exp_name)
    print("--" * 50)
    print(f'Saved to {res_path}')
    print("--" * 50)
    res.to_csv(res_path, index=False)


if __name__ == '__main__':
    # read config file
    parser = argparse.ArgumentParser(description='Run imperfect surrogate theory experiment')
    parser.add_argument('--config', type=str, help='Path to the config file', required=True)
    config_file = parser.parse_args().config

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    param_grid = config['param_grid']
    save_path = config['save_path']
    os.makedirs(save_path, exist_ok=True)

    generate_cube(param_grid['split_pct'], param_grid['lambd'], param_grid['alpha_exp'],
                  param_grid['data_p'], param_grid['reweight'], param_grid['N_gen'], param_grid['N_samplings'],
                  data_theta_norm=param_grid['data_theta_norm'], N_train=param_grid['N_train'],
                  N_val=param_grid['N_val'], N_test=param_grid['N_test'], N_surr=param_grid['N_surr'],
                  lambd_surr=param_grid['lambd_surr'], save_path=save_path,
                  label_method=param_grid['label_method'],
                  label_method_a=param_grid['label_method_a'], label_method_b=param_grid['label_method_b'])
