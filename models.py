import numpy as np
from scipy import stats as st
from simulation import data_simulation
from pathlib import Path
import pickle

def weighted_quantile(X, W, alpha):
    """Weighted quantile approximation"""
    return np.quantile(a=X, q=alpha, weights=W, method='inverted_cdf')

def weighted_es(X, W, alpha):
    """
    Weighted expected shortfall approximation as the average of 4 weighted quantiles
    with alpha \in {alpha, 0.75*alpha+0.25, 0.5*alpha+0.5, 0.25*alpha+0.75}
    """
    alphas = [alpha, 0.75*alpha+0.25, 0.5*alpha+0.5, 0.25*alpha+0.75]
    return np.mean([np.quantile(a=X, q=alpha, weights=W, method='inverted_cdf') for alpha in alphas])

def std_Z(X, W, alpha, qW_hat):
    return np.sqrt(np.mean(W**2 * (alpha - (X<=qW_hat))**2))

def confidence_interval_qW(X, W, alpha, eta):
    qW_hat = weighted_quantile(X, W, alpha)  # Weighted quantile estimation
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    c = c_eta * std_Z(X, W, alpha, qW_hat) / np.mean(W)
    alpha_left = np.maximum(0, alpha-(c/np.sqrt(n)))
    alpha_right = np.minimum(alpha + (c/np.sqrt(n)), 1)
    ci_left = weighted_quantile(X, W, alpha_left)
    ci_right = weighted_quantile(X, W, alpha_right)
    return ci_left, ci_right, qW_hat

def confidence_interval_esW(X, W, alpha, eta):
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    esW_hat = weighted_es(X, W, alpha)  # Weighted quantile estimation
    c = c_eta * np.sqrt(np.mean(W**2)) * np.maximum((1 - alpha), 0.75*alpha+0.25) / np.mean(W)

    alpha_left = np.maximum(0, alpha-(c/np.sqrt(n)))
    alpha_right = np.minimum(alpha + (c/np.sqrt(n)), 1)
    ci_left = weighted_es(X, W, alpha_left)
    ci_right = weighted_es(X, W, alpha_right)
    return ci_left, ci_right, esW_hat

def fit_ci(scenario, n_replications, n_samples, theta, alpha, eta, sW_real, method='qW'):
    """

    Parameters
    ----------
    scenario :
    n_replications :
    n_samples :
    theta :
    alpha :
    eta :
    sW_real : float
        Real Weighted statistic
    method : str
        Choose between {qW, esW}. Default qW.

    Returns
    -------

    """
    np.random.seed(scenario)
    dict_result = {'ci_left': [], 'ci_right': [], 'sW_hat': [], 'ci_width': [], 'coverage': []}
    for rep in range(n_replications):
        X_samples, W_samples = data_simulation(scenario=scenario, n=n_samples, theta=theta, seed=None)
        if method == 'qW':
            ci_left, ci_right, sW_hat = confidence_interval_qW(X_samples, W_samples, alpha, eta)
        elif method == 'esW':
            ci_left, ci_right, sW_hat = confidence_interval_esW(X_samples, W_samples, alpha, eta)
        else:
            raise ValueError('Method must be either qW or esW')
        dict_result['ci_left'].append(ci_left)
        dict_result['ci_right'].append(ci_right)
        dict_result['sW_hat'].append(sW_hat)
        dict_result['ci_width'].append(ci_right - ci_left)
        dict_result['coverage'].append((ci_left <= sW_real) & (sW_real <= ci_right))
    return dict_result

def get_real_data(scenario, n, theta, alpha, method='qW'):
    """

    Parameters
    ----------
    scenario :
    n :
    theta :
    alpha :
    method : str
        Choose between {qW, esW}. Default qW.

    Returns
    -------

    """
    """Load the real statistic (either qW or esW) stored in a csv with shape (alphas, scenarios) """
    pathdir = Path("ckpt")
    filename = f"{method}_n{n}_theta{theta}.pickle"
    pathdir.mkdir(parents=True, exist_ok=True)
    pathfile = pathdir / filename
    if pathfile.is_file():
        with open(pathfile, 'rb') as fr:
            dict_data = pickle.load(fr)
    else:
        dict_data = {}

    try:     # Check if the tuple (alpha, scenario) exists
        return dict_data[f'scenario_{scenario}'][f'alpha_{alpha}']
    except KeyError:
        # Simulations
        X_real, W_real = data_simulation(scenario=scenario, n=n, theta=theta)
        if method == 'qW':
            sW_real = weighted_quantile(X_real, W_real, alpha)
        elif method == 'esW':
            sW_real = weighted_es(X_real, W_real, alpha)
        else:
            raise ValueError('Method must be either qW or esW')

        # Update the dictionary
        if f'scenario_{scenario}' in dict_data.keys():  # alpha is missing in this scenario
            dict_data[f'scenario_{scenario}'].update({f'alpha_{alpha}': sW_real})
        else:  # update the dictionary
            dict_data.update({f'scenario_{scenario}':{f'alpha_{alpha}': sW_real}})

        # save the dictionary
        with open(pathfile, 'wb') as fw:
            pickle.dump(dict_data, fw, pickle.HIGHEST_PROTOCOL)
        return sW_real





