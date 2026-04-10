import numpy as np
from scipy import stats as st
from simulation import data_simulation
import matplotlib.pyplot as plt
from scipy import stats


def weighted_quantile(X, W, alpha):
    """
    Weighted quantile estimation
    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]

    Returns
    -------
    weighted_quantile: float
    """
    return np.quantile(a=X, q=alpha, weights=W, method='inverted_cdf')

def weighted_es(X, W, alpha, perturbation=0):
    """
    Weighted expected shortfall approximation as the average of 4 weighted quantiles
    with alpha \in {alpha, 0.75*alpha+0.25, 0.5*alpha+0.5, 0.25*alpha+0.75}
    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    perturbation : float, optional
        lower or upper perturbation. Default 0.

    Returns
    -------
    weighted_es: float
    """
    alphas = [alpha, 0.75*alpha+0.25, 0.5*alpha+0.5, 0.25*alpha+0.75]
    return np.mean([np.quantile(a=X, q=np.minimum(1, np.maximum(0, alpha+perturbation)), weights=W,
                                method='inverted_cdf') for alpha in alphas])

def sigma_n(X, W, alpha, qW_hat):
    """
    Empirical standard deviation \sigma_n used in the variance of the CLT
    \sqrt{E[W^2 (\alpha - \ind{X\leq q_W(\alpha)})**2]}

    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    qW_hat : array
        estimated weighted quantile

    Returns
    -------
    float or array
    """
    return np.sqrt(np.mean(W**2 * (alpha - (X<=qW_hat))**2))

def cov_n(X, W, alpha1, qW1_hat, alpha2, qW2_hat):
    """
    Empirical covariance \Sigma_n,k,k^\prime used in the variance of the mltivariate CLT
    \sqrt{E[W^2 (\alpha_1 - \ind{X\leq q_W(\alpha1)}) * (\alpha_2 - \ind{X\leq q_W(\alpha2)})]}

    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    qW_hat : array
        estimated weighted quantile

    Returns
    -------
    float or array
    """
    return np.mean(W**2 * (alpha1 - (X<=qW1_hat)) * (alpha2 - (X<=qW2_hat)))

def confidence_interval_qW(X, W, alpha, eta):
    """
    Confidence interval estimation of the weighted quantile according to the Wilks based method
    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    eta : float in [0, 1]
        confidence level

    Returns
    -------
    ci_left: float
    ci_right: float
    qW_hat: float
    """
    qW_hat = weighted_quantile(X, W, alpha)  # Weighted quantile estimation
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    c = c_eta * sigma_n(X, W, alpha, qW_hat) / np.mean(W)
    alpha_left = np.maximum(0, alpha-(c/np.sqrt(n)))
    alpha_right = np.minimum(alpha + (c/np.sqrt(n)), 1)
    ci_left = weighted_quantile(X, W, alpha_left)
    ci_right = weighted_quantile(X, W, alpha_right)
    return ci_left, ci_right, qW_hat

def confidence_interval_qW_density(X, W, alpha, eta):
    """
    Confidence interval estimation of the weighted quantile according to the density plug-in method
    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    eta : float in [0, 1]
        confidence level

    Returns
    -------
    ci_left: float
    ci_right: float
    qW_hat: float
    """
    qW_hat = weighted_quantile(X, W, alpha)  # Weighted quantile estimation
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    s = sigma_n(X, W, alpha, qW_hat) / (np.mean(W) * wkde(qW_hat, X, W))
    c = c_eta * s / np.sqrt(n)
    ci_left = qW_hat - c
    ci_right = qW_hat + c
    return ci_left, ci_right, qW_hat


def confidence_interval_esW(X, W, alpha, eta):
    """
    Confidence interval estimation of the weighted expected shortfall according to the Wilks based method
    Parameters
    ----------
    X : array
        main random variable in \R
    W : array
        weight random variable in \R^+
    alpha : float
        risk level in [0,1]
    eta : float
        confidence level in [0, 1]

    Returns
    -------
    ci_left: float
    ci_right: float
    esW_hat: float
    """
    c_eta = st.norm.ppf(1 - (1-eta)/2)  # confidence threshold
    n = len(X)  # number of samples
    esW_hat = weighted_es(X, W, alpha)  # Weighted quantile estimation
    alphas = [alpha, 0.75*alpha+0.25, 0.5*alpha+0.5, 0.25*alpha+0.75]
    list_covar = []
    for i, alpha1 in enumerate(alphas):
        for alpha2 in alphas[i:]: # upper triangular with diagonal
            qW1_hat = weighted_quantile(X, W, alpha1)
            qW2_hat = weighted_quantile(X, W, alpha2)
            covar = cov_n(X,W,alpha1, qW1_hat, alpha2, qW2_hat)
            list_covar.append(covar)
    sup_covar = np.max(list_covar)
    rho_s = np.sqrt(sup_covar) / np.mean(W)
    c = c_eta * rho_s
    perturbation = c/np.sqrt(n)
    ci_left = weighted_es(X, W, alpha, -perturbation)
    ci_right = weighted_es(X, W, alpha, perturbation)
    return ci_left, ci_right, esW_hat

def fit_ci(scenario, n_replications, n_samples, theta, alpha, eta, sW_real, method='qW'):
    """
    Fit the confidence interval estimator for a given scenario according to the Wilks-based method
    Parameters
    ----------
    scenario : int
        scenario number
    n_replications : int
        number of replications
    n_samples : int
        number of samples
    theta : float
        dependence parameter in the Gumbel Copula
    alpha : float
        risk level in [0,1]
    eta : float
        confidence level in [0, 1]
    sW_real : float
        Real Weighted statistic
    method : str
        Choose between {qW, esW}. Default qW.

    Returns
    -------
    dict_result: dict
    """
    np.random.seed(scenario)
    dict_result = {'ci_left': [], 'ci_right': [], 'sW_hat': [], 'ci_width': [], 'coverage': []}
    for rep in range(n_replications):
        X_samples, W_samples = data_simulation(scenario=scenario, n=n_samples, theta=theta, seed=None)
        # Get the (left, right) confidence interval and the estimated weighted statistics (qW_hat or esW_hat)
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


def fit_ci_density(scenario, n_replications, n_samples, theta, alpha, eta, sW_real):
    """
    Fit the confidence interval estimator for a given scenario according to the Density plug-in method
    Parameters
    ----------
    scenario : int
        scenario number
    n_replications : int
        number of replications
    n_samples : int
        number of samples
    theta : float
        dependence parameter in the Gumbel Copula
    alpha : float
        risk level in [0,1]
    eta : float
        confidence level in [0, 1]
    sW_real : float
        Real Weighted statistic

    Returns
    -------
    dict_result: dict
    """
    np.random.seed(scenario)
    dict_result = {'ci_left': [], 'ci_right': [], 'sW_hat': [], 'ci_width': [], 'coverage': []}
    for rep in range(n_replications):
        X_samples, W_samples = data_simulation(scenario=scenario, n=n_samples, theta=theta, seed=None)
        # Get the (left, right) confidence interval and the estimated weighted statistics (qW or esW)
        ci_left, ci_right, sW_hat = confidence_interval_qW_density(X_samples, W_samples, alpha, eta)
        dict_result['ci_left'].append(ci_left)
        dict_result['ci_right'].append(ci_right)
        dict_result['sW_hat'].append(sW_hat)
        dict_result['ci_width'].append(ci_right - ci_left)
        dict_result['coverage'].append((ci_left <= sW_real) & (sW_real <= ci_right))
    return dict_result

def wkde(x, X, W, h='scott'):
    """
    Weighted Kernel Density Estimation
    Parameters
    ----------
    x : array or list or float
        observations
    X : array
        random variable of interest in R
    W : array
        random variable of the weight in R^+
    h : string (scott, silverman) or float strictly positive
        bandwidth

    Returns
    -------

    """
    kde = stats.gaussian_kde(X, weights=W, bw_method=h)
    return kde.evaluate(x)

    # Implementation of the wKDE
    # --------------
    # weights = W / np.sum(W)
    # norm = 1/(h*np.sqrt(2*np.pi))
    # try:
    #     return np.array([np.sum(weights * np.exp(-(_x-X)**2/(2*h**2))) * norm for _x in x])
    # except:
    #     return np.sum(weights * np.exp(-(x - X)**2/(2*h**2))) * norm
    # ------------------








