from simulation import data_simulation
from pathlib import Path
import pickle
from models import weighted_quantile, weighted_es

def get_real_data(scenario, n, theta, alpha, method='qW'):
    """
    Load the real statistic (either qW or esW) stored in a csv with shape (alphas, scenarios)
    Parameters
    ----------
    scenario : int
        scenario number
    n : int
        number of samples
    theta : float
        dependence parameter in the Gumbel Copula
    alpha : float
        risk level in [0,1]
    method : str
        Choose between {qW, esW}. Default qW.

    Returns
    -------
    sW_real: array
    """
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





