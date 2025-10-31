import numpy as np
from pathlib import Path
import pickle

import pandas as pd
from scipy import stats as st
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula

# Define a dictionary of scenarios for each marginal
DICT_SCENARIOS = {
    1: [st.burr12(c=1/0.3, d=1), st.burr12(c=1/0.3, d=1)],  # where c=-rho/gamma, k=-1/rho with gamma>0 and rho<0
    2: [st.laplace(loc=0, scale=1), st.burr12(c=1/0.3, d=1)],
    3: [st.norm(loc=0, scale=1), st.burr12(c=1/0.3, d=1)],
    4: [st.burr12(c=1/0.3, d=1), st.lognorm(s=0.5)],
    5: [st.laplace(loc=0, scale=1), st.lognorm(s=0.5)],
    6: [st.norm(loc=0, scale=1), st.lognorm(s=0.5)],
}

def data_simulation(scenario, n, theta, seed=123):
    if seed is not None:
        np.random.seed(scenario)
    try:
        marginals = DICT_SCENARIOS[scenario]
    except KeyError:
        raise f"Please enter a valid scenario in {DICT_SCENARIOS.keys()}"
    copula = CopulaDistribution(copula=GumbelCopula(theta=theta, k_dim=2), marginals=marginals) # define a Bivariate Gumbel Copula
    output = copula.rvs(n) # one can also set the seed here with random_state=seed
    return output[:, 0], output[:, 1]





