import sys
import os
import pickle
import bz2
from matplotlib import pyplot as plt

path = 'C:/Users/me/research/relative-utility-value/'
sys.path.append(path)

from ruv.relative_utility_value import *
from ruv.damage_functions import *
from ruv.economic_models import *
from ruv.utility_functions import *
from ruv.helpers import *

awrc = '401012'
dataset_filepath = os.path.join(path, 'examples', '%s_data.pickle.bz2' % awrc)
parallel_nodes = 6
alpha_step = 0.1

with bz2.BZ2File(dataset_filepath, 'r') as f:
    data = pickle.load(f)
obs, clim_ens, fcst_ens = data['obs'], data['clim'], data['fcst']
print(obs.shape, fcst_ens.shape, clim_ens.shape)

decision_threshold = np.nanquantile(obs, 0.75)

decision_definition = {
    'alphas': np.arange(alpha_step, 1, alpha_step),
    'damage_function': [binary, {'max_loss': 1, 'min_loss': 0, 'threshold': decision_threshold}],        
    'utility_function': [cara, {'A': 0}],
    'economic_model': [cost_loss, cost_loss_analytical_spend],
    'decision_thresholds': np.insert([decision_threshold], 0, 0),
    'event_freq_ref': True
}
ref = clim_ens

decision_definition['decision_method'] = 'optimise_over_forecast_distribution'
results_optim = relative_utility_value(obs, fcst_ens, ref, decision_definition, parallel_nodes, verbose=True)

plt.plot(decision_definition['alphas'], results_optim['ruv'])
plt.savefig('ruv.png')



