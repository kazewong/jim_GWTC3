import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from bilby.gw.result import CBCResult
import pandas as pd

outdir = 'bilby_runs/outdir'
csv = pd.read_csv('event_status.csv')
events = csv['Event'].values

for event in events:
    try:
        if not os.path.exists(f'figures/{event}.jpg'):
            result_jim = jnp.load(f'jim_runs/outdir/{event}/samples.npz')
            n_samples = 5000

            keys = ['M_c', 'q', 'd_L', 'iota', 'ra', 'dec', 't_c', 'psi', 'phase_c', 's1_mag', 's2_mag']

            samples_jim = []
            for key in keys:
                samples_jim.append(result_jim[key])
            samples_jim = np.array(samples_jim).T
            samples_jim = samples_jim[np.random.choice(samples_jim.shape[0], n_samples), :]

            files = os.listdir(f'{outdir}/{event}/final_result')
            if len(files) != 1 and f'{event}_result.hdf5' not in files:
                print(f'Error: {event} does not have a unique result file')
                continue
            else:
                print(f'Plotting {event}')
                file = files[0]
            result_bilby = CBCResult.from_hdf5(f'{outdir}/{event}/final_result/{file}')
            trigger_time = float(result_bilby.meta_data['command_line_args']['trigger_time'])
            result_bilby = result_bilby.posterior
            result_bilby['geocent_time'] -= trigger_time
            if len(result_bilby) > n_samples:
                result_bilby = result_bilby.sample(n_samples)
            else:
                print(f'Warning: {event} has only {len(result_bilby)} samples')
            samples_bilby = []
            for key in keys:
                key = key.replace('M_c', 'chirp_mass')
                key = key.replace('q', 'mass_ratio')
                key = key.replace('d_L', 'luminosity_distance')
                key = key.replace('t_c', 'geocent_time')
                key = key.replace('phase_c', 'phase')
                key = key.replace('s1_mag', 'a_1')
                key = key.replace('s2_mag', 'a_2')
                samples_bilby.append(result_bilby[key].values)
            samples_bilby = np.array(samples_bilby).T

            fig = corner(samples_jim, labels=keys, color='blue', hist_kwargs={'density': True})
            corner(samples_bilby, labels=keys, fig=fig, color='red', hist_kwargs={'density': True})

            fig.legend(['jim', 'bilby'], loc='right', fontsize=20)

            fig.savefig(f'figures/{event}.jpg')
            plt.close(fig)

            csv.loc[csv['Event'] == event, 'Comparison'] = 'good'
            csv.to_csv('event_status.csv', index=False)
    except Exception as e:
        print(f'Error: {e}')
        continue
