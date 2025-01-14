import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax.numpy as jnp
from bilby.gw.result import CBCResult

outdir = '/home/user/ckng/project/jim_GWTC3/bilby_runs/outdir'
events = [d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))]
events.sort()

for event in events:
    try:
        print(f'Plotting {event}')
        result_jim = jnp.load(f'/home/user/ckng/project/jim_GWTC3/jim_runs/outdir/{event}/samples.npz')
        n_samples = 5000

        keys = ['M_c', 'd_L', 'dec', 'iota', 'phase_c', 'psi', 'q', 'ra']

        samples_jim = []
        for key in keys:
            samples_jim.append(result_jim[key])
        samples_jim = np.array(samples_jim).T
        samples_jim = samples_jim[np.random.choice(samples_jim.shape[0], n_samples), :]

        result_bilby = CBCResult.from_json(f'/home/user/ckng/project/jim_GWTC3/bilby_runs/result/{event}_GR.json.gz').posterior
        result_bilby = result_bilby.sample(n_samples)
        samples_bilby = []
        for key in keys:
            key = key.replace('M_c', 'chirp_mass')
            key = key.replace('q', 'mass_ratio')
            key = key.replace('d_L', 'luminosity_distance')
            key = key.replace('phase_c', 'phase')
            samples_bilby.append(result_bilby[key].values)
        samples_bilby = np.array(samples_bilby).T

        fig = corner(samples_jim, labels=keys, color='blue', hist_kwargs={'density': True})
        corner(samples_bilby, labels=keys, fig=fig, color='red', hist_kwargs={'density': True})
        plt.savefig(f'figures/{event}.jpg')
        plt.close()
    except Exception as e:
        print(f'Error: {e}')
        continue
