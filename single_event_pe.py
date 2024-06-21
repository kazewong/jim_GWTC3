import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import Composite, Unconstrained_Uniform, Sphere
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from flowMC.strategy.optimization import optimization_Adam

from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets
from tap import Tap
import argparse

from parameter_estimation.utilities import mkdir
from parameter_estimation.plotting import plotPosterior, plotRunningProgress, plotLikelihood
from parameter_estimation.save import savePosterior




def runSingleEventPE(output_dir, event, gps, duration, post_trigger_duration, Mc_prior, ifos, waveform, heterodyned):
    jax.config.update("jax_enable_x64", True)
    
    #################### Fetching the Data ####################
    total_time_start = time.time()

    # first, fetch a 4s segment centered on GW150914
   
    
    post_trigger_duration = duration // 2
    start_pad = duration - post_trigger_duration
    end_pad = post_trigger_duration
    fmin = 20.0
    fmax = 1024.0

    
    detectors = []

    if "H1" in ifos:
        H1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=duration*4, tukey_alpha=0.2)
        detectors.append(H1)

    if "L1" in ifos:
        L1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=duration*4, tukey_alpha=0.2)
        detectors.append(L1)
        
    if "V1" in ifos:    
        V1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=duration*4, tukey_alpha=0.2)
        detectors.append(V1)
        
    waveform = RippleIMRPhenomPv2(f_ref=20)

    ###########################################
    ########## Set up priors ##################
    ###########################################

    Mc_prior = Unconstrained_Uniform(Mc_prior[0], Mc_prior[1], naming=["M_c"])
    q_prior = Unconstrained_Uniform(
        0.125,
        1.0,
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    )
    s1_prior = Sphere(naming="s1")
    s2_prior = Sphere(naming="s2")
    dL_prior = Unconstrained_Uniform(0.0, 10000.0, naming=["d_L"])
    t_c_prior = Unconstrained_Uniform(-0.5, 0.5, naming=["t_c"])
    phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
    cos_iota_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["cos_iota"],
        transforms={
            "cos_iota": (
                "iota",
                lambda params: jnp.arccos(
                    jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
    ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
    sin_dec_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )

    prior = Composite(
        [
            Mc_prior,
            q_prior,
            s1_prior,
            s2_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
        ],
    )

    epsilon = 1e-3
    bounds = jnp.array(
        [
            [1.0, 120.0],
            [0.125, 1.0],
            [0, jnp.pi],
            [0, 2 * jnp.pi],
            [0.0, 1.0],
            [0, jnp.pi],
            [0, 2 * jnp.pi],
            [0.0, 1.0],
            [0.0, 10000],
            [-0.05, 0.05],
            [0.0, 2 * jnp.pi],
            [-1.0, 1.0],
            [0.0, jnp.pi],
            [0.0, 2 * jnp.pi],
            [-1.0, 1.0],
        ]
    ) + jnp.array([[epsilon, -epsilon]])

    if heterodyned:
        likelihood = HeterodynedTransientLikelihoodFD(detectors, prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger_duration,n_bins=1000)
    else:
        likelihood = TransientLikelihoodFD(detectors, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger_duration)


    mass_matrix = jnp.eye(prior.n_dim)
    mass_matrix = mass_matrix.at[1, 1].set(1e-3)
    mass_matrix = mass_matrix.at[9, 9].set(1e-3)
    local_sampler_arg = {"step_size": mass_matrix * 1e-3}

    Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1, bounds=bounds)

    import optax
    n_epochs = 40
    n_loop_training = 100
    total_epochs = n_epochs * n_loop_training
    start = total_epochs//10
    learning_rate = optax.polynomial_schedule(
        1e-3, 1e-4, 400, total_epochs - start, transition_begin=start
    )

    jim = Jim(
        likelihood,
        prior,
        n_loop_training=n_loop_training,
        n_loop_production=20,
        n_local_steps=10,
        n_global_steps=1000,
        n_chains=500,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        n_max_examples=30000,
        n_flow_sample=100000,
        momentum=0.9,
        batch_size=30000,
        use_global=True,
        keep_quantile=0.0,
        train_thinning=1,
        output_thinning=10,
        local_sampler_arg=local_sampler_arg,
        # strategies=[Adam_optimizer,"default"],
    )

    import numpy as np

    jim.sample(jax.random.PRNGKey(42))


    #################### Output ####################
    result = jim.get_samples()
    summary = jim.Sampler.get_sampler_state(training=True)

    mkdir(output_dir)
    plotPosterior(result, event, output_dir)
    savePosterior(result, event, output_dir)
    plotRunningProgress(summary, event, output_dir)
    plotLikelihood(summary, event, output_dir)


if __name__ == "__main__":
    #################### Parameters to Tune for each Events ####################
    """
    event: name of the event
    gps: trigger time of the event
    duration: duration of the segment for analysis
    post_trigger_duration: duration after the trigger time
    Mc_prior: prior for chirp mass, [min, max]
    ifos: list of interferometers to use
    waveform: waveform model to use
    heterodyned: whether to use heterodyned likelihood
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="output", help='Your name')
    parser.add_argument('--event', type=str, default="GW191216_213338", help='Your name')
    parser.add_argument('--gps', type=float, default=1260567236.4, help='Your name')
    parser.add_argument('--duration', type=float, default=4, help='Your name')
    parser.add_argument('--post_trigger_duration', type=float, default=2, help='Your name')
    parser.add_argument('--Mc_prior', type=float, nargs='+', help='Mc_prior', default=[3.0, 30.0])
    parser.add_argument('--ifos', type=str, nargs='+', help='ifos', default=["H1", "V1"])
    parser.add_argument('--waveform', type=str, default="RippleIMRPhenomPv2", help='Your name')
    parser.add_argument('--heterodyned', type=bool, default=False, help='Your name')
    args = parser.parse_args()
    
    
    runSingleEventPE(
        output_dir=args.output_dir,
        event=args.event, 
        gps=args.gps, 
        duration=args.duration, 
        post_trigger_duration=args.post_trigger_duration, 
        Mc_prior=args.Mc_prior, 
        ifos=args.ifos,
        waveform=args.waveform,
        heterodyned=args.heterodyned
        )
