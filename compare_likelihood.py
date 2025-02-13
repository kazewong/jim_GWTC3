import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import pickle
import jax
import numpy as np
import ast
import jax.numpy as jnp
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import bilby
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.result import CBCResult
import pandas as pd

csv = pd.read_csv('event_status.csv')
events = csv['Event'].values

if not os.path.exists("likelihood_comparison.pkl"):
    event_dict = {}
    event_max_abs_diff = 0
    event_logL_min = np.inf
    event_logL_max = 0

    for event_id in events:
        file = os.listdir("bilby_runs/outdir/{}/data".format(event_id))[0]
        with open(f"bilby_runs/outdir/{event_id}/data/{file}", "rb") as f:
            data_dump = pickle.load(f)

        duration = float(data_dump.meta_data['command_line_args']['duration'])
        post_trigger = float(data_dump.meta_data['command_line_args']['post_trigger_duration'])
        gps = float(data_dump.trigger_time)
        fmin: dict[str, float] = data_dump.meta_data['command_line_args']['minimum_frequency']
        fmax: dict[str, float] = data_dump.meta_data['command_line_args']['maximum_frequency']

        try:
            fmin = float(np.min(list(ast.literal_eval(fmin).values())))
        except AttributeError:
            fmin = float(fmin)
        try:
            fmax = float(np.min(list(ast.literal_eval(fmax).values())))
        except AttributeError:
            fmax = float(fmax)

        ifos_list_string = data_dump.interferometers.meta_data.keys()

        ifos: list[GroundBased2G] = []
        for i, ifo_string in enumerate(ifos_list_string):

            ifo_bilby = data_dump.interferometers[i]
            assert ifo_bilby.name == ifo_string, f"ifo_bilby.name: {ifo_bilby.name} != ifo_string: {ifo_string}"

            print("Adding interferometer ", ifo_string)
            eval(f'ifos.append({ifo_string})')

            frequencies, data, psd = ifo_bilby.frequency_array, ifo_bilby.frequency_domain_strain, ifo_bilby.power_spectral_density_array
            
            mask = (frequencies >= fmin) & (frequencies <= fmax)
            frequencies = frequencies[mask]
            data = data[mask]
            psd = psd[mask]

            ifos[i].frequencies = frequencies
            ifos[i].data = data
            ifos[i].psd = psd

        waveform = RippleIMRPhenomPv2(f_ref=float(data_dump.meta_data['command_line_args']['reference_frequency']))

        likelihood = TransientLikelihoodFD(
            ifos, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger
        )

        result_bilby = CBCResult.from_hdf5(f"bilby_runs/outdir/{event_id}/final_result/"+os.listdir("bilby_runs/outdir/{}/final_result".format(event_id))[0]).posterior

        waveform_arguments = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=data_dump.meta_data['command_line_args']['reference_frequency'],
            minimum_frequency=fmin,
        )

        waveform_generator = WaveformGenerator(
            duration=duration,
            sampling_frequency=data_dump.meta_data['command_line_args']['sampling_frequency'],
            start_time=gps+post_trigger-duration,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

        likelihood_bilby = GravitationalWaveTransient(
            interferometers=data_dump.interferometers,
            reference_frame=data_dump.interferometers,
            waveform_generator=waveform_generator,
        )

        n_samples = 100
        samples_bilby = result_bilby.sample(n_samples, random_state=42)

        param_list = []
        for i in range(n_samples):
            sample = samples_bilby.iloc[i].to_dict()
            likelihood_bilby.parameters = sample
            logL_bilby = likelihood_bilby.log_likelihood_ratio()

            params = {}
            params["logL_bilby"] = logL_bilby

            params["M_c"] = sample["chirp_mass"]
            params["eta"] = sample["symmetric_mass_ratio"]
            params["s1_x"] = sample["spin_1x"]
            params["s1_y"] = sample["spin_1y"]
            params["s1_z"] = sample["spin_1z"]
            params["s2_x"] = sample["spin_2x"]
            params["s2_y"] = sample["spin_2y"]
            params["s2_z"] = sample["spin_2z"]
            params["iota"] = sample["iota"]
            params["d_L"] = sample["luminosity_distance"]
            params["t_c"] = sample["geocent_time"] - gps
            params["phase_c"] = sample["phase"]
            params["psi"] = sample["psi"]
            params["ra"] = sample["ra"]
            params["dec"] = sample["dec"]

            logL_jim = likelihood.evaluate(params, None)

            params["logL_jim"] = logL_jim
            params.pop("gmst", None)
            params['event'] = event_id
            param_list.append(params)
        event_dict[event_id] = param_list

        differences = [x["logL_jim"] - x["logL_bilby"] for x in param_list]
        max_abs_diff = max(abs(min(differences)), abs(max(differences)))
        if max_abs_diff > event_max_abs_diff:
            event_max_abs_diff = max_abs_diff
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            [x["logL_bilby"] for x in param_list], 
            [x["logL_jim"] for x in param_list], 
            c=differences,
            cmap='RdBu_r',
            vmin=-max_abs_diff,
            vmax=max_abs_diff
        )
        
        logL_min = min([x["logL_bilby"] for x in param_list] + [x["logL_jim"] for x in param_list])
        event_logL_min = min(event_logL_min, logL_min)
        logL_max = max([x["logL_bilby"] for x in param_list] + [x["logL_jim"] for x in param_list])
        event_logL_max = max(event_logL_max, logL_max)
        plt.plot([logL_min, logL_max], [logL_min, logL_max], color="black", linestyle="--", label="1:1")
        plt.xlim(logL_min, logL_max)
        plt.ylim(logL_min, logL_max)

        plt.xlabel(r"$\log\mathcal{L}_{\text{bilby}}$")
        plt.ylabel(r"$\log\mathcal{L}_{\text{jim}}$")
        plt.title(event_id)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(r"$\log\mathcal{L}_{\text{jim}} - \log\mathcal{L}_{\text{bilby}}$")
        
        plt.savefig(f"figures/{event_id}_likelihood_comparison.jpg", dpi=300, bbox_inches="tight")
        plt.close()

    with open("likelihood_comparison.pkl", "wb") as f:
        pickle.dump(event_dict, f)

else:
    with open("likelihood_comparison.pkl", "rb") as f:
        event_dict = pickle.load(f)
    event_max_abs_diff = 15 # manually set
    event_logL_min = 0
    event_logL_max = 350

for event_id in events:
    plt.scatter(
        [x["logL_bilby"] for x in event_dict[event_id]], 
        [x["logL_jim"] for x in event_dict[event_id]], 
        c=[x["logL_jim"] - x["logL_bilby"] for x in event_dict[event_id]],
        cmap='RdYlBu',
        vmin=-event_max_abs_diff,
        vmax=event_max_abs_diff,
        marker="."
    )
    plt.xlabel(r"$\log\mathcal{L}_{\text{bilby}}$")
    plt.ylabel(r"$\log\mathcal{L}_{\text{jim}}$")
    plt.xlim(event_logL_min, event_logL_max)
    plt.ylim(event_logL_min, event_logL_max)
plt.plot([event_logL_min, event_logL_max], [event_logL_min, event_logL_max], color="black", linestyle="--", label="1:1")
plt.colorbar()
plt.savefig("figures/likelihood_comparison.jpg", dpi=300, bbox_inches="tight")
plt.close()
