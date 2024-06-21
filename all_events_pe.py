from single_event_pe import runSingleEventPE

from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets
import json


gwtc3 = datasets.find_datasets(type='events', catalog='GWTC-3-confident')

output_dir = 'gwtc3_pe'

file = open('configs_2.json')
configs = json.load(file)['configs']

for event in gwtc3:
    try:
        event_name = event
        event_config = configs[event_name]
        runSingleEventPE(
            output_dir=output_dir,
            event=event_name,
            gps=event_config['gps'],
            duration=float(event_config['duration']),
            post_trigger_duration=2,
            Mc_prior=[event_config['Mc_prior']['min'], event_config['Mc_prior']['max']],
            ifos=event_config['detectors'],
            waveform="RippleIMRPhenomPv2",
            heterodyned=True
            )
    except Exception as e: print(e)




