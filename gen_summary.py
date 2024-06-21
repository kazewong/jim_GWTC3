from parameter_estimation.save import saveSummary
from os import listdir
from os.path import isfile, join

from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets

path = "output/posterior_samples"
files = [f[:-3] for f in listdir(path) if isfile(join(path, f))]

saveSummary(files)

gwtc3 = datasets.find_datasets(type='events', catalog='GWTC-3-confident')
missing_events = set(gwtc3) - set(files)
print("Missing events: ", missing_events)