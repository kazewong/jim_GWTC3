# Parameter Estimation with JIM
## How to run
To run a single event: <br />
```
python single_event_pe.py
```
<br />

To generate a configuration file: <br />
```
python gen_config.py
```
<br />

To run all events in gwtc3: <br />
```
python all_events_pe.py
```
<br />

To generate a summary report for the PE run: <br />
```
python gen_summary.py
```


## Tasks
* single event parameter estimation
- [x] Be able to run parameter estimation for a single events
- [x] Getting reasonable posterior for high SNR events (comparable to BILBY)
- [x] Implement JL-divergence function to evaluate the similarity between two distribution
- [ ] Create a prior class for chirp mass that is using the assumption of uniform component masses

* GWTC-3 events parameter estimation
- [x] Generate a configuration file automatically for initiating PE run
- [x] Build a automatic programme which runs PE on GWTC-3 events
- [x] Generate a output .xlsx file which summarizes the PE run
- [ ] Incorporate single run event manager


