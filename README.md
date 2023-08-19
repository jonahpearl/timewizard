

![Videochef logo](/docs/_static/logo.png)

Timewizard is a Python library that makes it easier to work with timeseries data, especially in the context of neuroscience. 

To use timewizard, your data **must** meet the following minimal (and hopefully straightforward) standards:
* have time along the 0-th axis
* be sorted along the time axis
* be convertible to numpy arrays

Timewizard **can** help you cast the following spells:
* align data sampled at different rates
* align matched events across two different data streams
* collect peri-event traces of any-dimensional data
* enumerate peri-event events (i.e. lick times relative to a stimulus)
* describe stimulus trains (i.e. onsets and durations of complex optogenetic stimuli)

Timewizard **will not** help you to:
* synchronize your data (i.e. make sure that some time t=t0 refers to the same real moment in all your timeseries). This is your responsibility! (Although timewizard can help you verify that you've done it correctly.)
* plot your data (there are libraries for that!)
* run downstream analysis

Timewizard also provides a convenient collection of cute charms (achem -- utilities):
* round to a multiples (i.e. nearest multiple of 10)
* check if a matrix is symmetric
* check if an array is sorted
* describe runs of values in your data
* hierarchically cluster a distance or correlation matrix

Timewizard's core alignment functionality is modified from code originally written by the folks at the Allen Institute for their [Brain Observatory data](https://github.com/AllenInstitute/brain_observatory_utilities). Functions modified directly from their code are licensed under the Allen Institute License. All other code is provided under the MIT license.

## Why?
Timewizard provides mid-level, modality-agnostic functions that help you work with timeseries. 
* Everything is done with binary searches, meaning execution speed is very fast and doesn't require a GPU.
* No interpolation until you need it.
* Never use for-loops to find temporal epochs and events ever again!

If you're looking for help running standard neuroscience analyses, you might check out [pynapple](https://github.com/pynapple-org/pynapple/tree/main).

## Install

For now, clone the repo and use `pip install -e .` from inside the repo.

## Examples

-- Extract the location of an animal at specific event times (i.e. vocalizations). No interpolation necessary! (Therefore faster, easier, and no fooling yourself into thinking you have more temporal resolution than you really do.)
```python
# Make up some data
data_timestamps = np.arange(0,2*np.pi,0.033)  # eg, 30 Hz video
x = np.cos(t)  # say the animal is moving in a circle
y = np.sin(t)
event_times = [np.pi, 7*np.pi/4, 30]

# Map the animal's position to each event
event_idx = tw.index_of_nearest_value(data_timestamps, event_times)
event_idx = event_idx[event_idx != -1]  # remove -1's, which indicate event times that were outside of the range of the data
event_locs = np.hstack([x[event_idx].reshape(-1,1), y[event_idx].reshape(-1,1)])

# Show the results
plt.plot(x,y)
plt.scatter(event_locs[:,0], event_locs[:,1])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.axis('square')
offset = 0.05
plt.text(-1 + offset, 0, 'pi')
plt.text(0.707 + offset, -0.707, '7*pi/4')
plt.xlabel('X')
plt.ylabel('Y')

```
![image](https://github.com/jonahpearl/timewizard/assets/68478436/ad76c805-eed9-45fa-a9aa-f69291b87b31)

Note that the `event_times` do not have to correspond exactly to the times in `t`. They just have to be in the same units + reference frame.


-- Extract event onset and duration information from a series of booleans (e.g., extract bouts of fast running from a string of True/False's saying whether the animal was moving faster than some threshold):
```python
np.random.seed(10)
data_timestamps = np.arange(0, 10, 0.1)
speed = np.abs(np.sin(data_timestamps)) + tw.moving_average(np.random.normal(3, 3, data_timestamps.shape), 10, convolve_mode='same')
threshold = 5
speed_bool = speed > threshold
onset_idx, onset_times = tw.get_peristim_times(speed_bool, data_timestamps)
offset_idx, offset_times =  tw.get_peristim_times(speed_bool, data_timestamps, onsets_or_offsets='offsets')

plt.plot(data_timestamps, speed, label='Speed')
plt.hlines(threshold, *plt.xlim(), colors='k', linewidths=0.5, linestyles='--')
yl = plt.ylim()
for iBout, (t0,tf) in enumerate(zip(onset_times, offset_times)):
    label = 'Bouts' if iBout==0 else None
    plt.plot([t0, tf], [yl[1], yl[1]], 'r-', label=label)
plt.xlabel('Time (sec)')
plt.ylabel('Running speed')
plt.legend()
```
![image](https://github.com/jonahpearl/timewizard/assets/68478436/853de6d7-5a09-46d3-9296-1c2ca7451a8b)


-- Find all instances of some event type (e.g. saccades) that fall within the bounds of some other event (e.g. bouts of fast running):
```python
# say you have clean saccades from a recording
saccade_df = pd.DataFrame({  
    'start_time': np.random.uniform(0, 1000, size=(1000,)),
    'magnitude': np.random.uniform(0, 10, size=(1000,))
})
saccade_df = saccade_df.sort_values(by='start_time').reset_index()

# say you also have some fast running bouts from the previous example
running_df = pd.DataFrame({
    'running_bout_starts': [300, 400, 500, 600],
})
running_df['running_bout_ends'] = running_df['running_bout_starts'] + 5  # just making this simple...

# Find average saccade magnitude inside each running bout
all_saccade_times = saccade_df['start_time']
assert tw.issorted(all_saccade_times)  # timestamps MUST be sorted for timewizard funcs to work, in general
generator = tw.generate_perievent_slices(
    all_saccade_times,
    event_timestamps=running_df['running_bout_starts'],
    event_end_timestamps=running_df['running_bout_ends']
)
for _slice in generator:
    mags = saccade_df.loc[_slice, 'magnitude'].values
    running_df['avg_saccade_mag'] = np.mean(mags)
```
Again, note that the values in `all_saccade_times` doesn't have to correspond exactly to any values that show up elsewhere, so long as they're in the same units and reference frame.


-- Easy peri-stimulus heatmaps (it's a one liner!!)
```python
# Make some fake signal with something to see
np.random.seed(2)
fs = 4
t = np.arange(0, 1000, 1/fs)
window = (-2,4)
event_times = np.array([0, 100, 400, 600])  # requested window for first event will be before bounds of data
signal = np.zeros_like(t, dtype='float')
for t0 in event_times:
    _slice = slice(int(t0*fs), int(t0*fs + (window[1]-1)*fs))
    signal[_slice] += np.random.normal(0, 3, size=((window[1]-1)*fs))
event_times = np.hstack([event_times, 998])  # purposely add an event that will extend after the bounds of the data

# Get aligned traces
ts, traces = tw.get_aligned_traces(t, signal, event_times, time_window=window, fs=fs)

# Plot the data
plt.imshow(traces, aspect='auto', interpolation='none', cmap='coolwarm')
plt.vlines((0 - window[0]) * fs, *plt.ylim(), "w", linewidths=1, zorder=np.inf)
tw.xticks_from_timestamps(ts, interval=2, ax=plt.gca())
plt.xlabel('Time from event (sec)')
plt.ylabel('Event number')
plt.colorbar(label='Signal')
```
![image](https://github.com/jonahpearl/timewizard/assets/68478436/757d8ba8-b8d3-42b1-a700-9ffd2b6038fe)



## Roadmap
* allow mmap and lazy evaluation à la spike interface (and many others)
* re-factor edge case handling -- still a bit buggy and the code is sloppy
* 

## Citataions
Witch icons created by [Freepik - Flaticon](https://www.flaticon.com/free-icons/witch)

