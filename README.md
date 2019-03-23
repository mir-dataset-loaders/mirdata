# mir_dataset_loaders
common loaders for mir datasets


## Examples

### List available datasets
```python
import mir_dataset_loaders as mdl

mdl.list_datasets()
```

### Download the Orchset Dataset
```python
import mir_dataset_loaders as mdl

mdl.orchset.dowload()
```

### Load the Orchset Dataset
```python
import mir_dataset_loaders as mdl

orchset_data = mdl.orchset.load()
```

### Evaluate a melody extraction algorithm on Orchset
```python
import mir_eval
import mir_dataset_loaders as mdl
import numpy as np
import sox

def very_bad_melody_extractor(audio_path):
    duration = sox.file_info.duration(audio_path)
    time_stamps = np.linspace(0, duration, 0.01)
    melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
    return time_stamps, melody_f0

orchset_scores = {}
orchset_data = mdl.orchset.load()
for track_id, track_data in orchset_data.items():
    est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

    ref_melody_data = track_data.melody
    ref_times = ref_melody_data.times
    ref_freqs = ref_melody_data.frequencies

    score = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
    orchset_scores[track_id] = score

```