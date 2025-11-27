
from datasets import load_dataset

ds = load_dataset("Isma/librispeech_tiny")
ten_nm = ds["10mn"]

print(ten_nm.column_names)
print(ten_nm.features)

waveform = ds["10mn"][0]["audio"]["array"]  # numpy array
sr = ds["10mn"][0]["audio"]["sampling_rate"]
print(waveform, sr)