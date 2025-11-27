from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import torchaudio
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model.to(device)

print(model.config.model_type)
decoder_layer = model.model
print(decoder_layer)
print(hasattr(model.model, "proj_out"))  # True
print(hasattr(model, "lm_head"))          # True

print(model.__class__.__name__)
print(model.model.__class__.__name__)

# Check what attributes exist
print("proj_out" in [n for n, _ in model.named_modules()]) #prints true
print("lm_head" in [n for n, _ in model.named_modules()]) # prints false



ds = load_dataset("Isma/librispeech_tiny")
ds = ds["10mn"]
#ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))

def prepare_batch(batch):
    print(ds[0]["audio"]["array"])
    waveform = ds[0]["audio"]["array"]
    sr = ds[0]["audio"]["sampling_rate"]
    batch["input_features"] = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features[0]
    return batch

dataset = ds.map(prepare_batch)


transcriptions = []
for sample in dataset:
   # input_features = sample["input_features"].unsqueeze(0).to(device)  # batch dim
    input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)

    generated_ids = model.generate(input_features)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    transcriptions.append(text)

# 6 Print results
for i, t in enumerate(transcriptions):
    print(f"Sample {i}: {t}")
