from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor
from datasets import load_dataset, Audio
from src.ModelWrapper import ModelWrapper
import torch
from pyaml_env import parse_config

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_wrapped = ModelWrapper(model)

ds = load_dataset("Isma/librispeech_tiny")
ds = ds["1h"]

def contains_word(example, word):
    return word in example["text"].lower().split()

def transcribe(ds):
    result_list = []
    for sample in ds:
        audio = sample["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.to(model.device)

        generated_ids = model.generate(input_features, language="en", task="transcribe")
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result_list.append(text)
    return result_list

pair_homophones = ds.filter(
    lambda ex: contains_word(ex, "pair") or contains_word(ex, "pear")
)

# pair_homophones consists of only one element which is ""

target = " pair"
foil = " pear"

CORRECT_ID = processor.tokenizer(target, add_special_tokens=False).input_ids[0]
FOIL_ID = processor.tokenizer(foil, add_special_tokens=False).input_ids[0]

input_features = None
for sample in pair_homophones:
    audio = sample["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features.to(model.device)

generated_ids = model.generate(input_features, language="en", task="transcribe")
decoded_ids - generated_ids[0]
decoded_text = processor.decode(decoded_ids, skip_special_tokens=True)
decoded_tokens = processor.tokenizer(
    decoded_text,
    add_special_tokens=False
).input_ids

target_pos = decode_tokens.index(CORRECT_ID)

# decoder prefix
decoder_input_ids = torch.tensor(
    decoded_tokens[:target_pos],
    device=model.device
).unsqueeze(0)

outputs = model(
    input_features=input_features,
    decoder_input_ids=decoder_input_ids,
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True,
)

logits = outputs.logits
hidden_states = outputs.decoder_hidden_states
attentions = outputs.cross_attentions

step_logits = logits[0, -1]
logit_diff = step_logits[CORRECT_ID] - step_logits[FOIL_ID]

logit_trans_vect_dict, logits_modules, layer_alti_data = (
    model_wrapped.get_logit_contributions(
        hidden_states,
        attentions,
        token
    )
)

contrastive_contributions = (
    layer_alti_data[CORRECT_ID] - layer_alti_data[FOIL_ID]
)