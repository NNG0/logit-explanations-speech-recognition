# Explaining How Transformers Use Context to Build Predictions for ASR

## Abstract
TODO

## Environment Setup
Create a conda environment and install requirements:
```bash
conda create -n alti python=3.10.9
conda activate alti
cd logit-explanations
pip install -r requirements.txt
```

## Data
TODO

## Usage with Transformers
It can be extended to other models in Huggingface's [transformers](https://github.com/huggingface/transformers "Huggingface's transformers github") library.

Add to `./src/config.yaml` your model with the required layers' names. For example, for Whisper model:
```yaml
  whisper:
    layer: 'model.model.decoder.layers'
    ln1: 'self_attn_layer_norm'
    ln2: 'final_layer_norm'
    values: 'self_attn.v_proj'
    dense: 'self_attn.out_proj'
    lnf: 'model.model.decoder.final_layer_norm'
    enc_values: 'encoder_attn.v_proj'
    enc_out: 'encoder_attn.out_proj'
    fc1: 'fc1'
    fc2: 'fc2'
    unembed: 'proj_out'
    pre_layer_norm: 'True'```

## Extract Linguistic Evidence
REWORK
Follow `extract_ling_evidence.ipynb` to create the data files with the linguistic evidence included (through spaCy).

We also provide the data with linguistic evidence included, as obtained with `extract_ling_evidence.ipynb`. It can be found in `./data`, in the folders ending with `with_targets`.

## Extract Explanations
```bash
python extract_explanations.py --name_path $model  \ # LM
                               --dataset $dataset \ # blimp subset, sva_$num_attractor or ioi
                               --explanation_type $method # ours/erasure/grad
```

## Explanations Notebook
In `explanations.ipynb` you can extract GPT-2, BLOOM, and OPT-125M explanations.

## Evaluation
`evaluation.ipynb`


## Citation
Code inspiered by the paper [Explaining How Transformers Use Context to Build Predictions](https://arxiv.org/pdf/2305.12535.pdf)
```bibtex
@misc{ferrando2023explaining,
      title={Explaining How Transformers Use Context to Build Predictions}, 
      author={Javier Ferrando and Gerard I. Gállego and Ioannis Tsiamas and Marta R. Costa-jussà},
      year={2023},
      eprint={2305.12535},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## development notes
- add documentation
- experiment setup
