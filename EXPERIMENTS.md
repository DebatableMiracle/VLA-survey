# VLA_survey

Companion repository for the survey **"From Seeing to Acting: A Survey of Vision-Language and Vision-Language-Action Models for Automated and Connected Vehicles"** by A. Attri, A. Verma, and G. Thandavarayan (under review).

The repository has two parts:

- [`REFERENCES.md`](REFERENCES.md) lists all 157 papers, standards, and reports cited in the survey, in citation order, with links.
- [`survey_code/`](survey_code) holds the code, prompts, sampled items, and raw model outputs of the case study in Section VI, together with tools that check the counts in Section III.

## Case study (Section VI)

The case study tests two things on DriveLM-nuScenes with three open VLMs, zero-shot:
- how much the answers depend on the image (experiment E1);
- how much latency step-by-step reasoning adds (experiment E2).

### Models (exact checkpoints)

| Model | Hugging Face checkpoint |
|---|---|
| Qwen2.5-VL-7B | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| InternVL3-8B | [`OpenGVLab/InternVL3-8B-hf`](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) |
| LLaVA-1.5-7B | [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |

### Environment used for the paper

- PyTorch 2.5.1 (CUDA 12.1) and Transformers 5.18.0.
- One NVIDIA A100-SXM4-80GB.
- bfloat16 weights and SDPA attention.
- Greedy decoding with batch size 1.
- At most 64 new tokens for direct answers and 512 for reasoning.

The exact settings are recorded in `survey_code/cs_final/meta.json`.

### Data

We use the DriveLM-nuScenes v1.1 training split (`v1_1_train_nus.json`, from [OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)) and the nuScenes camera images. Neither is redistributed here, so please download them under their own licenses.

### What is in `survey_code/`

| Path | Contents | Used in the paper for |
|---|---|---|
| `survey_tools.py` | All commands: `drivelm-stats`, `casestudy`, `merge`, `make-tables`, `summary`, `kappa-template`, `kappa`, `prisma-search`, `screen-kappa`, `prisma-count`, `check-links`, `readme`. | Everything below |
| `cs_final/` | The final results:<br>• `items.json`: sampled items, seed 0<br>• `prompts.json`<br>• `meta.json`<br>• `results.jsonl`: every raw response with its parsed label, token count, and latency<br>• `ci.csv`, `e2_stats.csv`, `diagnostics.txt` | Tables 7 and 8, Fig. 5, and every number in Section VI |
| `cs_llava_pad/` | Sensitivity run with the LLaVA-1.5 input padded to a square, so that it is not center-cropped | The 68.3% / 15.5-point result in Section VI-B |
| `cs_rerun/` | Repeat run of experiment E2 | The check that all 1,800 E2 responses repeat exactly, and the E2 block merged into `cs_final/` |
| `setup_models.sh`, `download_models.slurm` | Download the three checkpoints on a Slurm cluster | Setup |

### Reproducing the numbers

Install the packages and check the counts behind Section III. These steps need no GPU:

```bash
pip install "torch>=2.3" "transformers>=4.53" accelerate pillow numpy

# Counts behind Section III (43 systems; coupling and reasoning grid)
python survey_tools.py summary
```

Rebuild Tables 7 and 8 and the CIs from the released raw outputs (also no GPU):

```bash
python survey_tools.py make-tables --outdir cs_final
```

Recompute the target-action label distribution in Section V-C:

```bash
python survey_tools.py drivelm-stats --drivelm-json /path/to/v1_1_train_nus.json
```

Re-run both experiments from scratch. This needs one GPU with at least 24 GB:

```bash
python survey_tools.py casestudy --drivelm-json /path/to/v1_1_train_nus.json \
       --image-root /path/to/nuscenes --outdir cs_new
python survey_tools.py make-tables --outdir cs_new
```

Run the sensitivity check with padded LLaVA-1.5 input:

```bash
python survey_tools.py casestudy --drivelm-json /path/to/v1_1_train_nus.json \
       --image-root /path/to/nuscenes --models llava-1.5-7b --llava-pad-square --outdir cs_llava_pad_new
```

Note: `summary` groups systems by the year of the version cited in the paper. Section III-D of the paper counts by year of first public release (arXiv v1 or proceedings), so the per-year counts differ.

### Classification agreement (Section III-D)

To label the 43 systems with two independent annotators, first create the annotation sheets:

```bash
python survey_tools.py kappa-template --outdir kappa
```

This writes `kappa/annotator_A.csv`, `kappa/annotator_B.csv`, and `kappa/definitions.txt`. Each annotator fills in their sheet from the primary publications, without looking at Table 3. Then compute the agreement:

```bash
python survey_tools.py kappa --a kappa/annotator_A.csv --b kappa/annotator_B.csv
```

Pull requests that add papers are welcome. Please state the coupling class (I, M, D, U) and the reasoning regime (R0–R4) of the system you add.


