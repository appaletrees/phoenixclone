# Dual-arm Piper data interface

Run all commands from the repository root. This interface extends the
`post-train` branch; its `piper_posttrain` task inherits `robotwin_posttrain`'s
online training settings. Model architecture, losses, optimizer, learning rate,
batch size, training duration and sampling windows remain unchanged. The
specialization replaces benchmark-specific data and embodiment bindings.

## Sources and units

`configs/data/piper_sources.json` declares four original LeRobot v2.1 sources:

| Source | Numeric fields | Width | Joint units |
| --- | --- | --- | --- |
| `datasets/connector_plugin` | `observation.state`, `action` | 14 | degrees |
| `datasets/fold_towel_mit` | `state`, `actions` | 15 | radians |
| `datasets/piper_clean_table_lerobot` | `state`, `actions` | 15 | radians |
| `datasets/piper_set_placemat_lerobot` | `state`, `actions` | 15 | radians |

The current source plan removes the deleted `piper_fold_towel` folder and adds
table cleaning and placemat placement. Each source keeps its own instruction;
the deleted advantage-labeled towel data is not part of this snapshot.
Original paths and episode IDs remain traceable in the prepared manifest.

The first 14 coordinates are user-confirmed: left joints 1–6, left gripper,
right joints 1–6, right gripper. Units are inferred from the source numeric
ranges: maximum absolute joint values are approximately 119.24, 2.42, 2.60 and 2.58,
respectively. The declarations are explicit and reproducible, not re-inferred
on every training run. Change the source plan and prepare a new output directory
if acquisition metadata establishes different units.

Only the twelve joint coordinates are converted from degrees to radians.
Gripper fractions and their direction are retained. The fifteenth coordinate
is preserved as `observation.auxiliary` / `action.auxiliary`, with
`auxiliary_present` distinguishing absent source coordinates. Its physical
meaning has not been established; it is not treated as a robot joint, an episode
termination rule, or a canonical80 loss target. Original records remain intact.

No frames are shifted, resampled, removed or duplicated. All sources remain at
30 Hz. Each episode retains its task and source episode mapping.

## Prepare

```bash
python tools/prepare_piper_data.py --output resources/piper/dataset_v3
python tools/check_piper_data.py --dataset resources/piper/dataset_v3
```

Preparation refuses an existing destination. It creates:

```text
resources/piper/dataset_v3/
  data/chunk-000/episode_*.parquet
  videos/chunk-000/observation.images.cam_*/episode_*.mp4
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
  meta/normalization.json
  meta/piper_interface.json
  meta/source_episodes.json
resources/piper/interface_check.json
```

The updated data has 756 episodes, 693,620 frames and four distinct task texts.
The old `resources/piper/dataset` and `resources/piper/dataset_v2` snapshots are
not refreshed automatically and must not be used for the updated folder task. Video files
are relative symlinks into the original `datasets/` tree, without transcoding.
Copy both `datasets/` and `resources/piper/` to the server with the same layout
and preserve symlinks. The generated manifests/configuration contain no host
absolute paths. Dataset outputs remain Git-ignored under `resources/`.

The checker compares every projected numeric record and auxiliary value with
its source, verifies video links, decodes head/tail windows from each source,
and checks the action/state masks. It does not load a policy or use a GPU.
Missing T5 caches are listed separately; passing data checks does not assert
that all training resources are present.

## Model-facing contract

- Cameras: main, left wrist, right wrist, each independently resized/cropped by
  the original pipeline to 256 × 256.
- RGB: `[3 cameras, 3 channels, 9 frames, 256, 256]`, sampled at offsets
  `0,4,...,32` from a 33-timestamp window.
- Actions: `[32,80]`; proprioception: `[1,80]`.
- Native joint14 → canonical80: left joints `0:6`, left gripper `16`,
  right joints `40:46`, right gripper `56`. Other coordinates are zero and
  masked out of supervision.
- Z-score normalization is calculated from the prepared training membership,
  before canonical80 scatter. Do not substitute benchmark statistics.
- Existing dual-arm Piper registry entry `openneo_aloha_piper`, ID **14**,
  retains the complete pretrained 28-slot, eight-token prompt table.
- Tail padding repeats only the last frame/action of the same episode and
  masks padded targets using the original reader.

As in the online reference recipe, `val_set_proportion: 0` uses the training
pool for diagnostics. This is not a held-out validation score. The RoboTwin
periodic 50-of-550 selector is explicitly removed. By default all Piper episodes
are used; the source selector below restricts both readers to one source.

## Train each source separately

Use `task=piper_folder_posttrain` and set `data.source_root` to one original
source folder. This task requires a non-null folder selection and defaults to
`resources/piper/dataset_v3`. The reader uses
`meta/source_episodes.json` to select complete episodes before loading their
Parquet files and building windows. It retains original episode IDs and video
paths, even when selected episodes are not contiguous. Selecting a declared
source group includes all its child datasets; selecting a full leaf path also
works. Equal task texts do not merge separate folders. Training and diagnostic
readers must select the same source; an unknown source fails instead of falling
back to the combined pool. The source is saved in the run configuration and
checkpoint training contract.

| `data.source_root` | Episodes | Frames | Windows | Steps/epoch, 16 × batch 8 |
| --- | ---: | ---: | ---: | ---: |
| `datasets/connector_plugin` | 210 | 54,364 | 53,524 | 419 |
| `datasets/fold_towel_mit` | 225 | 185,446 | 184,546 | 1,442 |
| `datasets/piper_clean_table_lerobot` | 201 | 388,506 | 387,702 | 3,029 |
| `datasets/piper_set_placemat_lerobot` | 120 | 65,304 | 64,824 | 507 |

Folders remain independently selectable regardless of their instruction text
or task difficulty. Rebuild preparation after changing source data, using
a fresh output directory, or transfer the complete newly prepared snapshot with
the updated original data. Preparation computes normalization statistics from
the updated combined data using the same algorithm; each folder run uses that
snapshot's `meta/normalization.json`. The z-score/clipping procedure, model,
losses and hyperparameters are unchanged. No labels are repaired or removed.
The old `task=piper_posttrain data.source_root=null` remains available for the
original combined interface; the new folder task rejects a null selection.

Use the original direct training entry, selecting exactly one folder per run.
This command uses one node, 16 GPUs, batch 8 per GPU, accumulation 1, BF16 and
ZeRO-1. Each invocation strictly initializes from `pretrain_256013.pt` with a
fresh optimizer/scheduler. Choose a fresh output directory for another run.

```bash
cd /data/share/Phoenix-MWA
# connector_plugin | fold_towel_mit | piper_clean_table_lerobot | piper_set_placemat_lerobot
TRAIN_SOURCE=piper_clean_table_lerobot
TRAIN_EPOCHS=5
TRAIN_OUT="../../../efs/share/Phoenix-MWA/${TRAIN_SOURCE}_v3"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH=src:.
export DIFFSYNTH_MODEL_BASE_PATH=checkpoints
export WAM_PRETRAIN_CHECKPOINT=resources/hf/Phoenix-Pretrain/pretrain_256013.pt
export DINO_ENCODER_CHECKPOINT=resources/encoders/dino110k.pt
export CSVAE_CHECKPOINT=resources/encoders/csvae.pt
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HYDRA_FULL_ERROR=1

mkdir -p "$TRAIN_OUT"
set -o pipefail
python -u -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_joint96_posttrain_zero1.yaml \
  --num_machines 1 --num_processes 16 \
  --machine_rank 0 --main_process_ip 127.0.0.1 \
  --main_process_port 24491 --deepspeed_multinode_launcher standard \
  scripts/train.py task=piper_folder_posttrain \
  "data.source_root=datasets/$TRAIN_SOURCE" \
  "output_dir=$TRAIN_OUT" "num_epochs=$TRAIN_EPOCHS" max_steps=null \
  2>&1 | tee "$TRAIN_OUT/console.node0.log"
```

The relative output path above resolves under `/efs/share/Phoenix-MWA/` from
the shown working directory. Change `TRAIN_SOURCE` to choose the folder and
`TRAIN_EPOCHS` independently for its size/difficulty. No sequential training
loop or shell launcher is needed. Equal epochs do not imply equal optimizer
steps: five epochs yield 2,095, 7,210, 15,145 and 2,535 steps respectively for the four
rows above. These counts describe data volume, not a recommended optimum.
Checkpoint/evaluation cadence remains every
5,000 steps, plus an end-of-training checkpoint. Thus the first source saves
its checkpoint at the end when using five epochs. Weights are written under
each output directory's `checkpoints/weights/`; full training state is under
`checkpoints/state/latest`.

## Server preparation

The new task uses relative defaults for data, encoders, initialization weights
and caches. For T5 caches, use the project's existing generator on the server:

```bash
export DIFFSYNTH_MODEL_BASE_PATH=checkpoints
python tools/text_cache.py task=piper_folder_posttrain data.source_root=datasets/piper_clean_table_lerobot
python tools/check_piper_data.py --dataset resources/piper/dataset_v3
```

For the updated data use the folder-task command above. The generator writes
the four task embeddings under `resources/piper/text_cache`, reusing existing
valid entries and computing missing ones, including table cleaning and placemat
placement. Old cached prompts are not selected merely because their files exist.
It loads only the existing T5 encoder and tokenizer, not a Wan DiT or the Phoenix
policy. No dummy text tensors are installed by preparation or verification.

## Strict Phoenix initialization

The Piper training entry now constructs the same Joint96 model and restores
`resources/hf/Phoenix-Pretrain/pretrain_256013.pt` directly. It does not download
or load the official Wan DiT base shards. The saved task configuration keeps
its original model settings; the runtime selects checkpoint-only construction
for Piper and uses the existing strict Joint96 checkpoint loader.

All video/action expert tensors, the complete pretrained prompt table and the
proprioception encoders must match. Missing/unexpected keys, shape or semantic
metadata mismatches, additional uncovered trainable parameters, and missing
checkpoint files stop initialization. There is no random-weight fallback.
The strict restore completes before AdamW or DeepSpeed optimizer state is
created. This is a fresh fine-tuning run from pretrained weights, with a new
optimizer/scheduler and training step zero; it does not resume optimizer state.

Configuration paths remain relative. They are resolved internally for the
existing checkpoint loader without rewriting the configuration. The initialization
receipt records `initialization_mode: strict_phoenix_checkpoint` and
`wan_dit_base_loaded: false`. Wan VAE, DINO, CSVAE and the required T5 text caches
remain separate required resources. Other task initialization entries are unchanged.

Select `task=piper_posttrain` when using the project's training entry. This
change does not select local GPU settings, change distributed topology or start
training. The strict initialization entry is the only training behavior changed
after data-interface alignment; architecture, losses and hyperparameters remain
those of the inherited recipe. Validate CUDA/DeepSpeed on the server.

CPU regression checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The direct-loading tests cover real tensor restoration and an identical first
optimizer update on a small fixture, rejection of incomplete/incompatible
checkpoints before optimizer creation, and all 1,653 released policy tensor
keys/shapes through the actual factory/strict loader on the meta device. The
meta check uses stand-ins for frozen encoders; it is not a CUDA training run.
