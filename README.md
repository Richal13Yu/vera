<h1 align="center">Turning Video Models into Generalist Robot Policies</h1>

<p align="center">
  Sizhe Lester Li<sup>*</sup>,
  Evan Kim<sup>*</sup>,
  Xingjian Bai<sup>*</sup>
</p>
<p align="center">
  Tong Zhao,
  Tao Pang,
  Max Simchowitz,
  Vincent Sitzmann
</p>

<p align="center"><sup>*</sup>equal contribution</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.27817">[Paper]</a> &nbsp;·&nbsp;
  <a href="https://vera.csail.mit.edu/">[Project Page]</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/sizhe-lester-li/VERA">[Models &amp; Data]</a>
</p>

https://github.com/user-attachments/assets/4d5d7325-43df-43e8-ae25-222e3b2c5417

**VERA** (**V**ideo-to-**E**mbodied **R**obot **A**ction model) is a **two-stage**, closed-loop
video-to-action policy. It leaves a video generative model **as-is** as an action-free world model that
"dreams" the future, and trains an embodiment-specific **inverse-dynamics model (IDM)** — built on the
robot's **Jacobian** — to translate that dream into actions:

1. **Video planner** (`vera.video_model` / `vera.idm.dfot`)
2. **Jacobian IDM** (`vera.idm` + `vera.policy`)

**Start here:** [Install](#install) → [Run VERA](#run-vera) (each task is one server command + one
notebook). To see the planner alone — no simulator, no robot — jump straight to the
[DROID generation walkthrough](#droid-video-generation-from-language-no-sim-no-robot).

---

<a id="news"></a>
## 🔥 News

- **Aug 2026 — PushT training data released.** Both packed training sets now ship on
  [HF](https://huggingface.co/sizhe-lester-li/VERA): `pusht-packed/` (206 teleop episodes, ~1.9 GB) and
  `pusht-noise-packed/` (18,685 random-exploration episodes, ~61 GB) — JPEG frames + MegaFlow optical
  flow. Setup: [TRAINING.md](TRAINING.md); regeneration: `scripts/data/pack_pusht.py` +
  [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md).
- **Jul 2026 — DROID: language-conditioned video generation.** The 14B DROID WAN planner + a
  self-contained notebook ([walkthrough](#droid-video-generation-from-language-no-sim-no-robot)) — watch
  the planner follow different language prompts from real multi-camera context, no robot required.
- **Jun 2026 — Wave 1.** MimicGen + PushT: full code, checkpoints, serving stack, and client notebooks.

---

<a id="install"></a>
## Install

VERA targets **Python 3.11** + **PyTorch 2.6 (CUDA 12.4)**. Self-contained — no sibling repos on `sys.path`.

```bash
git clone git@github.com:sizhe-li/VERA.git && cd VERA
pip install -e ".[idm,video]"            # the two stages (IDM + video planner)
pip install -e ".[eval]"                 # simulators: gymnasium, gym-pusht, robomimic, robosuite, mimicgen, mujoco
```

Notes:

- **PushT** rollouts seed initial states from the original replay buffer `pusht_cchi_v7_replay.zarr`
  ([Diffusion Policy](https://github.com/real-stanford/diffusion_policy) release):
  ```bash
  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip && unzip pusht.zip
  ```
  Point the notebook's `ZARR_PATH` at `.../pusht/pusht_cchi_v7_replay.zarr`.
- **MimicGen** needs the task dataset HDF5 for initial states (e.g. `stack_d0.hdf5`) from
  [🤗 `amandlek/mimicgen_datasets`](https://huggingface.co/datasets/amandlek/mimicgen_datasets).
- **VGGT** (IDM backbone for both Wave-1 IDMs) installs with the `idm` extra as a git dependency; if
  git installs are blocked: `pip install "git+https://github.com/facebookresearch/vggt.git"`. Weights
  pull from `facebook/VGGT-1B` on first use.
- **flash-attn** is optional — the WAN path falls back to SDPA if absent.

Verify: `python -c "import vera, vera.policy, vera.idm, vera.server; print('vera ok')"`

---

<a id="run-vera"></a>
## ⚡ Run VERA

Every sim task runs the **same two steps**: start a policy server in one terminal, run its client
notebook in another. The notebook drives the sim, prints the success rate, and inlines rollout videos.

```
  Terminal 1 — server                         Jupyter — client notebook
  ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │ python -m vera.server        │ ───────────▶ │ open the notebook → Run All  │
  │   .start_vera_server ...     │  :8800/:8820 │ → success rate + videos      │
  └──────────────────────────────┘              └──────────────────────────────┘
```

| Task | Server | **Client notebook** |
|---|---|---|
| **PushT** — planar push-to-goal | `--embodiment pusht` | `examples/pusht_dfot_stack.ipynb` |
| **MimicGen** — 2-block stacking | `--embodiment mimicgen` | `examples/mimicgen_stack.ipynb` |
| **DROID** — video generation from language | *(no server)* | `examples/droid_generation.ipynb` |

### PushT (DFoT planner — small, loads in seconds)

```bash
python -m vera.server.start_vera_server --embodiment pusht --port 8820 --vis-port 8821
```
Then open **`examples/pusht_dfot_stack.ipynb`** → **Run All**.

- rolls out the walkthrough's default start state (set `FRAME_INDICES = None` for a population success
  rate), prints the result, and inlines the rollout + composite policy-vis;
- checkpoint paths come from the `VERA_PUSHT_*` env vars (see `vera/server/start_server_pusht.py`);
- plans 3 future frames per replan, executes 2 (`VERA_PUSHT_ACTION_CHUNK_HORIZON=3`,
  `VERA_PUSHT_N_ACTION_STEPS=2`, env-overridable);
- **reproducing the paper's success rate** (exact state list, recipe, verified numbers, video viewer):
  [`docs/PUSHT_REPRODUCTION.md`](docs/PUSHT_REPRODUCTION.md).

### MimicGen two-block stacking (WAN planner)

```bash
export VERA_WAN_CKPT_ROOT=/path/to/Wan2.1-T2V-1.3B            # frozen Wan2.1 base (text-enc + VAE)
export VERA_MIMICGEN_CKPT_DIR=./vera-ckpts/mimicgen-wan-1.3b  # specialist DiT + flow decoder
python -m vera.server.start_vera_server --embodiment mimicgen --port 8800 --vis-port 8801 \
    --algo-config $VERA_MIMICGEN_CKPT_DIR/algo_config.yaml \
    --text "A robot arm stacks one block on top of another block"
```
Then open **`examples/mimicgen_stack.ipynb`** → **Run All**.

- set **both** env vars before launching; the Jacobian IDM loads via `VERA_MIMICGEN_DYNAMICS_CKPT`
  (default `./vera-ckpts/idm-mimicgen-285ouq1q/model.ckpt`);
- swap pieces live via `VERA_DYNAMICS_RUN_ID`, `VERA_TRACKER_BACKEND`, `VERA_MOTION_PLAN_SCALE`,
  `VERA_N_ACTION_STEPS`.

### DROID: video generation from language (no sim, no robot)

The planner by itself: the notebook continues real multi-camera context frames under different language
prompts and the generated futures follow (executed outputs ship in the notebook, so you can inspect
before running anything). Single GPU, ~60 GB VRAM (bf16); no server.

```bash
export VERA_DROID_CKPT_DIR=./vera-ckpts/wan-droid-14b       # DROID WAN planner (DiT + algo_config.yaml)
export VERA_WAN14B_CKPT_ROOT=/path/to/Wan2.1-I2V-14B-480P   # frozen Wan2.1 base (text-enc + VAE + CLIP)
```
Then open **`examples/droid_generation.ipynb`** → **Run All**.

- two experiments on the bundled clips (`examples/droid_demo_videos/`, three synchronized DROID
  cameras): one prompt from five start times, and three prompts from one start frame;
- **have a DROID setup?** Swap in short clips from your own cameras for a zero-shot check of how the
  planner generalizes to your scene — before anything runs on the robot.

---

## Live viewer — watch the policy think

Pass `--vis-port` to any server and open `http://localhost:<vis-port>/` for a dashboard that streams
VERA's **entire two-stage pipeline live** as the rollout runs — the policy is interpretable by
construction, not a black box:

![VERA live viewer](docs/assets/viewer.png)

Each row is one camera view, read left → right:

| Panel | What it shows |
|---|---|
| **Current** | the robot's live observation |
| **Dream + tracks** | the video model's predicted future, with motion tracks overlaid |
| **Dream** | the decoded future frames |
| **Jacobian field** | the map that turns the dream into the next action |

The per-chunk player scrubs each dream chunk frame-by-frame. The notebooks inline the same composite via
`show_policy_vis()`; snapshot with `python -m vera.server.save_vis_video --output dream.mp4`.

---

## Checkpoints & data

Hosted at [`huggingface.co/sizhe-lester-li/VERA`](https://huggingface.co/sizhe-lester-li/VERA). VERA hosts
only the **trained** artifacts and **training data**; frozen upstream pieces pull from their original homes.

| Group | dir | what |
|---|---|---|
| **Planners** | `mimicgen-wan-1.3b/` | MimicGen specialist WAN planner (DiT-only bf16, ~2.8 GB) + flow decoder + `algo_config.yaml` |
| | `pusht-dfot/` | PushT DFoT flow planner (~39 MB) + `run_config.yaml` |
| | `wan-droid-14b/` | DROID WAN planner (DiT-only bf16, ~31 GB) + `algo_config.yaml` |
| | `omni-wan/` | cross-embodiment OMNI WAN planner (DiT-only bf16, ~33 GB) — Wave 2 |
| **Jacobian IDMs** | `pusht-idm/` | PushT IDM (~232 MB) — reproduction: [`docs/PUSHT_REPRODUCTION.md`](docs/PUSHT_REPRODUCTION.md) |
| | `idm-mimicgen-285ouq1q/` | MimicGen IDM, VGGT-based (~11.3 GB) — the serving default |
| | `idm-mimicgen/` | MimicGen IDM, DPT variant (~230 MB) |
| | `idm-droid/` | DROID IDM, VGGT-based (~5.1 GB) — Wave 2 |
| **Training data** | `pusht-packed/` | packed PushT training set (206 episodes, ~1.9 GB) — [TRAINING.md](TRAINING.md) |
| | `pusht-noise-packed/` | packed PushT noise set (18,685 episodes, ~61 GB) — [TRAINING.md](TRAINING.md) |
| **Demo assets** | `droid-demo-clips/` | multi-view robot clips for the generation walkthrough (~100 MB) |
| **Upstream** | `Wan-AI/Wan2.1-T2V-1.3B` · `Wan-AI/Wan2.1-I2V-14B-480P` · `facebook/VGGT-1B` | WAN bases + IDM backbone (not re-hosted) |

**Download** (`pip install -U huggingface_hub` — **1.x or newer required**: older versions
(≤0.36) silently truncate this repo's large file listing, so `--include` patterns and full
downloads match nothing or miss folders; the symptom is `Fetching 0 files`):

```bash
# (1) Wave-1 only — everything the MimicGen + PushT notebooks need              (~15 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts \
  --include "mimicgen-wan-1.3b/*" "idm-mimicgen-285ouq1q/*" "idm-mimicgen/*" \
            "pusht-dfot/*" "pusht-idm/*"

# (2) + the DROID generation walkthrough                                        (~46 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts \
  --include "mimicgen-wan-1.3b/*" "idm-mimicgen-285ouq1q/*" "idm-mimicgen/*" \
            "pusht-dfot/*" "pusht-idm/*" "wan-droid-14b/*" "droid-demo-clips/*"

# (3) everything — all planners, IDMs, and both training-data packs             (~136 GB)
hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts
```

Then point the server/notebook at the downloaded paths (`--algo-config`, `VERA_PUSHT_*`,
`VERA_WAN_CKPT_ROOT`).

---

## Training

Both stages train through one Hydra entry point, `python -m vera.main` — see **[TRAINING.md](TRAINING.md)**
for the full guide (getting the training data, data format, IDM training, WAN / OMNI video-planner
finetuning, multi-GPU/FSDP), and **[docs/DATA_GENERATION.md](docs/DATA_GENERATION.md)** for regenerating
the packed datasets from source.

The cross-embodiment **OMNI** planner trains on a weighted mixture of Allegro-Sim + Allegro-Real +
MimicGen + DROID (native fps/aspect, black-padded to a 576-wide multiview canvas); PushT currently uses
its own DFoT planner. The 5-environment mixture config ships in
`vera/configurations/config_wan_combined_5env.yaml`.

<a id="omega-warp-training"></a>

### Omega-Warp Jacobian: data preparation and multi-GPU training

This fork trains the **VGGT-Omega student backbone and Jacobian decoder**
(`algorithm.model.freeze_aggregator=false`). The separate **Omega-Warp teacher,
including its backbone and warp head, stays frozen** and generates flow/confidence
labels online. The student receives only current observations, not future images.
See [the implementation and evaluation notes](docs/OMEGA_WARP_JACOBIAN.md).

**1. Clone the pinned Omega dependency and install.** Run subsequent commands from
this repository's root, using Python 3.11:

```bash
git clone --recurse-submodules https://github.com/Richal13Yu/vera.git
cd vera
# For an existing clone:
git submodule update --init --recursive

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[idm,video]" -e ./vggt-omega-warp \
  "numpy<2" "zarr>=3,<3.1" tomli-w pytest
```

The Omega submodule is pinned to `9bf088b2cbb84f359e30768631a4a39bae44091c`.
Keep this version for the existing teacher checkpoint's source fingerprint.
The NumPy/Zarr constraints reconcile Omega's `numpy<2` requirement with VERA.
The root package pins its PyTorch version; use a CUDA runtime compatible with your GPU.

**2. Download the nine official MimicGen source tasks (~15.9 GB).**

```bash
python scripts/data/download_mimicgen.py \
  --revision 33016f8a62c02334f929f2913af8fdd2a8a129e1 \
  --output-root data/mimicgen_raw --workers 3
```

Source: [`amandlek/mimicgen_datasets`](https://huggingface.co/datasets/amandlek/mimicgen_datasets),
`core/`. Tasks: `coffee_d0`, `coffee_d1`, `square_d0`, `square_d1`, `square_d2`,
`stack_d0`, `stack_d1`, `stack_three_d0`, `stack_three_d1` — **1,000 demonstrations
each, 9,000 total**. The downloader resumes partial HTTP downloads, verifies each
file's SHA-256 against the source manifest, and writes `DOWNLOAD_COMPLETE.json`.
Re-running verifies and skips completed files. The command above uses the exact
source revision of the existing experiments.

**3. Convert RGB and robot trajectories (~11 GB additional disk space).**

```bash
python -m scripts.data.pack_mimicgen \
  --source-root data/mimicgen_raw \
  --output-root data/datasets/jacobian/mimicgen_official_rgb_warp \
  --workers 3
```

The pack contains both recorded cameras (`agentview_image`, `robot0_eye_in_hand_image`)
and end-effector/gripper observations; VERA derives its normalized actions from
those observations. Source RGB is **84×84**, resized to **128×128** by the loader.
The converter preserves the task/demo order, supports restarting, and writes the
complete `index.json` after all selected tasks finish.

This is **not** the author's `mimicgen_packed_v3_megaflow` preprocessing package.
It contains no MegaFlow labels; Omega-Warp supplies labels during training.
The original packed data was unavailable for these experiments. Budget space for
both source and converted data, plus model weights and optimizer checkpoints.

**4. Supply the Omega backbone and your trained Warp head.** These large files are
not stored in Git. Copy `vggt_omega_1b_512.pt` and the supplied Warp `best.pt` to the
new machine, then export their local paths:

```bash
export VERA_OMEGA_CHECKPOINT=/absolute/path/to/vggt_omega_1b_512.pt
export WARP_SOURCE_CHECKPOINT=/absolute/path/to/best.pt

python scripts/prepare_omega_warp_teacher.py "$WARP_SOURCE_CHECKPOINT" \
  --backbone "$VERA_OMEGA_CHECKPOINT" \
  --output-dir outputs/omega_warp_preflight

export VERA_WARP_CHECKPOINT="$PWD/outputs/omega_warp_preflight/best_inference.pt"
export VERA_WARP_CONFIG="$PWD/outputs/omega_warp_preflight/teacher.toml"
export VERA_MIMICGEN_ROOT="$PWD/data/datasets/jacobian/mimicgen_official_rgb_warp"
```

Preparation checks the backbone SHA-256 against the Warp checkpoint, preserves
every head tensor, and exports a weights-only inference file and matching TOML.
The existing experiment's backbone SHA-256 is
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
Re-export these environment variables in each training shell.

**5. Check the setup, then launch on multiple GPUs.**

```bash
# Two-example pack and two training steps, using the unfrozen backbone.
python -m scripts.data.pack_mimicgen --tasks stack_d0 --limit-per-task 2 \
  --workers 1 --output-root outputs/omega_warp_preflight/pack_smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/train_mimicgen_omega_warp.sh smoke

# Single-node, four-GPU DDP. Lightning launches one process per visible GPU.
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mimicgen_omega_warp.sh train
```

Use the launcher once; do not additionally wrap this command in `torchrun`.
Each GPU has its own student, optimizer state, and frozen teacher; DDP does not
combine GPU memory. Full-backbone training needs more memory per GPU than the
historical frozen run. A real full-size multi-GPU run must be checked on the target
machine; the unit tests alone do not establish its memory requirements.

Defaults remain **batch size 1 per GPU**, 8 frames, two cameras, 128×128 RGB,
BF16 mixed precision, AdamW learning rate **5e-5**, accumulation **1**, and at most
**600,000 total optimizer steps**. Inverse-action, flow-gradient, and Jacobian-TV
training weights remain zero. Four GPUs therefore give global batch size **4**;
this changes the examples processed per step compared with the old single-GPU run.
The only model-training switch changed from that run is unfreezing the student backbone.
To reproduce its freezing behavior, append `algorithm.model.freeze_aggregator=true`.

Runs save the Hydra config and checkpoints under `outputs/mimicgen_omega_warp/`.
`VERA_PYTHON` can select another Python executable; otherwise the launcher uses
`.venv/bin/python`. W&B is disabled by default. Validation retains the nine pinned
MimicGen tasks every 1,000 steps; these episodes are also in the training pool,
so this is not a held-out evaluation or a robot success-rate measurement.

**Continue a full-backbone run** by copying its checkpoint and passing:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mimicgen_omega_warp.sh train \
  'load="/absolute/path/to/last.ckpt"'
```

For a checkpoint from the **older frozen-backbone run**, first copy that run's
`.hydra/config.yaml` as well and migrate its optimizer parameter list:

```bash
python scripts/unfreeze_omega_checkpoint.py \
  /absolute/path/to/frozen-last.ckpt /absolute/path/to/unfrozen-resume.ckpt \
  --config /absolute/path/to/frozen-run/.hydra/config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_mimicgen_omega_warp.sh train \
  'load="/absolute/path/to/unfrozen-resume.ckpt"'
```

Migration preserves student weights, decoder AdamW moments, learning rate, and
training step, and initializes moments only for the newly unfrozen backbone.
The Warp teacher remains separate and unchanged. Export the new machine's
`VERA_*` paths before migration. A continued run stops at the configured **total**
step limit, not 600,000 additional steps.

---

## 🗺️ Release roadmap

_Last updated: **Aug 17, 2026**. The repo contains the unified code for **all** embodiments; the lists
below track what is documented end-to-end._

**Ready today**

- **MimicGen** (Panda, 2-block stacking) — checkpoints · serving · notebook · **training data** &nbsp;*(Jun 2026)*
- **PushT** (planar pusher) — checkpoints · serving · notebook · **training data + reproduction guide** &nbsp;*(Jun–Aug 2026)*
- **DROID video generation** — 14B WAN planner + walkthrough notebook, no robot required &nbsp;*(Jul 2026)*
- **DROID policy serving** (FR3 real) — checkpoints · [serving walkthrough](docs/DROID_SERVING.md) ·
  example client, robot-free validation included &nbsp;*(Aug 2026)*

**In progress**

- **Allegro-Sim / Allegro-Real / IIWA-Sim** — code in-tree; simulators + docs coming (the `eval` extra
  currently covers the MimicGen + PushT environments)

---

## Acknowledgements

This work was supported by the National Science Foundation under Grant No. 2211259, by the Intelligence
Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC)
under 140D0423C0075, by the Amazon Science Hub, by the MIT-Google Program for Computing Innovation, by
Advanced Micro Devices, Inc. under the AMD University Program's support of the MIT Hardware Consortium, and
by a 2025 MIT Office of Research Computing and Data Seed Grant.

## License & Citation

Released under the **MIT License** (see `LICENSE`); depended-upon code retains its own license (see
`NOTICE`). VERA builds on **Wan2.1** (Apache-2.0), **VGGT** (Meta), **CLIP/open_clip** (MIT), and
**cotracker/AllTracker**; the DFoT/DiT backbones are adapted from `facebookresearch/DiT` and `NVlabs/edm2`.

**Checkpoint licenses:** the hosted weights are Apache-2.0, except `idm-droid/` and
`idm-mimicgen-285ouq1q/`, which bundle the **VGGT-1B** backbone weights
([CC-BY-NC-4.0](https://huggingface.co/facebook/VGGT-1B)) and are therefore **non-commercial**.
Per-checkpoint details are on the [HF model card](https://huggingface.co/sizhe-lester-li/VERA).

```bibtex
@article{li2026turningvideomodelsgeneralist,
      title={Turning Video Models into Generalist Robot Policies}, 
      author={Sizhe Lester Li and Evan Kim and Xingjian Bai and Tong Zhao and Tao Pang and Max Simchowitz and Vincent Sitzmann},
      year={2026},
      eprint={2605.27817},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.27817}, 
}
```
