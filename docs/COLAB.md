# Training on Google Colab A100

Open `train_on_colab.ipynb` in Colab and select **Runtime > Change runtime type > A100 GPU**.
The notebook checks the assigned GPU and its actual VRAM; a paid subscription does
not guarantee A100 availability. By default it requires one A100 with at least
39 GiB of reported memory (approximately the 40 GB model).

Extract the CheXpert training dataset into `/content/chexpert`, or change
`DATA_ROOT` to an extracted dataset folder in Drive. The folder must contain one
`train.csv` and resolve its image paths. Local `/content` storage is faster but
disappears when the runtime is deleted. Only training-cohort data is accepted.

Authorize the Drive mount. `SAVE_ROOT` defaults to
`/content/drive/MyDrive/chex_colab`. Splits, per-epoch `best.pt` and `last.pt`,
calibration files and artifact ZIPs are stored there. A fresh run refuses to
overwrite an existing run's checkpoints. The Kaggle notebook remains available.

Start with `RUN_MODE="smoke"` and `SMOKE_PHASE="fresh"`. This trains one epoch
of a planned two-epoch run and creates `artifacts/<arch>_seed<seed>_resume.zip`.
Record the SHA-256 printed by the producer independently. To finish the smoke
test, set `SMOKE_PHASE="resume"`, `RESUME_BUNDLE` to that ZIP's Drive path and
`EXPECTED_RESUME_BUNDLE_SHA256` to the recorded digest. Keep the same source
data, split manifest, repository revision and batch size.

For full training, set `RUN_MODE="full"`, pin `REPO_REF` to the committed version
containing these changes and supply the independently recorded
`EXPECTED_SOURCE_CSV_SHA256`. Keep batch size 32, as specified by protocol v0.1.
Each execution trains one architecture and one seed. Repeat for both architectures
and seeds 42–46. CUDA mixed precision uses autocast plus GradScaler; loss math
remains float32. Checkpoint and calibration packaging retain checksum validation.

If Colab disconnects before a resume ZIP is produced, the last completed epoch's
checkpoints remain on Drive. After mounting Drive and checking out the same
revision, create a bundle from that run before resuming:

```python
import subprocess, sys
from pathlib import Path
run = Path(SAVE_ROOT) / "runs" / ARCH / f"seed_{SEED}"
subprocess.check_call([
    sys.executable, "scripts/package_resume_bundle.py",
    "--best-checkpoint", str(run / "best.pt"),
    "--last-checkpoint", str(run / "last.pt"),
    "--resolved-config", str(run / "resolved_config.json"),
    "--manifest", str(Path(SAVE_ROOT) / "splits/protocol_v0_1/manifest.json"),
    "--protocol-config", "configs/protocol_v0_1.yaml",
    "--output", str(Path(SAVE_ROOT) / "artifacts" / f"{ARCH}_seed{SEED}_resume.zip"),
])
```

Record its printed digest, set the resume bundle configuration and rerun the
training cell. Interrupted writes or inconsistent checkpoint pairs fail
verification rather than silently continuing. Final locked-test evaluation is
handled separately by the protocol scripts.
