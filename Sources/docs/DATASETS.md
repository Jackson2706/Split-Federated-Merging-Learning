# Datasets: preparation and verification

Entry point: `Sources/scripts/prepare_datasets.py`. Run it from `Sources/`; configs use paths relative to `Sources/`.

```bash
cd Sources
python scripts/prepare_datasets.py                          # cifar10 cifar100 ham10000 (default set)
python scripts/prepare_datasets.py --verify-only            # check only, never download
python scripts/prepare_datasets.py --datasets isic2018      # optional, ~11 GB download / ~24 GB peak disk
python scripts/prepare_datasets.py --root /other/data       # default root: Sources/data
```

Status for each dataset (source, archive MD5s, timestamps, last check) is in `data/DATASETS_STATUS.json`.
The exit code is 0 only when every requested dataset verifies.

| Dataset | Needed by | Source | Integrity check | Layout read by loaders |
|---|---|---|---|---|
| CIFAR-10 | E-HSFP (Tables I, VI, figures) | `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` | archive MD5 `c58f3010…`, then torchvision's per-batch MD5s | `data/cifar/cifar-10-batches-py/` |
| CIFAR-100 | E-HSFP (most tables), MiSFP (all tables) | `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` | archive MD5 `eb9058c3…`, per-file MD5s | `data/cifar/cifar-100-python/` |
| HAM10000 | E-HSFP (Tables I, VI) | Harvard Dataverse `doi:10.7910/DVN/DBW86T` (v4.0, CC BY-NC 4.0) | published MD5s of both image zips and of the original metadata CSV; 10,015 rows, every `image_id` resolves to one of 10,015 JPEGs | `data/HAM10000/HAM10000_metadata.csv`, `HAM10000_images_part_{1,2}/` |
| ISIC-2018 Task 1 | E-HSFP Table II (complete; only for reruns) | `https://isic-challenge-data.s3.amazonaws.com/2018/` | no published checksums: zip CRC and image/mask counts (2594 train, 100 validation) | `data/ISIC2018/ISIC2018_Task1-2_{Training,Validation}_Input`, `ISIC2018_Task1_{Training,Validation}_GroundTruth` |
| ImageNet-1K | E-HSFP Table I ImageNet column | licence-gated | not downloadable here | see the instructions the script prints (`--datasets imagenet`) |

## Guarantees

- Downloads go to `data/.partial/`. They resume per byte range and use parallel range requests, because a
  single connection here gets only ~35–90 KB/s. The Toronto host is limited to 3 connections: it reset TLS
  connections when 32 were opened. Data is moved into place only after its checksums and content checks pass.
- Tar members are extracted with the `data` filter. Zip members are checked against path traversal and
  CRC-tested.
- Existing data is never deleted. An invalid existing directory is renamed to `*.invalid-<epoch>`.
- A per-dataset `flock` in `data/.locks/` serialises concurrent callers. The loaders call
  `torchvision.datasets.CIFAR*(download=True)`. Once the batches verify, torchvision only checks the MD5s
  and does not download, so parallel jobs cannot race.
- The script does no splitting or normalisation. The train/test split, the `val_from_train` hold-out
  (seed-derived), client partitions (`camera_ready/partition.py`, seeded) and normalisation all stay in the
  loaders. Preparation therefore cannot leak test data or change the partitions of earlier runs.
- CIFAR must be the original Toronto pickles, not a re-encoded copy such as the Hugging Face parquet.
  Client partitions index the original sample order, so a different order would silently change every
  client's data relative to the 55 completed protocol-v3 runs.

## Known data caveats (not changed: they would break comparability with completed runs)

- HAM10000 uses an image-level stratified 80/20 split (`train_test_split(..., random_state=42)`) that
  ignores `lesion_id`. Images of the same lesion can fall on both sides, so test accuracy may be optimistic.
  The MiSFP manuscript calls this an unverified "patient-aware split". The E-HSFP Table I runs already
  use it.
- The HAM10000 and ISIC test sets double as the validation set in the H-SFP runners (see
  `reports/completion_audit.md` §3). For this reason journal tables report final-round metrics.
