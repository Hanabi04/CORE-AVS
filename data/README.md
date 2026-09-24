# Evidence and manifests

| Split | Images | Output rows | File |
|---|---:|---:|---|
| Fitting | 1,316 | 3,948 | `train_features/train.npz` |
| Controlled | 200 | 600 | `eval_features/controlled.npz` |
| CORE A | 200 | 600 | `eval_features/core_a.npz` |
| CORE B | 200 | 600 | `eval_features/core_b.npz` |
| CORE C | 673 | 2,019 | `eval_features/core_c.npz` |

Each image occupies three consecutive rows: matched (0), silence (1), mismatch
(2). CORE concatenates A, B, C in that order. `features` has shape
`[3 * images, 21, 13]`, float32. `truth_risk` stores reference-mask risk; `arms`
stores audio-state labels. `uids` and `folds` are image-level. Normalization is
computed on the fitting partition, not the evaluation arrays.

Manifests contain the UID, source-group IDs, class indices, prepared-asset
fingerprints and a hash for each three-row feature block. Top-level class
indices are zero-based. Training source metadata also preserves its original
one-based labels as `original_class_indices`. Available source identities and
clip IDs remain in `source_metadata`.

## Obtain upstream media

- COCO: https://cocodataset.org/#download
- VGGSound: https://github.com/hche11/VGGSound (video ID/start-time CSV)
- Open Images: https://storage.googleapis.com/openimages/web/download_v7.html
- FSD50K: https://zenodo.org/records/4060432
- CAVP / VPO: https://github.com/cyh-0/CAVP

Acquire media under the original providers' terms. The distributed scalar
features allow evaluation without downloading those files. Global start/end
fields are null where the original entry did not record source offsets; a
prepared three-second mixture alone cannot recover the complete mixing recipe.

`datasets/download_manifest_media.py` restores already-prepared authorized
assets from a URL map. Its map fields are `uid`, `asset`, `url`, `sha256` and
`relative_path`. Each hash must occur in the corresponding manifest entry.

```sh
python datasets/download_manifest_media.py --manifest data/manifests/controlled_test_200.json --sources authorized_sources.json --output raw_media
```

This downloader copies and validates prepared assets. Reconstructing all source
clips, mixes and target masks from internet IDs additionally requires the
original upstream preparation pipeline and missing source offsets. It is not
part of the cached-evidence reproduction command.

See `THIRD_PARTY_NOTICES.md` for the separation between code and asset terms.

## AVSegFormer transfer evidence

`features/avsegformer_train.npz` and `features/avsegformer_test.npz` contain
the original 1,316-image fit and 200-image confirmation evidence under the
AVSegFormer-R50 static-overlap adaptation. Each row is one audio condition;
`uids` here is row-aligned (three copies per image), unlike the image-aligned
UID arrays in `train_features/` and `eval_features/`. Both preserve the original
image order. `soft_dice` stores the original scalar confidence-derived risk.
No AVSegFormer code, official weight or adapted backbone weight is redistributed.
