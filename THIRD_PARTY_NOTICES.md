# Third-party sources and asset terms

The repository's MIT license applies to its original code. It does not replace
licenses on data, annotations, derived assets, pretrained backbones or weights.
Dependencies are installed separately and retain their own terms: PyTorch
(BSD-style), NumPy and SciPy (BSD-3-Clause), and Matplotlib (PSF-based).

| Source | Use in this work | Provider's terms |
|---|---|---|
| [COCO](https://cocodataset.org/#termsofuse) | images and instance masks | annotation and image terms are distinct; original image owners retain rights |
| [VGGSound](https://github.com/hche11/VGGSound) | audio-source IDs and clips | dataset CC BY 4.0; original videos remain with their copyright owners |
| [Open Images](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | images and segmentation annotations | annotations CC BY 4.0; images listed as CC BY 2.0, with per-image verification required by the provider |
| [FSD50K](https://zenodo.org/records/4060432) | cross-domain sound clips | sound-level license and attribution metadata accompany the dataset |
| [CAVP](https://github.com/cyh-0/CAVP) | frozen AVS and VPO preparation | obtain code/data/weights under the upstream terms |
| [DenseAV](https://github.com/mhamilton723/DenseAV) | zero-shot reference scores | upstream implementation commit 2221b8c9c1ae9317fd765ec9f151f9224aaff9f5; code MIT; source and weights not bundled |

The compact package contains scalar evidence, manifests, evaluation scores
and seven auditor/assignment checkpoints. Raw dataset media and upstream
backbone weights are obtained separately from their providers.

The MIT license covers original code and the author-trained auditor/assignment
checkpoints. Derived evidence and metadata retain applicable source-dataset
terms. The source identifiers and fingerprints in the manifests record their
provenance. Media restored with the optional downloader remain subject to the
providers' licenses and per-item attribution requirements.

## Additional evaluation sources

[AVSegFormer](https://github.com/vvvb-github/AVSegFormer) supplies the upstream architecture for the transfer experiment. Only derived scalar evidence is included here; upstream code and weights are not redistributed. Neither the AVSegFormer nor the CAVP repository had a root license file in the repository snapshots checked on 24 September 2026. Public source availability alone does not grant redistribution rights; obtain the applicable terms from the upstream authors before redistributing their implementations or weights.

The code's MIT license does not grant additional rights over any source dataset. Preserve source attribution when using the derived evidence and follow the provider's terms when obtaining the original media.
