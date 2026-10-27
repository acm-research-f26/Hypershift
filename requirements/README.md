# Dependency sets

The base runtime remains in [requirements.txt](../requirements.txt). Additional
sets retain their original bytes and package pins:

| File | Purpose |
| --- | --- |
| requirements-data.txt | Public data acquisition |
| requirements-cuda.txt | Explicit CUDA PyTorch wheel |
| requirements-tested.txt | Recorded CUDA environment |
| requirements-tested-cpu.txt | Recorded historical CPU environment |
| requirements-research-20261020.txt | Additional pinned research libraries |

Pass the organized path to the installer, for example
`-r requirements/requirements-data.txt`. Use the CUDA index described in
[the hardware guide](../docs/TRAINING_HARDWARE.md) for the CUDA wheel. Historical
command comments inside pinned files are preserved as part of their provenance.
