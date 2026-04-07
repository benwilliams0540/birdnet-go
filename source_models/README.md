# Source Models

This folder is for local source-of-truth BirdNET model assets used during Uno Q
backend work.

## What belongs here

- trusted upstream model files kept locally for conversion or parity checks
- helper scripts such as `split_birdnet_v24.py`
- small notes that explain how the folder is meant to be used

## What does not belong here

- generated NCNN artifacts
- temporary split-model byproducts
- benchmark output
- anything that can be regenerated on demand

## Git Policy

Large model binaries in this folder are intentionally ignored by git. That
keeps the working repo usable while leaving room for a future Git LFS or
separate model-storage workflow.

Only lightweight helper files should be committed here.
