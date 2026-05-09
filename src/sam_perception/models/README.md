# Model Files

The original local project used the SAM checkpoint file below:

- `sam_vit_b_01ec64.pth`

It is not stored in normal Git history because the file is larger than GitHub's regular push limit.

Download it from the repository release asset:

```bash
bash scripts/download_models.sh
```

You can also place the checkpoint in this directory with the exact filename above, or set:

`SAM_CHECKPOINT_PATH=/absolute/path/to/sam_vit_b_01ec64.pth`
