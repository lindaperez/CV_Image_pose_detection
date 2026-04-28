# Commit Code Helper

Use `commit_code.py` to stage and commit repository changes while avoiding local
or generated artifacts.

## Files

- `scripts/commit_code.py`: main implementation and source of truth.

## What It Excludes

The helper does not commit these files unless you stage and commit them manually:

- `.venv/`, `venv/`, `env/`
- `.DS_Store`
- `.ipynb_checkpoints/`
- `__pycache__/`, `*.pyc`, `*.pyo`
- large raw videos such as `*.mp4`, `*.mov`, `*.avi`, `*.mkv`, `*.webm`
- archives such as `*.tar`, `*.tar.gz`, `*.tgz`, `*.zip`
- generated arrays such as `*.npy`, `*.npz`
- model/checkpoint files such as `*.pt`, `*.pth`, `*.onnx`
- local training output folders

## Preview a Commit

From the repository root:

```bash
cd /Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/CV_Image_pose_detection
scripts/commit_code.py --dry-run
```

This shows the files that would be staged and committed. It does not create a
commit, and it leaves the Git index clean afterward.

## Create a Commit

```bash
scripts/commit_code.py -m "Clean repository structure"
```

## Notes

- Run `--dry-run` first when the tree has many changes.
- The helper refuses to run if files are already staged. Unstage them first with:

```bash
git restore --staged :/
```

- If a generated file is intentionally tiny and should be committed, commit it
  manually with `git add -f <path>` and `git commit`.
