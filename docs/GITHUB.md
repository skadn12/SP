# GitHub Publishing

This directory is designed to be published as a small standalone repository.
It does not include generated datasets, run outputs, model weights, or secrets.

## Create a Local Repo

```bash
cd /root/autodl-tmp/gxy-selfplay/current_method_github
git init
git add .
git status
git commit -m "Add current method self-play code"
```

## Push to GitHub

Create an empty GitHub repository, then run:

```bash
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

If you use HTTPS instead of SSH:

```bash
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Before Pushing

Check that no data, model, or secret files slipped in:

```bash
du -sh .
find . -type f -size +10M -print
grep -R "api_key\\|sk-" . --exclude-dir=.git
```

The expected package size is small, roughly a few megabytes.

