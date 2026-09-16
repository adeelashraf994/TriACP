# Upload to GitHub

## Web interface

1. Create a new GitHub repository, for example `TriACP`.
2. Extract the provided ZIP.
3. Upload all files and folders from `TriACP_reference_implementation/` to the repository root.
4. Before making the repository public, replace placeholder manuscript wording/URLs in `README.md` and confirm that no private datasets, local paths, credentials, or institutional files are included.

## Git command line

```bash
git init
git add .
git commit -m "Initial TriACP reference implementation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/TriACP.git
git push -u origin main
```

Recommended repository description:

> Reference implementation of TriACP, a three-stage computational framework for ACP identification, cancer-type activity prediction, and IC50-based potency classification.
