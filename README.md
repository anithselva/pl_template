# PyTorch Lightning Project Template

This repo contains a flexible framework for config-based model development. This structure is designed to allow for drop in models, losses, and metrics, through modularized components. Adoption of this flow will allow for easy, flexible, and reproducible development.
## Folder Structure

```.
├── configs             - store experiment configuration json files
├── data                - location for storing raw data (opt)
├── dataset             - dataset classes
├── experiments         - session logs, artifacts checkpoints
├── src                 - store Model files, loss functions, and evaluation metrics here
│   ├── losses
│   ├── metrics
│   ├── models
├── utils               - commonly used general functions
├── README.md
└── train.py            - main loop
├── requirements        - python deps
```

## Experiment Tracking

ML Flow is used for experiment tracking. All artifacts and logs are stored in the experiment folder.

The ML Flow UI can be accessed by running:

`mlflow ui --backend-store-uri ./experiments`

and directing your browser to:

`127.0.0.1:5000` or `localhost:5000`

## Pre-Commit

Pre-commit hooks is used to maintain good SW practices and consistency.

Use the following git flow:
```
git add <files>
pre-commit
git add <files that were modified after pre-commit>
git commit -m "commit message
git push
```

`pre-commit` will run various checks and make appropriate modifications to the staged files. Files will need to be re-staged after any changes made from pre-commit

# Future Work

- Dockerization
