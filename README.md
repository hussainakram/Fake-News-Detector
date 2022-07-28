# Fake News Detector
Code implementation for fake news detection using GAT, GCN and GraphSAGE

## Requirements
- Numpy
- Pytorch (`torch_geometric`)
- scikit-learn (`sklearn.metrics`)

## Dataset
Download dataset using this [link](https://drive.google.com/file/d/1KOmSrlGcC50PjkvRVbyb_WoWHVql06J-/view?usp=sharing)

## Run
```bash
python run.py
```

## Configuration

For changing models you can pass --model='sage' or set value manually. The allowed values for models are:
  1. sage
  2. gcn
  3. gat

Other available args are as follows:

- --batch_size
- --lr
- --hidden_size
- --epochs
- --feature
