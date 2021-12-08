# torchnerf

### train

```
python train.py \
    --gin_files ./configs/blender.py ./configs/c64f128.py \
    --data_dir ./data/nerf_synthetic/lego \
    --train_dir ./logs/lego
```

### eval

```
python eval.py \
    --gin_files ./configs/blender.py ./configs/c64f128.py \
    --data_dir ./data/nerf_synthetic/lego \
    --train_dir ./logs/lego
```

## Acknowledgement

The code base is origined from an awesome [torchnerf](https://github.com/liruilong940607/torchnerf) implementation.
