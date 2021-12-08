Dataset.image_batching = True
Dataset.white_bkgd = False
Dataset.factor = 8
Dataset.spherify = True

NerfModel.num_coarse_samples = 64
NerfModel.num_fine_samples = 128
NerfModel.near = 0.2
NerfModel.far = 100.
NerfModel.noise_std = 1.
NerfModel.lindisp = True