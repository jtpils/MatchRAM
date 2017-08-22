class Config(object):
  win_size = 8
  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 100  # changed for mnist pair
  loc_std = 0.22
  original_size = 40 # 40 for mnistpair
  num_channels = 1
  depth = 1
  sensor_size = win_size**2 * depth
  minRadius = 8
  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 512
  loc_dim = 2
  cell_size = 512
  cell_out_size = cell_size
  num_glimpses = 6
  num_classes = 2 # 2 for mnist pair
  max_grad_norm = 5.
  training_steps_per_epoch = 1000

  step = 100000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10
