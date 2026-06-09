"""Vanilla GELU MLP baseline for FineWeb (GPT-2 small)."""

out_dir = "out-fineweb-vanilla-100k"
# eval/ckpt cadence reachable within a 4h Slurm chunk (~9.5 s/iter); ckpt_interval
# (250, in train.py) saves the resume checkpoint independently. Matched to the
# prism config so the two 100k runs are directly comparable.
eval_interval = 500
eval_iters = 100
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "vanilla-4x-100k"

dataset = "fineweb"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "vanilla"
mlp_expansion = 4

learning_rate = 4e-3            # matched recipe (same schedule as the prism run)
max_iters = 100000
lr_decay_iters = 100000
min_lr = 4e-4                   # lr/10 cosine floor
warmup_iters = 2000

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dtype = "bfloat16"
compile = False
