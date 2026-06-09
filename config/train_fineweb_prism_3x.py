"""GPT-2 small (124M) with prism-3× MLPs on FineWeb-Edu sample-10BT.

Multi-epoch confirmation of the param-efficient match. At ~1 epoch (20k iters)
the (gs=64, r=32) shape matched the vanilla-4× baseline (val 3.0938 vs 3.0916,
within the ~0.5% noise floor) at -3% params. This run extends it to 100k iters
(~5 epochs) with the matched recipe (LR=4e-3, r_lr_mult=8, r_init_scale=0.5) to
test whether the match holds with repeated data. Compare against
config/train_fineweb_vanilla.py run at the SAME 100k budget.

Wallclock: ~10 days on 2x A100 at ~9.5 s/iter (or ~half on 4 GPUs).
"""

out_dir = "out-fineweb-prism-3x-gs64r32"
# ~9.5 s/iter on 2xA100 => a 4h Slurm chunk only reaches ~1500 iters, so the old
# eval_interval=2000 was never hit: no eval, no checkpoint, every requeue
# restarted from scratch at iter 0. Eval (hence wandb scalars) every 500 iters
# (~1.3h) is reachable within a chunk; ckpt_interval (250, in train.py) saves the
# resume checkpoint independently every ~40min. eval_iters trimmed to keep eval cheap.
eval_interval = 500
eval_iters = 100
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "prism-3x-gs64r32-100k"

dataset = "fineweb"
# 5 * 8 = 40 micro-batches. Per-rank micro-batch = 12, block_size = 1024.
# Effective batch tokens = 12 * 1024 * 40 = 491,520 ≈ nanoGPT's 0.5M.
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "prism"
mlp_expansion = 3               # hidden width = 3 * 768 = 2304
group_size = 64                 # 2304 / 64 = 36 groups
reconn_sz = 32                  # gate rank; the param-efficient match shape (~120M, -3%)
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 0.5              # matched recipe
r_lr_mult = 8                   # matched recipe: R learns at 8x the base LR

learning_rate = 4e-3            # matched recipe
max_iters = 100000
lr_decay_iters = 100000
min_lr = 4e-4                   # lr/10 cosine floor
warmup_iters = 2000

weight_decay = 0.1
r_wd_factor = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dtype = "bfloat16"
compile = False
