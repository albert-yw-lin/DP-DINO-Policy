import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--dataset_path', default=None, help='Override dataset path for evaluation')
@click.option('--env_name', default=None, help='Override environment name (e.g., MugLift_2, Lift, PickPlaceCan)')
@click.option('--n_test', default=None, type=int, help='Override number of test episodes (random seeds)')
@click.option('--n_train', default=None, type=int, help='Override number of train episodes (from dataset initial states)')
def main(checkpoint, output_dir, device, dataset_path, env_name, n_test, n_train):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill, weights_only=False)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner_cfg = cfg.task.env_runner
    
    # Override dataset path if provided
    if dataset_path is not None:
        env_runner_cfg.dataset_path = dataset_path
        print(f"Overriding dataset path to: {dataset_path}")
    
    # Override environment name if provided
    if env_name is not None:
        # This will be picked up by the runner when it loads env_meta from the dataset
        # We need to modify the env_meta after loading
        print(f"Will override environment name to: {env_name}")
    
    # Override number of test/train episodes if provided
    if n_test is not None:
        env_runner_cfg.n_test = n_test
        # env_runner_cfg.n_test_vis = min(n_test, env_runner_cfg.get('n_test_vis', 1))
        env_runner_cfg.n_test_vis = 0
        print(f"Overriding n_test to: {n_test}, n_test_vis to: {env_runner_cfg.n_test_vis}")
    else:
        # Default: reduce parallel environments to avoid resource exhaustion
        env_runner_cfg.n_test = 0
        env_runner_cfg.n_test_vis = 0
        print(f"Using default: n_test=1, n_test_vis=1")
    
    if n_train is not None:
        env_runner_cfg.n_train = n_train
        # env_runner_cfg.n_train_vis = min(n_train, env_runner_cfg.get('n_train_vis', 0))
        env_runner_cfg.n_train_vis = 0
        print(f"Overriding n_train to: {n_train}, n_train_vis to: {env_runner_cfg.n_train_vis}")
    else:
        # Default: no train episodes
        env_runner_cfg.n_train = 0
        env_runner_cfg.n_train_vis = 0
        print(f"Using default: n_train=0, n_train_vis=0")
    
    # Override max_steps for longer evaluation episodes
    env_runner_cfg.max_steps = 2000
    print(f"Overriding max_steps to: 2000")
    
    # Prepare kwargs for instantiation
    runner_kwargs = {'output_dir': output_dir}
    if env_name is not None:
        runner_kwargs['env_name_override'] = env_name
    
    env_runner = hydra.utils.instantiate(
        env_runner_cfg,
        **runner_kwargs)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()