#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Train a VLA policy in VAE latent action space (RTR lerobot_train_latent_rdp_vae recipe)."""

import logging
import time
from pprint import pformat

import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.latent_train import LatentTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats
from lerobot.rl.acp_hook import build_acp_raw_batch_hook
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging, inside_slurm
from lerobot.vae.factory import load_action_vae
from lerobot.vae.latent_policy import (
    configure_policy_for_latent_training,
    encode_actions_to_latent_batch,
    latent_reconstruction_l1,
    resolve_temporal_downsample_ratio,
)


@parser.wrap()
def train_latent(cfg: LatentTrainPipelineConfig, accelerator: Accelerator | None = None) -> None:
    cfg.validate()
    acp_raw_batch_hook = build_acp_raw_batch_hook(cfg.acp, cfg.seed)

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project and is_main_process else None
    if wandb_logger is None and is_main_process:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    cfg.vae.device = str(device)
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        if cfg.acp.enable:
            indicator_stats = compute_acp_indicator_stats(dataset, cfg.acp.indicator_field)
            if indicator_stats is not None and indicator_stats.total_count >= 0:
                logging.info(
                    "ACP indicator stats (%s): field='%s' ratio=%.6f positive=%d total=%d",
                    indicator_stats.source,
                    indicator_stats.indicator_field,
                    indicator_stats.positive_ratio,
                    indicator_stats.positive_count,
                    indicator_stats.total_count,
                )

    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)

    temporal_ratio = resolve_temporal_downsample_ratio(cfg.vae, cfg.temporal_downsample_ratio)
    if is_main_process:
        logging.info("Loading frozen action VAE from %s", cfg.vae_checkpoint_path)
    vae = load_action_vae(cfg.vae, cfg.vae_checkpoint_path)
    vae._load_latent_dataset_statistics(cfg.latent_dataset_statistics)

    eval_env = None

    if is_main_process:
        logging.info("Creating latent policy (chunk_size=%d before downsampling)", cfg.policy.chunk_size)
    configure_policy_for_latent_training(
        cfg.policy,
        cfg.vae,
        temporal_ratio,
        match_rtr_n_action_steps=cfg.match_rtr_n_action_steps,
    )
    if is_main_process:
        logging.info(
            "Latent policy horizon: chunk_size=%d n_action_steps=%d max_action_dim=%d temporal_ratio=%d",
            cfg.policy.chunk_size,
            cfg.policy.n_action_steps,
            cfg.policy.max_action_dim,
            temporal_ratio,
        )

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        import dataclasses

        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    accelerator.wait_for_everyone()

    processor_pretrained_path = cfg.policy.pretrained_path
    if (
        getattr(cfg.policy, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms; "
            "building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=processor_pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=getattr(cfg, "rabc_head_mode", "sparse"),
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    if is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="LatentTraining",
            unit="step",
            disable=inside_slurm(),
        )
        logging.info("Start latent-space offline training")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)

        if acp_raw_batch_hook is not None:
            batch = acp_raw_batch_hook(batch, step)

        gt_action = batch["action"].clone()
        latent_action = encode_actions_to_latent_batch(
            vae,
            batch,
            normalization_type=cfg.latent_normalization_type,
        )

        batch = preprocessor(batch)
        batch["action"] = latent_action.to(device=batch["action"].device, dtype=batch["action"].dtype)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps)
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_saving_step:
            if is_main_process:
                logging.info("Checkpoint latent policy after step %d", step)
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if is_eval_step:
            if is_main_process:
                with torch.no_grad(), accelerator.autocast():
                    unwrapped = accelerator.unwrap_model(policy)
                    predicted_latent = unwrapped.predict_action_chunk(batch)
                l1loss = latent_reconstruction_l1(
                    vae,
                    predicted_latent,
                    gt_action.to(device=predicted_latent.device),
                    normalization_type=cfg.latent_normalization_type,
                )
                eval_metrics = {"l1loss": AverageMeter("l1loss", ":.4f")}
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.l1loss = l1loss
                logging.info("Latent recon eval @ step %d: L1=%.6f", step, l1loss)
                if wandb_logger:
                    wandb_logger.log_dict(eval_tracker.to_dict(), step, mode="eval")
            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()
        logging.info("End of latent training")
        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    if eval_env:
        close_envs(eval_env)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main() -> None:
    register_third_party_plugins()
    train_latent()


if __name__ == "__main__":
    main()
