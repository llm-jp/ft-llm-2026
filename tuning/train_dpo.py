import glob
import os
import shutil
from functools import partial

import torch.multiprocessing as mp
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo_aligner.algorithms.dpo import DPOTrainer, dpo_custom_collate
from nemo_aligner.models.nlp.gpt.megatron_gpt_dpo_model import MegatronGPTDPOModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_using_ptl,
    resolve_and_create_trainer,
)
from nemo_aligner.utils.utils import (
    load_checkpoint_model_config,
    load_from_nemo,
    remove_overwritten_fields,
    retrieve_model_state_dict_in_cpu,
)
from omegaconf.omegaconf import OmegaConf

from dpo_dataset import DPOModelDataset, create_dpo_dataloader, load_dpo_datasets

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

mp.set_start_method("spawn", force=True)


def custom_load_and_override_model_config(
    restore_path, cfg_to_overwrite, remove_meta_info=True
):
    """
    This function loads the original model configuration from the checkpoint and overrides it with the new configuration.
    This is based on the original `nemo_aligner.utils.utils.load_and_override_model_config` function but adds additional fields.
    """
    checkpoint_cfg = load_checkpoint_model_config(restore_path)

    if remove_meta_info:
        checkpoint_cfg.pop("target", None)
        checkpoint_cfg.pop("nemo_version", None)

    if "overwrite_base_config" in cfg_to_overwrite.model:
        remove_overwritten_fields(
            checkpoint_cfg, cfg_to_overwrite.model.overwrite_base_config
        )

    merged_cfg = OmegaConf.merge(checkpoint_cfg, cfg_to_overwrite.model)

    merged_cfg.micro_batch_size = cfg_to_overwrite.mbs
    merged_cfg.global_batch_size = cfg_to_overwrite.gbs
    merged_cfg.data = cfg_to_overwrite.data
    merged_cfg.sequence_parallel = cfg_to_overwrite.model.sequence_parallel
    merged_cfg.activations_checkpoint_granularity = (
        cfg_to_overwrite.model.activations_checkpoint_granularity
    )
    merged_cfg.activations_checkpoint_num_layers = (
        cfg_to_overwrite.model.activations_checkpoint_num_layers
    )
    merged_cfg.activations_checkpoint_method = (
        cfg_to_overwrite.model.activations_checkpoint_method
    )
    merged_cfg.activations_checkpoint_layers_per_pipeline = (
        cfg_to_overwrite.model.activations_checkpoint_layers_per_pipeline
    )
    merged_cfg.optim = cfg_to_overwrite.model.optim
    merged_cfg.precision = cfg_to_overwrite.trainer.precision
    merged_cfg.restore_from_path = cfg_to_overwrite.model.restore_from_path
    merged_cfg.resume_from_checkpoint = cfg_to_overwrite.model.resume_from_checkpoint
    merged_cfg.save_nemo_on_validation_end = (
        cfg_to_overwrite.model.save_nemo_on_validation_end
    )
    merged_cfg.gradient_as_bucket_view = cfg_to_overwrite.model.gradient_as_bucket_view
    merged_cfg.hidden_dropout = cfg_to_overwrite.model.hidden_dropout
    merged_cfg.attention_dropout = cfg_to_overwrite.model.attention_dropout
    merged_cfg.ffn_dropout = cfg_to_overwrite.model.ffn_dropout
    merged_cfg.use_flash_attention = cfg_to_overwrite.model.use_flash_attention
    merged_cfg.tensor_model_parallel_size = (
        cfg_to_overwrite.model.tensor_model_parallel_size
    )
    merged_cfg.pipeline_model_parallel_size = (
        cfg_to_overwrite.model.pipeline_model_parallel_size
    )

    # dpo
    merged_cfg.dpo = cfg_to_overwrite.dpo

    # transformer engine
    merged_cfg.transformer_engine = cfg_to_overwrite.model.transformer_engine
    merged_cfg.fp8 = cfg_to_overwrite.model.fp8
    merged_cfg.fp8_e4m3 = cfg_to_overwrite.model.fp8_e4m3
    merged_cfg.fp8_hybrid = cfg_to_overwrite.model.fp8_hybrid
    merged_cfg.fp8_margin = cfg_to_overwrite.model.fp8_margin
    merged_cfg.fp8_interval = cfg_to_overwrite.model.fp8_interval
    merged_cfg.fp8_amax_history_len = cfg_to_overwrite.model.fp8_amax_history_len
    merged_cfg.fp8_amax_compute_algo = cfg_to_overwrite.model.fp8_amax_compute_algo
    merged_cfg.reduce_amax = cfg_to_overwrite.model.reduce_amax
    merged_cfg.use_emha = cfg_to_overwrite.model.use_emha

    # Remove the "overwrite_base_config" key to avoid cluttering the model config.
    merged_cfg.pop("overwrite_base_config", None)

    return merged_cfg


@hydra_runner(config_path="configs", config_name="dpo")
def main(cfg) -> None:
    if cfg.use_mpi:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if cfg.use_slurm:
            os.environ["SLURM_PROCID"] = str(global_rank)
            os.environ["SLURM_LOCALID"] = str(local_rank)
            os.environ["SLURM_NTASKS"] = str(world_size)
            os.environ["SLURM_NTASKS_PER_NODE"] = "8"
            os.environ["SLURM_NNODES"] = f"{cfg.trainer.num_nodes}"
            os.environ["SLURM_NODEID"] = os.getenv("OMPI_COMM_WORLD_NODE_RANK", "0")
        logging.info(
            f"global_rank: {global_rank}, local_rank: {local_rank}, world_size: {world_size}"
        )

    cfg.model = custom_load_and_override_model_config(
        restore_path=cfg.model.restore_from_path,
        cfg_to_overwrite=cfg,
    )
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg.model)}")

    trainer = resolve_and_create_trainer(cfg, "dpo")
    log_dir = exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model, updated_cfg = load_from_nemo(
        MegatronGPTDPOModel,
        cfg.model,
        trainer,
        strict=True,
        load_base_model_only=False,
        restore_path=cfg.model.restore_from_path,
        return_updated_cfg=True,
    )

    # save the updated config to the log directory
    if is_global_rank_zero():
        updated_config_path: str = f"{log_dir}/checkpoints/model_config.yaml"
        os.makedirs(os.path.dirname(updated_config_path), exist_ok=True)
        OmegaConf.save(updated_cfg, updated_config_path)

    ref_policy_state_dict = retrieve_model_state_dict_in_cpu(
        ptl_model, megatron_amp_O2=True
    )
    ptl_model.ref_policy_state_dict = ref_policy_state_dict

    train_examples, dev_examples = load_dpo_datasets(cfg, ptl_model.tokenizer)

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    train_dataset = DPOModelDataset(
        tokenized_examples=train_examples,
        tokenizer=ptl_model.tokenizer,
        max_seq_length=cfg.model.encoder_seq_length,
        sequence_parallel=cfg.model.sequence_parallel,
    )

    val_dataset = DPOModelDataset(
        tokenized_examples=dev_examples,
        tokenizer=ptl_model.tokenizer,
        max_seq_length=cfg.model.encoder_seq_length,
        sequence_parallel=cfg.model.sequence_parallel,
    )

    train_dataloader = create_dpo_dataloader(
        dataset=train_dataset,
        consumed_samples=0,
        mbs=cfg.data.train_ds.micro_batch_size,
        gbs=cfg.data.train_ds.global_batch_size,
        seed=cfg.seed,
        use_random_sampler=True,
        collate_fn=partial(
            dpo_custom_collate,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        ),
    )

    val_dataloader = create_dpo_dataloader(
        dataset=val_dataset,
        consumed_samples=0,
        mbs=cfg.data.validation_ds.micro_batch_size,
        gbs=cfg.data.validation_ds.global_batch_size,
        seed=cfg.seed,
        use_random_sampler=False,
        collate_fn=partial(
            dpo_custom_collate,
            eos_id=ptl_model.tokenizer.eos_id,
            reset_position_ids=cfg.model.data.get("reset_position_ids", False),
            reset_attention_mask=cfg.model.data.get("reset_attention_mask", False),
            eod_mask_loss=cfg.model.data.get("eod_mask_loss", False),
        ),
    )

    init_using_ptl(trainer, ptl_model, train_dataloader, train_dataset)
    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    timer = Timer(cfg.exp_manager.get("max_time_per_run"))
    dpo_trainer = DPOTrainer(
        cfg=cfg.trainer.dpo,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )

    dpo_trainer.fit()

    # remove optimizer state files
    for optimizer_state_file in glob.glob(
        f"{log_dir}/checkpoints/step*/optimizer.state.*"
    ):
        try:
            shutil.rmtree(optimizer_state_file)
            logging.info(f"Deleted directory: {optimizer_state_file}")
        except OSError as e:
            logging.error(f"Error: {optimizer_state_file} : {e.strerror}")


if __name__ == "__main__":
    main()
