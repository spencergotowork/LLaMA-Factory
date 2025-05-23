# from typing import TYPE_CHECKING, Optional

# import torch
# import torch.nn.functional as F
# from trl import GRPOTrainer
# from transformers import Trainer, TrainerState, TrainerControl
# from transformers.trainer_utils import get_current_device

# from ...extras.logging import get_logger
# from ...extras.misc import AverageMeter
# from ...extras.packages import is_transformers_version_greater_than
# from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments
# from ..callbacks import SaveProcessorCallback
# from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
# from ..utils import create_custom_scheduler

# if TYPE_CHECKING:
#     from datasets import Dataset
#     from transformers import (
#         DataCollatorWithPadding,
#         PreTrainedTokenizer,
#         ProcessorMixin,
#         Seq2SeqTrainingArguments,
#         TrainerCallback,
#     )
#     from trl import AutoModelForCausalLMWithValueHead

# logger = get_logger(__name__)

# class CustomGRPOTrainer(PPOTrainer, Trainer):
#     r"""Inherit PPOTrainer and implement GRPO algorithm."""

#     def __init__(
#         self,
#         model_args: "ModelArguments",
#         training_args: "Seq2SeqTrainingArguments",
#         finetuning_args: "FinetuningArguments",
#         generating_args: "GeneratingArguments",
#         callbacks: Optional[list["TrainerCallback"]],
#         model: "AutoModelForCausalLMWithValueHead",
#         reward_model: Optional["AutoModelForCausalLMWithValueHead"],
#         ref_model: Optional["AutoModelForCausalLMWithValueHead"],
#         tokenizer: "PreTrainedTokenizer",
#         processor: Optional["ProcessorMixin"],
#         data_collator: "DataCollatorWithPadding",
#         train_dataset: Optional["Dataset"] = None,
#         eval_dataset: Optional["Dataset"] = None,
#     ) -> None:
#         if eval_dataset is not None:
#             raise NotImplementedError("GRPOTrainer does not support eval dataset yet.")

#         backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
#         ppo_config = PPOConfig(
#             model_name=model_args.model_name_or_path,
#             learning_rate=training_args.learning_rate,
#             mini_batch_size=training_args.per_device_train_batch_size,
#             batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
#             gradient_accumulation_steps=training_args.gradient_accumulation_steps,
#             ppo_epochs=finetuning_args.ppo_epochs,
#             max_grad_norm=training_args.max_grad_norm,
#             seed=training_args.seed,
#             optimize_device_cache=True,
#             target=finetuning_args.ppo_target,
#             use_score_scaling=finetuning_args.ppo_score_norm,
#             use_score_norm=finetuning_args.ppo_score_norm,
#             whiten_rewards=finetuning_args.ppo_whiten_rewards,
#             accelerator_kwargs={"step_scheduler_with_optimizer": False},
#             log_with=training_args.report_to[0] if training_args.report_to else None,
#             project_kwargs={"logging_dir": training_args.logging_dir},
#         )

#         # Add deepspeed config
#         if training_args.deepspeed_plugin is not None:
#             ppo_config.accelerator_kwargs["kwargs_handlers"] = [
#                 DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
#             ]
#             ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
#             if ppo_config.log_with is not None:
#                 logger.warning_rank0("GRPOTrainer cannot use external logger when DeepSpeed is enabled.")
#                 ppo_config.log_with = None

#         # Create optimizer and scheduler
#         if training_args.max_steps > 0:
#             num_training_steps = training_args.max_steps
#         else:
#             total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
#             num_training_steps = training_args.num_train_epochs * math.ceil(
#                 len(train_dataset) / total_train_batch_size
#             )

#         optimizer = self.create_optimizer(model, training_args, finetuning_args)
#         scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

#         GRPOTrainer.__init__(
#             self,
#             config=ppo_config,
#             model=model,
#             ref_model=ref_model,
#             tokenizer=tokenizer,
#             dataset=train_dataset,
#             optimizer=optimizer,
#             data_collator=data_collator,
#             lr_scheduler=scheduler,
#         )

#         self.args = training_args
#         self.model_args = model_args
#         self.finetuning_args = finetuning_args
#         self.reward_model = reward_model
#         self.current_device = get_current_device()

#         self.generation_config = GenerationConfig(
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
#             **generating_args.to_dict(),
#         )

#         self.state = TrainerState()
#         self.control = TrainerControl()
#         self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
#         self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
#         callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
#         self.callback_handler = CallbackHandler(
#             callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
#         )

#         self.amp_context = torch.autocast(self.current_device.type)
#         warnings.simplefilter("ignore")

#         if finetuning_args.reward_model_type == "full":
#             if self.is_deepspeed_enabled:
#                 if not (
#                     getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
#                     or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
#                 ):
#                     self.reward_model = self._prepare_deepspeed(self.reward_model)
#             else:
#                 self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

#         self.add_callback(FixValueHeadModelCallback)

#         if processor is not None:
#             self.add_callback(SaveProcessorCallback(processor))

#     def grpo_loss(self, values: torch.Tensor, old_values: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
#         r"""Compute GRPO loss."""
#         # Compute value loss
#         value_loss = F.mse_loss(values, old_values)
        
#         # Compute policy loss with GRPO
#         policy_loss = -torch.mean(advantages * torch.exp(values - old_values))
        
#         # Combine losses
#         total_loss = policy_loss + 0.5 * value_loss
        
#         return total_loss

#     def step(self, queries, responses, rewards):
#         r"""Run a PPO step with GRPO loss."""
#         # Get old values
#         old_values = self.get_values(queries, responses)
        
#         # Get new values and compute advantages
#         values = self.get_values(queries, responses)
#         advantages = rewards - old_values
        
#         # Compute GRPO loss
#         loss = self.grpo_loss(values, old_values, advantages)
        
#         # Backward pass
#         self.accelerator.backward(loss)
        
#         # Update model
#         self.optimizer.step()
#         self.optimizer.zero_grad()
        
#         return {
#             "grpo/loss/total": loss.item(),
#             "grpo/loss/policy": loss.item(),
#             "grpo/loss/value": loss.item(),
#             "grpo/mean_reward": rewards.mean().item(),
#             "grpo/mean_value": values.mean().item(),
#             "grpo/mean_advantage": advantages.mean().item(),
#         }

#     def get_values(self, queries, responses):
#         r"""Get value predictions from the model."""
#         batch = self.prepare_model_inputs(queries, responses)
#         with torch.no_grad():
#             values = self.model(**batch, return_dict=True, use_cache=False)[-1]
#         return values.float().detach()

#     def grpo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
#         r"""Implement training loop for the GRPO stage."""
#         self.callback_handler.on_train_begin(self.args, self.state, self.control)

#         max_steps = self.args.max_steps if self.args.max_steps > 0 else self.args.num_train_epochs * len(self.dataloader)
#         logger.info_rank0(f"  Total training steps = {max_steps:,}")
#         logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

#         dataiter = iter(self.dataloader)
#         loss_meter = AverageMeter()
#         reward_meter = AverageMeter()

#         for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
#             try:
#                 batch = next(dataiter)
#             except StopIteration:
#                 dataiter = iter(self.dataloader)
#                 batch = next(dataiter)

#             # Get inputs
#             self.model.eval()
#             self.tokenizer.padding_side = "right"
#             queries, responses, rewards = [], [], []
#             for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
#                 mini_batch = {
#                     "input_ids": batch["input_ids"][idx : idx + self.config.mini_batch_size],
#                     "attention_mask": batch["attention_mask"][idx : idx + self.config.mini_batch_size],
#                 }
#                 mini_batch_queries, mini_batch_responses = self.get_inputs(mini_batch)
#                 mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses)
#                 queries.extend(mini_batch_queries)
#                 responses.extend(mini_batch_responses)
#                 rewards.extend(mini_batch_rewards)

#             # Run GRPO step
#             self.model.train()
#             stats = self.step(queries, responses, rewards)
#             self.tokenizer.padding_side = "left"
#             loss_meter.update(float(stats["grpo/loss/total"]), n=len(rewards))
#             reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

#             if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
#                 self.log(
#                     {
#                         "loss": loss_meter.avg,
#                         "reward": reward_meter.avg,
#                         "learning_rate": self.lr_scheduler.get_last_lr()[0],
#                         "epoch": (step + 1) / len(self.dataloader),
#                     }
#                 )

#             if (step + 1) % self.args.save_steps == 0:
#                 self.save_model()

#         self.callback_handler.on_train_end(self.args, self.state, self.control) 

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from ..trainer_utils import create_modelcard_and_push
from ...extras.callbacks import SaveProcessorCallback
from ...extras.logging import get_logger
from ...extras.misc import count_parameters, get_logits_processor
from ...hparams import ModelArguments, DataArguments, FinetuningArguments

logger = get_logger(__name__)


class CustomGRPOTrainer(GRPOTrainer):
    """自定义GRPO训练器，集成LLaMA-Factory功能"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        model_args: Optional[ModelArguments] = None,
        data_args: Optional[DataArguments] = None,
        finetuning_args: Optional[FinetuningArguments] = None,
        **kwargs,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        
        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        
        # 添加保存处理器回调
        if self.finetuning_args.save_processor:
            self.add_callback(SaveProcessorCallback)

    def grpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        group_labels: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """计算GRPO损失函数"""
        
        # 计算log ratio
        policy_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        policy_ratio_rejected = policy_rejected_logps - reference_rejected_logps
        
        # Group-wise相对优化
        unique_groups = torch.unique(group_labels)
        total_loss = 0.0
        chosen_rewards = []
        rejected_rewards = []
        
        for group_id in unique_groups:
            group_mask = (group_labels == group_id)
            if group_mask.sum() == 0:
                continue
                
            group_chosen_ratio = policy_ratio_chosen[group_mask]
            group_rejected_ratio = policy_ratio_rejected[group_mask]
            
            # 组内相对比较
            group_logits = group_chosen_ratio - group_rejected_ratio
            group_loss = -F.logsigmoid(self.args.beta * group_logits).mean()
            
            total_loss += group_loss
            chosen_rewards.extend(self.args.beta * group_chosen_ratio.detach())
            rejected_rewards.extend(self.args.beta * group_rejected_ratio.detach())
        
        chosen_rewards = torch.stack(chosen_rewards)
        rejected_rewards = torch.stack(rejected_rewards)
        
        return total_loss / len(unique_groups), chosen_rewards, rejected_rewards

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型"""
        super().save_model(output_dir, _internal_call)
        
        if not _internal_call and self.finetuning_args.create_model_card:
            create_modelcard_and_push(self, output_dir)


def run_grpo(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: GRPOConfig,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List] = None,
):
    """运行GRPO训练"""
    from ...data import get_dataset, split_dataset
    from ...model import load_model_and_tokenizer
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, is_trainable=True)
    
    # 加载数据集
    dataset = get_dataset(model_args, data_args, training_args, stage="rm", **split_dataset(dataset, data_args, training_args))
    
    # 创建训练器
    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
    )
    
    # 添加回调
    if callbacks:
        for callback in callbacks:
            trainer.add_callback(callback)
    
    # 开始训练
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 保存模型
    trainer.save_model()
    trainer.save_state()
