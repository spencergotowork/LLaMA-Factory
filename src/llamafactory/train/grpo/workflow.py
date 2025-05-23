# from typing import TYPE_CHECKING, Optional

# from ...data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
# from ...extras.ploting import plot_loss
# from ...model import load_model, load_tokenizer
# from ..callbacks import fix_valuehead_checkpoint
# from ..trainer_utils import create_ref_model, create_reward_model
# from .trainer import CustomGRPOTrainer

# if TYPE_CHECKING:
#     from transformers import Seq2SeqTrainingArguments, TrainerCallback
#     from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

# def run_grpo(
#     model_args: "ModelArguments",
#     data_args: "DataArguments",
#     training_args: "Seq2SeqTrainingArguments",
#     finetuning_args: "FinetuningArguments",
#     generating_args: "GeneratingArguments",
#     callbacks: Optional[list["TrainerCallback"]] = None,
# ):
#     tokenizer_module = load_tokenizer(model_args)
#     tokenizer = tokenizer_module["tokenizer"]
#     template = get_template_and_fix_tokenizer(tokenizer, data_args)
#     dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
#     model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)

#     tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
#     data_collator = MultiModalDataCollatorForSeq2Seq(template=template, model=model, **tokenizer_module)

#     # Create reference model and reward model
#     ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
#     reward_model = create_reward_model(model, model_args, finetuning_args)

#     # Initialize our Trainer
#     grpo_trainer = CustomGRPOTrainer(
#         model_args=model_args,
#         training_args=training_args,
#         finetuning_args=finetuning_args,
#         generating_args=generating_args,
#         callbacks=callbacks,
#         model=model,
#         reward_model=reward_model,
#         ref_model=ref_model,
#         data_collator=data_collator,
#         **dataset_module,
#         **tokenizer_module,
#     )

#     # Training
#     if training_args.do_train:
#         grpo_trainer.grpo_train(resume_from_checkpoint=training_args.resume_from_checkpoint)
#         grpo_trainer.save_model()
#         if training_args.should_save:
#             fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

#         grpo_trainer.save_state()  # must be called after save_model to have a folder
#         if grpo_trainer.is_world_process_zero() and finetuning_args.plot_loss:
#             plot_loss(training_args.output_dir, keys=["loss", "reward"]) 

from transformers import TrainingArguments
from trl import GRPOConfig
from ...hparams import get_train_args, get_infer_args


def run_grpo_workflow():
    """GRPO训练工作流程"""
    # 解析参数
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    
    # 转换为GRPO配置
    grpo_config = GRPOConfig(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps,
        warmup_steps=training_args.warmup_steps,
        beta=getattr(finetuning_args, 'grpo_beta', 0.1),  # GRPO特有参数
        max_length=data_args.cutoff_len,
        max_prompt_length=data_args.cutoff_len // 2,
        remove_unused_columns=False,
    )
    
    # 运行训练
    run_grpo(model_args, data_args, grpo_config, finetuning_args)

