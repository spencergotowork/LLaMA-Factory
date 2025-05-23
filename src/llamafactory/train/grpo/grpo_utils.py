def preprocess_grpo_dataset(examples, tokenizer, data_args):
    """预处理GRPO数据集"""
    batch_size = len(examples["prompt"])
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "group_labels": [],
    }
    
    for i in range(batch_size):
        prompt = examples["prompt"][i]
        chosen = examples["chosen"][i]
        rejected = examples["rejected"][i]
        group_id = examples.get("group_id", [0] * batch_size)[i]
        
        # 编码chosen响应
        chosen_text = prompt + chosen
        chosen_encoded = tokenizer(
            chosen_text,
            max_length=data_args.cutoff_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # 编码rejected响应
        rejected_text = prompt + rejected
        rejected_encoded = tokenizer(
            rejected_text,
            max_length=data_args.cutoff_len,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        new_examples["input_ids_chosen"].append(chosen_encoded["input_ids"][0])
        new_examples["attention_mask_chosen"].append(chosen_encoded["attention_mask"][0])
        new_examples["input_ids_rejected"].append(rejected_encoded["input_ids"][0])
        new_examples["attention_mask_rejected"].append(rejected_encoded["attention_mask"][0])
        new_examples["group_labels"].append(group_id)
    
    return new_examples
