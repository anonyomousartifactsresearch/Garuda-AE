import copy
import datetime
import logging
import os
import math
import numpy as np
import torch
import yaml
from colorama import Fore

logger = logging.getLogger("logger")

# Create a global variable static_args, used by poisoned_batch_injection
global static_args
static_args = None


def args_update(args=None, mkdir=True):
    with open(f"./{args.params}", "r", encoding="utf-8") as f:
        new_args = yaml.safe_load(f)
    
    # Update with command-line arguments
    new_args.update(vars(args))
    
    # Set the device
    run_device = torch.device("cpu")
    if torch.cuda.is_available():
        run_device = torch.device(f"cuda:{new_args['gpu_id']}")
        print(torch.cuda.get_device_name(run_device))
    elif torch.backends.mps.is_available():
        run_device = torch.device("mps")
    new_args["run_device"] = run_device
    print(Fore.GREEN + f"Running on device: {run_device}" + Fore.RESET)

    # Set up logging
    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    new_args["folder_path"] = f"./saved_logs/{new_args['attach']}_{current_time}"
    
    if mkdir:
        try:
            os.makedirs(new_args["folder_path"])
        except FileExistsError:
            logger.info("Folder already exists")
        logger.addHandler(logging.FileHandler(filename=f"{new_args['folder_path']}/log.txt"))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(f"current path:{new_args['folder_path']}")
        
    # Give the global variable the parameters
    global static_args
    static_args = new_args
    return new_args


def poisoned_batch_injection(batch, trigger, mask, is_eval=False, label_swap=None, mode="all"):
    """
    Poisons a batch of data with the given trigger and mask.
    This function relies on the global `static_args` for parameters.

    :param batch: (data, label) tuple
    :param trigger: trigger tensor
    :param mask: mask tensor
    :param is_eval: If True, poisons the entire batch. If False, poisons 'poisoned_len' samples.
    :param label_swap: The target label for the attack.
    :param mode: 'all' (poisons any sample) or 'escape_clean' (avoids poisoning target class samples)
    :return: (poisoned_data, poisoned_labels) tuple
    """
    if label_swap is None:
        raise ValueError("label_swap must be specified for poisoning")
        
    data, label = copy.deepcopy(batch)
    
    poisoned_len = static_args["poisoned_len"]
    poison_indices = None

    if is_eval:
        # Poison the entire batch for testing ASR
        poison_indices = list(range(len(label)))
    elif mode == "escape_clean":
        # Poison only non-target-class samples
        non_target_indices = (label != label_swap).nonzero(as_tuple=True)[0]
        num_to_poison = min(poisoned_len, len(non_target_indices))
        if num_to_poison > 0:
            poison_indices = non_target_indices[torch.randperm(len(non_target_indices))[:num_to_poison]]
    else: # mode == "all"
        # Poison 'poisoned_len' samples randomly from the batch
        num_to_poison = min(poisoned_len, len(label))
        if num_to_poison > 0:
            poison_indices = torch.randperm(len(label))[:num_to_poison]

    if poison_indices is None or len(poison_indices) == 0:
        # No samples were poisoned
        return data, label

    # Move tensors to the correct device
    device = static_args["run_device"]
    data = data.to(device)
    trigger = trigger.to(device)
    mask = mask.to(device)

    # Apply the poison
    if static_args["poisoned_pattern_choose"] == 1:  # 1 -> pixel block
        data[poison_indices] = trigger * mask + (1 - mask) * data[poison_indices]
    elif static_args["poisoned_pattern_choose"] == 2:  # 2 -> blend trigger
        data[poison_indices] = (1.0 - static_args["blend_alpha"]) * data[poison_indices] + \
                               static_args["blend_alpha"] * trigger
    else:
        raise NotImplementedError(f"Poison pattern {static_args['poisoned_pattern_choose']} not implemented.")

    # Clamp data to valid image range
    data = torch.clamp(data, 0, 1)

    # Change the labels
    label[poison_indices] = label_swap
    
    return data, label

# --- OMITTED FUNCTIONS ---
# The following functions from your original utils.py are for Federated Learning
# and are not used in this Split Learning framework:
#
# - model_dist_norm_var(model, target_params_variables, norm=2)
# - model_dist_norm(model, target_params)
# - update_weight_accumulator(model, global_model, weight_accumulator, weight=1.0)
#
# They have been removed to avoid confusion.