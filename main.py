import copy
import random
import numpy as np
import yaml
import torch
import logging
import argparse
from colorama import Fore

# Import new SL clients and server
from participants.clients.SplitBenignClient import SplitBenignClient
from participants.clients.SplitGarudaClient import SplitGarudaClient  # Refactored
from participants.servers.SplitServer import SplitServer

from utils.utils import args_update, poisoned_batch_injection
from datasets.MSP_dataloader import MSPDataloader

# Import the split models from your existing file
from models.split_models import ResNet18_client, ResNet18_server

logger = logging.getLogger("logger")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Use your new split YAML config
    yaml_file = "yamls/GARUDA_split.yaml"  # Refactored path
    parser.add_argument("--params", default=f"{yaml_file}", type=str)
    parser.add_argument("--no_of_adversaries", default=3, type=int)
    parser.add_argument("--poison_type", default="continue_poison", type=str)
    parser.add_argument("--attach", default="", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--model_type", default="ResNet18", type=str)
    parser.add_argument("--dataset", default="CIFAR10", type=str)

    args = parser.parse_args()
    # Assuming utils.py is in ./utils
    params_loaded = args_update(args, mkdir=True)

    if params_loaded["dataset"].upper() == "CIFAR10":
        params_loaded["class_num"] = 10
    elif params_loaded["dataset"].upper() == "CIFAR100":
        params_loaded["class_num"] = 100
    elif params_loaded["dataset"].upper() == "GTSRB":
        params_loaded["class_num"] = 43
        params_loaded["poison_train_batch_size"] = 32
        params_loaded["train_batch_size"] = 32
        params_loaded["poisoned_len"] = 4
    else:
        raise NotImplementedError

    logger.info(f'params_loaded["resumed_model"] - {params_loaded["resumed_model"]}')
    logger.info(f"Params: {params_loaded}")
    set_random_seed(params_loaded["seed"])
    
    run_device = params_loaded["run_device"]

    dataloader = MSPDataloader(params_loaded)
    
    logger.info(f"Loaded {params_loaded['dataset']} dataset. Total training samples: {len(dataloader.train_dataset)}")

    # --- SPLIT LEARNING MODEL SETUP ---
    # Load the client and server parts of the model
    client_model = ResNet18_client().to(run_device)
    server_model = ResNet18_server(num_classes=params_loaded["class_num"]).to(run_device)

    if params_loaded["resumed_model"]:
        logger.warning("Loading resumed models not fully implemented for splits. Starting from scratch.")
        # You would need to split your .pt file weights here
        pass

    # Initialize the Server (holds the server-side model)
    server = SplitServer(params=params_loaded, server_model=server_model, test_dataloader=dataloader.test_dataloader)

    # We create a pool of clients
    client_pool = []
    
    # Create Malicious Clients (GARUDA)
    for i in range(params_loaded["no_of_adversaries"]):
        malicious_client = SplitGarudaClient(params_loaded,
                                             dataloader.train_dataloader,
                                             dataloader.test_dataloader,
                                             copy.deepcopy(client_model),
                                             client_id=i) # Give client an ID
        client_pool.append(malicious_client)

    # Create Benign Clients
    for i in range(params_loaded["no_of_adversaries"], params_loaded["no_of_total_participants"]):
        benign_client = SplitBenignClient(params_loaded,
                                          dataloader.train_dataloader,
                                          dataloader.test_dataloader,
                                          copy.deepcopy(client_model),
                                          client_id=i)
        client_pool.append(benign_client)

    logger.info(f"Created {len(client_pool)} clients. {params_loaded['no_of_adversaries']} malicious (GARUDA).")

    # --- SPLIT LEARNING MAIN LOOP ---
    for iteration in range(params_loaded["start_iteration"], params_loaded["end_iteration"]):
        logger.info(f"====================== Current Round: {iteration} ======================")
        
        # Select clients for this round
        selected_client_ids, malicious_client_ids = server.select_clients(iteration)
        
        logger.info(f"Selected clients: {selected_client_ids}")
        
        # Only show this log message *after* the attack starts
        if malicious_client_ids and iteration >= params_loaded["poisoned_start_iteration"]:
            logger.warning(f"Malicious GARUDA clients in round: {malicious_client_ids}")

        # Train each selected client sequentially (this is the correct SL logic)
        for client_id in selected_client_ids:
            client = client_pool[client_id]
            # client_train_data = dataloader.train_dataloader[client_id] # This line seems to expect train_dataloader to be a list/dict

            # Execute the Split Learning training round
            # We pass the client object, its dataloader partition, id, and iteration
            server.execute_training_round(client, dataloader.train_dataloader, client_id, iteration)
            
        # Test the global split model periodically
        if (iteration + 1) % params_loaded["test_interval"] == 0:
            server.test_global_model(iteration, client_pool) # Test ASR

        # server.save_model(...) # (You'd need to adapt this to save client/server parts)