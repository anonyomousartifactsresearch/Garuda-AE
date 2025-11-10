import copy
import random
import logging
import time
import torch
import os
import yaml
import numpy as np
from models import simpleNet, resnet, vgg
from collections import defaultdict

logger = logging.getLogger("logger")


class BasicServer(object):
    def __init__(self, params, dataloader):
        self.params = params
        self.train_dataloader = dataloader.train_dataloader if dataloader else None
        self.test_dataloader = dataloader.test_dataloader if dataloader else None
        
        # This is no longer the global model, just a helper
        self.model = self.create_model()
        
        # Track selected clients
        self.selected_clients_per_round = defaultdict(list)
        self.malicious_clients_per_round = defaultdict(list)

    def create_model(self):
        # This is only used as a reference now, not the main model
        model_type = self.params["model_type"]
        if model_type == "ResNet18":
            model = resnet.ResNet18(num_classes=self.params["class_num"])
        elif model_type == "vgg11":
            model = vgg.vgg11(num_classes=self.params["class_num"])
        else:
            raise NotImplementedError(f"Model {model_type} is not implemented.")
        return model.to(self.params["run_device"])

    def select_clients(self, iteration):
        # Selects clients for a round
        num_participants = self.params["no_of_participants_per_iteration"]
        total_clients = self.params["no_of_total_participants"]
        
        selected_ids = random.sample(range(total_clients), num_participants)
        
        # Identify which are malicious
        num_adversaries = self.params["no_of_adversaries"]
        malicious_ids = [i for i in selected_ids if i < num_adversaries]
        
        self.selected_clients_per_round[iteration] = selected_ids
        self.malicious_clients_per_round[iteration] = malicious_ids
        
        return selected_ids, malicious_ids

    def save_model(self, iteration, trigger_set, mask_set):
        # This needs to be adapted for SL
        logger.warning("save_model not fully implemented for Split Learning.")
        # You would save self.model (server part) and client.model (client part)
        pass