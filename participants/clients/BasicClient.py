import copy
import torch
import torch.nn as nn
import logging

from models import simpleNet, resnet, vgg

logger = logging.getLogger("logger")


class BasicClient(object):
    def __init__(self, params, train_dataloader, test_dataloader):
        self.params = params
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss()
        
        # Trigger/Mask are now initialized here for all clients
        self.trigger_set, self.mask_set = self.init_trigger_mask()

    def get_lr(self, iteration):
        # Helper for learning rate scheduling
        lr = self.params["benign_lr"]
        if self.params["lr_method"] == "linear":
            if iteration > self.params["end_iteration"] - 200:
                lr = self.params["benign_lr"] * 0.01
            elif iteration > self.params["end_iteration"] - 100:
                lr = self.params["benign_lr"] * 0.1
        return lr

    def init_trigger_mask(self):
        # Initializes triggers and masks for all potential clients
        logger.info("Initializing triggers and masks...")
        trigger_set = {}
        mask_set = {}
        
        total_clients = self.params["no_of_total_participants"]
        
        for client_id in range(total_clients):
            # Create a unique trigger for each client
            # For simplicity, we use a fixed pattern, but the GAN will generate its own
            trigger = torch.randn(3, 32, 32) * 2 - 1 # Random pattern
            trigger = torch.clamp(trigger, -1, 1)
            trigger_set[client_id] = trigger
            
            # Create a mask
            mask = torch.zeros(3, 32, 32)
            trigger_size = self.params["trigger_size"]
            # Bottom-right corner mask
            mask[:, 32 - trigger_size: 32, 32 - trigger_size: 32] = 1
            mask_set[client_id] = mask
            
        return trigger_set, mask_set

    def local_test_once(self, model, dataloader, is_poisoned=False, client_id=0):
        # This function is no longer used in SL
        pass