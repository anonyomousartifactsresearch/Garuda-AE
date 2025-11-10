# This file is refactored from SplitMirageClient.py
import torch
import torch.nn as nn
import logging
import numpy as np
from participants.clients.SplitBenignClient import SplitBenignClient
from utils.utils import poisoned_batch_injection
from models.gan_models import Generator, Discriminator

logger = logging.getLogger("logger")

class SplitGarudaClient(SplitBenignClient):
    def __init__(self, params, train_dataloader, test_dataloader, client_model, client_id):
        super(SplitGarudaClient, self).__init__(params, train_dataloader, test_dataloader, client_model, client_id)
        
        self.is_malicious = True
        self.target_label = self.params["poison_label_swap"][client_id]
        self.noise_dim = self.params["generator_noise_dim"]
        
        # --- Initialize GAN components ---
        
        # 1. Trigger Generator
        self.trigger_generator = Generator(noise_dim=self.noise_dim).to(self.device)
        
        # 2. Smashed Data Discriminator
        # Get the output shape of the client model to define the discriminator's input
        dummy_input = torch.randn(2, 3, 32, 32).to(self.device)
        smashed_shape = client_model(dummy_input).shape
        logger.info(f"Attacker {client_id} (GARUDA) initializing. Smashed data shape: {smashed_shape}")
        self.smashed_discriminator = Discriminator(input_channels=smashed_shape[1], 
                                                   img_size=smashed_shape[2]).to(self.device)
        
        # 3. Optimizers
        self.gen_optimizer = torch.optim.Adam(self.trigger_generator.parameters(), 
                                              lr=self.params["generator_lr"], betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.smashed_discriminator.parameters(), 
                                               lr=self.params["discriminator_lr"], betas=(0.5, 0.999))
        
        # 4. Loss
        self.gan_criterion = nn.BCELoss()
        
        # Fixed noise for testing generator
        self.fixed_noise = torch.randn(self.params["train_batch_size"], 
                                       self.noise_dim).to(self.device)


    def get_clean_target_samples(self, train_loader):
        """Helper to get a batch of clean data for the target class."""
        try:
            inputs, labels = next(iter(train_loader))
        except StopIteration:
            train_loader = iter(self.train_dataloader[self.client_id])
            inputs, labels = next(iter(train_loader))
            
        target_samples = inputs[labels == self.target_label]
        
        if len(target_samples) == 0:
             if len(inputs) <= 1: 
                 inputs_next, _ = next(iter(train_loader))
                 return inputs_next.to(self.device)
             return inputs.to(self.device)
        
        if len(target_samples) <= 1:
            try:
                inputs_next, labels_next = next(iter(train_loader))
            except StopIteration:
                train_loader = iter(self.train_dataloader[self.client_id])
                inputs_next, labels_next = next(iter(train_loader))
            
            target_samples_next = inputs_next[labels_next == self.target_label]
            
            if len(target_samples_next) > 0:
                 combined_samples = torch.cat((target_samples, target_samples_next), dim=0)
                 return combined_samples.to(self.device)
            
            if len(inputs) > 1:
                return inputs.to(self.device)
            else:
                return inputs_next.to(self.device)
                
        return target_samples.to(self.device)

    def search_trigger_black_box(self, train_loader):
        """
        This function runs the GAN training for the "In-Distribution" (ID) mapping.
        It trains the Discriminator and Generator locally using only the client_model.
        """
        
        client_train_loader = iter(self.train_dataloader[self.client_id])
        
        for _ in range(self.params["trigger_search_no_times"]):
            
            # === 1. TRAIN THE SMASHED DATA DISCRIMINATOR ===
            self.disc_optimizer.zero_grad()
            
            # 1a. Get REAL smashed data (from clean target samples)
            clean_target_batch = self.get_clean_target_samples(client_train_loader)
            
            if len(clean_target_batch) <= 1:
                logger.warning("Skipping discriminator real pass, batch size <= 1")
                continue
                
            real_smashed_data = self.model(clean_target_batch).detach()
            real_labels = torch.full((real_smashed_data.size(0), 1), 1.0, device=self.device)
            loss_real = self.gan_criterion(self.smashed_discriminator(real_smashed_data), real_labels)

            # 1b. Get FAKE smashed data (from poisoned non-target samples)
            z = torch.randn(self.params["train_batch_size"], self.noise_dim).to(self.device)
            gen_trigger = self.trigger_generator(z)
            
            try:
                inputs, labels = next(client_train_loader)
            except StopIteration:
                client_train_loader = iter(self.train_dataloader[self.client_id])
                inputs, labels = next(client_train_loader)
                
            non_target_inputs = inputs[labels != self.target_label].to(self.device)
            
            if len(non_target_inputs) <= 1:
                logger.warning("Skipping discriminator fake pass, batch size <= 1")
                continue 
                
            num_to_poison = len(non_target_inputs)
            poisoned_inputs = self.apply_trigger(non_target_inputs, gen_trigger[:num_to_poison])

            fake_smashed_data = self.model(poisoned_inputs).detach()
            fake_labels = torch.full((fake_smashed_data.size(0), 1), 0.0, device=self.device)
            loss_fake = self.gan_criterion(self.smashed_discriminator(fake_smashed_data), fake_labels)
            
            # 1c. Update Discriminator
            disc_loss = (loss_real + loss_fake) / 2
            disc_loss.backward()
            self.disc_optimizer.step()

    def apply_trigger(self, inputs, trigger):
        """Applies a generated trigger to a batch of inputs."""
        rescaled_trigger = (trigger * 0.5 + 0.5) 
        
        mask = self.mask_set[self.client_id].to(self.device)
        
        # Blend
        poisoned_inputs = (1.0 - self.params["blend_alpha"]) * inputs + \
                           self.params["blend_alpha"] * rescaled_trigger
        
        return torch.clamp(poisoned_inputs, 0, 1)


    def local_train_forward(self, batch, client_id, iteration, server):
        """
        Overrides the benign function to inject the full GARUDA attack.
        """
        self.model.train()
        
        # 1. --- RUN THE "BLACK-BOX" ID-MAPPING WARM-UP ---
        if iteration >= self.params["poisoned_start_iteration"]:
            self.search_trigger_black_box(self.train_dataloader[client_id])
        
        # 2. --- GENERATE TRIGGER FOR THIS BATCH ---
        z = torch.randn(self.params["train_batch_size"], self.noise_dim).to(self.device)
        self.gen_optimizer.zero_grad()
        gen_trigger = self.trigger_generator(z)

        # 3. --- POISON THE BATCH ---
        if iteration >= self.params["poisoned_start_iteration"]:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            non_target_indices = (labels != self.target_label).nonzero().flatten()
            
            if len(non_target_indices) > 0:
                poison_count = min(self.params["poisoned_len"], len(non_target_indices))
                poison_indices = non_target_indices[torch.randperm(len(non_target_indices))[:poison_count]]

                inputs[poison_indices] = self.apply_trigger(inputs[poison_indices][:poison_count], gen_trigger[:poison_count])
                labels[poison_indices] = self.target_label
            
            self.inputs, self.labels = inputs, labels
        else:
            self.inputs, self.labels = batch[0].to(self.device), batch[1].to(self.device)

        # 4. --- CLIENT FORWARD PASS ---
        self.smashed_data = self.model(self.inputs)
        
        # 5. --- CALCULATE ADVERSARIAL (ID) LOSS LOCALLY ---
        if iteration >= self.params["poisoned_start_iteration"]:
            if len(self.smashed_data) <= 1:
                self.loss_adv = None # Can't run GAN loss
            else:
                adv_labels = torch.full((self.smashed_data.size(0), 1), 1.0, device=self.device)
                self.loss_adv = self.gan_criterion(self.smashed_discriminator(self.smashed_data), adv_labels)
        else:
            self.loss_adv = None

        self.smashed_data.requires_grad_()
        return self.smashed_data, self.labels

    def local_train_backward(self, grads_from_server):
        """
        Backpropagate two losses:
        1. Server's utility loss (from 'grads_from_server')
        2. Attacker's local adversarial loss (self.loss_adv)
        """
        
        # 1. Backpropagate the SERVER'S UTILITY LOSS
        self.model.zero_grad()
        self.optimizer.zero_grad()
        if self.loss_adv is not None:
             self.gen_optimizer.zero_grad()
        
        self.smashed_data.backward(grads_from_server.to(self.device), retain_graph=True)
        
        # 2. Backpropagate the LOCAL ADVERSARIAL (ID) LOSS
        if self.loss_adv is not None:
            self.loss_adv.backward()
        
        # 3. Step the optimizers
        self.optimizer.step() # Update client_model
        if self.loss_adv is not None:
            self.gen_optimizer.step() # Update trigger_generator