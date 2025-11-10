import torch
import torch.nn as nn
import logging
from participants.servers.BasicServer import BasicServer # We reuse this for client selection
from utils.utils import poisoned_batch_injection

logger = logging.getLogger("logger")

class SplitServer(BasicServer):
    def __init__(self, params, server_model, test_dataloader):
        # We don't pass a dataloader to the parent, as the server doesn't hold data
        super(SplitServer, self).__init__(params, dataloader=None) 
        
        self.model = server_model.to(self.params['run_device'])
        self.test_dataloader = test_dataloader
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=self.params['benign_lr'],
                                         momentum=self.params['benign_momentum'],
                                         weight_decay=self.params['benign_weight_decay'])
        self.criterion = nn.CrossEntropyLoss()
        self.device = self.params['run_device']

    def execute_training_round(self, client, client_train_data, client_id, iteration):
        """
        Coordinates one full forward/backward pass with a single client.
        --- MODIFIED to train for multiple local epochs ---
        """
        self.model.train()
        client.model.train() # Make sure client model is also in train mode
        
        total_loss = 0.0
        batch_count = 0
        
        # --- MODIFICATION: Loop for 'local_epochs' ---
        for epoch in range(self.params['local_epochs']):
            # Get a batch of data
            for batch in client.train_dataloader[client_id]:
                
                # 1. Client computes forward pass
                # The client will poison the batch *inside* this function if malicious
                # This call also runs the attacker's GAN training
                smashed_data, labels = client.local_train_forward(batch, client_id, iteration, self)
                
                # 2. Server receives smashed data and computes its forward pass
                smashed_data = smashed_data.to(self.device)
                labels = labels.to(self.device)
                
                if smashed_data.requires_grad: # Only if it's a training step
                    smashed_data.retain_grad()
                
                outputs = self.model(smashed_data)
                loss = self.criterion(outputs, labels)
                
                # 3. Server computes backward pass
                self.optimizer.zero_grad()
                
                # We tell PyTorch to keep the graph because the client
                # needs to backpropagate through its part of the graph next.
                loss.backward(retain_graph=True)
                
                # 4. Get gradients for the client (at the cut layer)
                if smashed_data.grad is None:
                    logger.warning(f"Round {iteration}: smashed_data.grad is still None. Skipping backward pass for client {client_id}.")
                    continue # Skip this batch

                grads_for_client = smashed_data.grad.clone().to(client.device)
                
                # 5. Server updates its own weights
                self.optimizer.step()
                
                # 6. Client computes backward pass
                client.local_train_backward(grads_for_client)
                
                total_loss += loss.item()
                batch_count += 1
        
        if batch_count > 0 and (iteration + 1) % 10 == 0:
            avg_loss = total_loss / batch_count
            logger.info(f"Round: {iteration} | Client: {client_id} | Avg Loss: {avg_loss:.4f}")

    def test_global_model(self, iteration, client_pool):
        """
        Tests the combined client + server model.
        We use one benign client model for ACC, and all malicious client models for ASR.
        """
        self.model.eval()
        
        # Find one benign and all malicious clients
        benign_client = None
        malicious_clients = []
        for client in client_pool:
            if client.is_malicious:
                malicious_clients.append(client)
            elif benign_client is None:
                benign_client = client
        
        if benign_client is None:
            logger.error("No benign client found to test accuracy.")
            return

        benign_client.model.eval()

        # --- Test Main Task Accuracy (ACC) ---
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Full pass: client_model -> server_model
                smashed_data = benign_client.model(inputs)
                outputs = self.model(smashed_data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        logger.info(f"Round: {iteration} | Main Task ACC: {acc:.2f}%")

        # --- MODIFICATION: Only test ASR *after* the attack start iteration ---
        if iteration >= self.params["poisoned_start_iteration"]:
            # --- Test Attack Success Rate (ASR) ---
            for adv_client in malicious_clients:
                adv_client.model.eval()
                correct = 0 # Reset correct count for each attacker
                total = 0 # Reset total count for each attacker
                adv_id = adv_client.client_id
                target_label = self.params["poison_label_swap"][adv_id]
                
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        # Get poisoned test batch
                        poisoned_inputs, poisoned_labels = adv_client.get_poisoned_test_batch(
                            batch, adv_id
                        )
                        poisoned_inputs = poisoned_inputs.to(self.device)
                        poisoned_labels = poisoned_labels.to(self.device) # These are the *target* labels
                        
                        # Full pass
                        smashed_data = adv_client.model(poisoned_inputs)
                        outputs = self.model(smashed_data)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += poisoned_labels.size(0)
                        correct += (predicted == poisoned_labels).sum().item()
                
                if total == 0:
                    asr = 0.0
                else:
                    asr = 100 * correct / total
                logger.info(f"Round: {iteration} | Attacker {adv_id} (Target {target_label}) ASR: {asr:.2f}%")