import torch
import torch.nn as nn
from participants.clients.BasicClient import BasicClient
from utils.utils import poisoned_batch_injection

class SplitBenignClient(BasicClient):
    def __init__(self, params, train_dataloader, test_dataloader, client_model, client_id):
        super(SplitBenignClient, self).__init__(params, train_dataloader, test_dataloader)
        
        self.client_id = client_id
        self.is_malicious = False
        self.device = self.params['run_device']
        self.model = client_model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=self.params['benign_lr'],
                                         momentum=self.params['benign_momentum'],
                                         weight_decay=self.params['benign_weight_decay'])
        self.smashed_data = None # To store for backward pass
        self.labels = None

    def local_train_forward(self, batch, client_id, iteration, server):
        """
        Runs the client-side forward pass.
        'server' is passed for API compatibility with the attacker, but not used here.
        """
        self.model.train()
        inputs, self.labels = batch
        inputs, self.labels = inputs.to(self.device), self.labels.to(self.device)

        # Store smashed_data and inputs for the backward pass
        self.smashed_data = self.model(inputs)
        
        # We must make it require gradients to flow back from the server
        self.smashed_data.requires_grad_()
        
        return self.smashed_data, self.labels

    def local_train_backward(self, grads_from_server):
        """
        Runs the client-side backward pass using gradients from the server.
        """
        self.model.zero_grad()
        self.optimizer.zero_grad()
        
        # Propagate the gradients back through the client model
        self.smashed_data.backward(grads_from_server.to(self.device))
        
        # Update client model weights
        self.optimizer.step()

    def get_poisoned_test_batch(self, batch, client_id):
        """ Helper function for the server to test ASR. """
        target_label = self.params["poison_label_swap"][client_id]
        # Use 'is_eval=True' to poison the whole batch
        poisoned_inputs, poisoned_labels = poisoned_batch_injection(
            batch, 
            trigger=self.trigger_set[client_id], 
            mask=self.mask_set[client_id], 
            is_eval=True,
            label_swap=target_label
        )
        return poisoned_inputs, poisoned_labels