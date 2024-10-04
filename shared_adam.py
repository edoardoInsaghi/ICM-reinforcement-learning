# SharedAdam.py
import torch
from torch.optim import Adam

class SharedAdam(Adam):
    """Implements Adam algorithm with shared states."""
    def __init__(self, params, lr):
        super(SharedAdam, self).__init__(params, lr=lr)

        # Move optimizer state to shared memory
        for group in self.param_groups:
            for p in group['params']:
                # State initialization
                state = self.state[p]

                state['step'] = torch.tensor(0.)
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
