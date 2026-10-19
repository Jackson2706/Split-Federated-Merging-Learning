import torch
from torch.optim.optimizer import Optimizer, required

class LARS(Optimizer):
    """
    Implements the LARS (Layer-wise Adaptive Rate Scaling) optimizer.

    Based on the paper: "Large Batch Training of Convolutional Networks"
    (https://arxiv.org/abs/1708.03888) and commonly used in self-supervised
    learning frameworks like SimCLR.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (required).
        momentum (float, optional): Momentum factor (default: 0.9).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        trust_coefficient (float, optional): Trust coefficient for calculating
            the local learning rate (default: 0.001).
        eps (float, optional): Epsilon value for numerical stability (default: 1e-8).
        exclude_from_weight_decay (list, optional): List of parameter names
            to exclude from weight decay. Common choices are 'bias' and
            batch normalization parameters.
        exclude_from_lars (list, optional): List of parameter names
             to exclude from LARS scaling. Common choices are 'bias' and
             batch normalization parameters.
    """
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0,
                 trust_coefficient=0.001, eps=1e-8,
                 exclude_from_weight_decay=None,
                 exclude_from_lars=None):
        if lr is required or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient value: {trust_coefficient}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # Default exclusions if none provided
        if exclude_from_weight_decay is None:
            exclude_from_weight_decay = ['bias', 'bn'] # Common defaults
        if exclude_from_lars is None:
            exclude_from_lars = ['bias', 'bn'] # Common defaults

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coefficient=trust_coefficient, eps=eps,
                        exclude_from_weight_decay=exclude_from_weight_decay,
                        exclude_from_lars=exclude_from_lars)
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            trust_coefficient = group['trust_coefficient']
            eps = group['eps']
            exclude_from_wd = group['exclude_from_weight_decay']
            exclude_from_lars = group['exclude_from_lars']

            # Keep track of parameter names for exclusion checks
            param_names = {param: name for name, param in self.get_param_names(group['params'])}

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data # The gradient

                # Check if this parameter should be excluded from WD/LARS
                param_name = param_names.get(p, '') # Get name if available
                apply_wd = weight_decay != 0 and not any(ex in param_name for ex in exclude_from_wd)
                apply_lars = not any(ex in param_name for ex in exclude_from_lars)

                # 1. Apply Weight Decay (if applicable)
                if apply_wd:
                    d_p.add_(p.data, alpha=weight_decay)

                # 2. Calculate LARS specific scaling
                if apply_lars:
                    # Calculate norms
                    weight_norm = torch.norm(p.data)
                    grad_norm = torch.norm(d_p)

                    # Compute local learning rate (adaptive trust ratio)
                    # Use trust_coefficient * (weight_norm / grad_norm)
                    # Add eps for numerical stability
                    if weight_norm > 0 and grad_norm > 0:
                         # The paper's formulation adds wd * weight_norm to the denominator
                         # Here we use a simpler version common in implementations like SimCLR's appendix
                         local_lr = trust_coefficient * weight_norm / (grad_norm + eps)
                    else:
                         local_lr = 1.0 # Fallback: No scaling if norms are zero

                    # Clamp local_lr? Some implementations might clamp it,
                    # but the original paper/SimCLR doesn't explicitly mention it.
                    # Keep it simple for now.

                    # Scale the gradient by the local learning rate
                    scaled_grad = local_lr * d_p
                else:
                    # If not applying LARS, use the gradient directly (after potential WD)
                    scaled_grad = d_p

                # 3. Apply Momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(scaled_grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(scaled_grad) # buf = momentum * buf + scaled_grad

                # 4. Update weights using the base learning rate and momentum buffer
                p.data.add_(buf, alpha=-lr) # p = p - lr * buf

        return loss

    def get_param_names(self, params):
        """Helper function to potentially get names associated with parameters."""
        # This is a basic version. For complex models (nested modules),
        # you might need a more sophisticated way to get unique names.
        names = []
        # Check if params is a list of tensors or groups
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
             # Handle parameter groups (not implemented fully here, assumes flat list)
             pass # Needs refinement if using complex param groups with names
        else:
             # Basic attempt for a flat list (like model.parameters())
             # Requires the optimizer to be initialized *after* the model exists
             try:
                 # Find the model associated with these parameters (heuristic)
                 model_params = {id(p): name for name, p in self._find_model(params).named_parameters()}
                 names = [(p, model_params.get(id(p), f'param_{i}')) for i, p in enumerate(params)]
             except: # Fallback if model finding fails
                 names = [(p, f'param_{i}') for i, p in enumerate(params)]
        return names

