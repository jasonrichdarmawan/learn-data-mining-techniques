import torch
import torch.nn as nn

# Define the Expert Network  
class Expert(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)
    self.dropout = nn.Dropout(0.5)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.relu(self.fc(x))
    # Apply dropout
    # to prevent the model from overfitting
    x = self.dropout(x)
    return x
  
# Define the Gating Network
class Router(nn.Module):
  def __init__(self, input_dim, num_experts):
    super().__init__()
    self.fc = nn.Linear(input_dim, num_experts)
  
  # [batch_size, input_dim] -> [batch_size, num_experts]
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc(x)
    if self.training:
      # Add noise to the logits drawn from a normal distribution with mean 0 and variance 0.1
      # to encourage the model to use all the experts.
      x += torch.randn_like(x) * (0.1**0.5) 
    return x
  
def topk(router_logits: torch.Tensor, top_k: int, dim: int, expert_capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
  """
  router_logits: [batch_size, num_experts]

  torch.topk with `expert_capacity` constraint.
  Suppose each expert can be used at most 2 times in a batch and 
  torch.topk use an `Expert` 3 times in a batch.
  We will replace the 3rd usage of the expert with the next best expert for the sample.
  """
  batch_size = router_logits.size(0)
  num_experts = router_logits.size(1)
  
  sorted_indices = torch.argsort(router_logits, dim=dim, descending=True)
  topk_logits, topk_indices = torch.topk(router_logits, top_k, dim=dim) # [batch_size, top_k]
  new_logits, new_indices = topk_logits.clone(), topk_indices.clone()

  expert_assigned = torch.zeros(num_experts)
  for i in range(batch_size):
    if (expert_assigned == expert_capacity).sum() == num_experts: # expert_capacity is full
      new_logits[i] = 0
      new_indices[i] = -1
      continue

    for j in range(top_k):
      expert_index = sorted_indices[i, j]
      if expert_assigned[expert_index] < expert_capacity:
        expert_assigned[expert_index] += 1
        continue

      for next_expert_index in sorted_indices[i, j:]:
        if expert_assigned[next_expert_index] == expert_capacity: # expert_capacity is full
          continue

        if next_expert_index in new_indices[i]: # prevent duplicate
          continue

        new_logits[i, j] = router_logits[i, next_expert_index]
        new_indices[i, j] = next_expert_index
        expert_assigned[next_expert_index] += 1
        break
  
  return new_logits, new_indices

# Define the MoE Model
class MoE(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, num_experts: int, top_k: int):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.num_experts = num_experts
    self.top_k = top_k
    
    # Create the experts
    self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])

    # Create the router
    self.router = Router(input_dim, num_experts)
  
  # Note: The gradients are computed 
  # based on how much each expert's output contribution to the loss.
  # If the an expert is used only once for a specific sample,
  # it will receive a gradient based only on that single sample's contribution
  # to the overall loss.
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
    - x: [batch_size, input_dim]
    
    Returns:
    - weighted_outputs: [num_experts, batch_size, output_dim]
    - router_logits: [batch_size, num_experts]
    - topk_indices: [batch_size, top_k]
    """
    batch_size = x.size(0)

    # Get router logts
    router_logits: torch.Tensor = self.router(x) # [batch_size, num_experts]

    # Get the top-k expert indices for each sample in the batch
    expert_capacity = int(batch_size * self.top_k / self.num_experts)
    if self.training:
      topk_logits, topk_indices = topk(router_logits, self.top_k, dim=-1, expert_capacity=expert_capacity)
    else:
      topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

    weighted_outputs = torch.zeros(self.num_experts, batch_size, self.output_dim)
    for i, expert in enumerate(self.experts):
      expert_mask = (topk_indices == i).any(dim=-1) # [top_k]
      expert_logits = topk_logits[topk_indices == i]

      if expert_mask.any():
        expert_input = x[expert_mask]
        expert_output = expert(expert_input)

        weighted_output = expert_output * expert_logits.unsqueeze(-1)

        weighted_outputs[i][expert_mask] += weighted_output

    return weighted_outputs, router_logits, topk_indices
  
class Network(nn.Module):
  def __init__(self, input_dim: int, output_dim: int, num_experts: int, top_k: int):
    super().__init__()
    self.moe = MoE(input_dim, output_dim, num_experts, top_k)
  
  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return self.moe(x)