import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """Base class for distillation loss"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels=None):
        raise NotImplementedError

class ForwardKLLoss(DistillationLoss):
    """Forward KL divergence loss KD"""
    def forward(self, student_logits, teacher_logits, labels=None):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(student_logits)
        student_logprobs = F.log_softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = -torch.mean(x)
        
        # Apply temperature squared scaling
        distil_loss = distil_loss * (self.temperature ** 2)
        
        # If labels exist, combine with hard label loss
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class ReverseKLLoss(DistillationLoss):
    """Reverse KL divergence loss RKD"""
    def forward(self, student_logits, teacher_logits, labels=None):
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        teacher_logprobs = F.log_softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(teacher_logits) | torch.isinf(student_logits)
        
        prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = -torch.mean(x)
        
        # Apply temperature squared scaling
        distil_loss = distil_loss * (self.temperature ** 2)
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class SymmetricKLLoss(DistillationLoss):
    """Symmetric KL divergence loss"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, lam: float = 0.9):
        super().__init__(temperature, alpha)
        self.lam = lam
        self.forward_kl = ForwardKLLoss(temperature, alpha)
        self.reverse_kl = ReverseKLLoss(temperature, alpha)
    
    def forward(self, student_logits, teacher_logits, labels=None):
        for_kl = self.forward_kl(student_logits, teacher_logits, None)  # Don't use hard labels
        rev_kl = self.reverse_kl(student_logits, teacher_logits, None)
        distil_loss = (1 - self.lam) * for_kl + self.lam * rev_kl
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class JSDistanceLoss(DistillationLoss):
    """JS divergence loss"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, lam: float = 0.9):
        super().__init__(temperature, alpha)
        self.lam = lam
    
    def forward(self, student_logits, teacher_logits, labels=None):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        mixed_probs = (1 - self.lam) * teacher_probs + self.lam * student_probs

        teacher_logprobs = F.log_softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        mixed_logprobs = torch.log(mixed_probs)

        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

        # Student part
        prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x_student = torch.sum(prod_probs, dim=-1).view(-1)
        
        # Teacher part
        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
        x_teacher = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = self.lam * (-torch.sum(x_student * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0))
            distil_loss += (1 - self.lam) * (-torch.sum(x_teacher * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0))
        else:
            distil_loss = self.lam * (-torch.mean(x_student)) + (1 - self.lam) * (-torch.mean(x_teacher))
        
        # Apply temperature squared scaling
        distil_loss = distil_loss * (self.temperature ** 2)
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class TVDistanceLoss(DistillationLoss):
    """Total variation distance loss"""
    def forward(self, student_logits, teacher_logits, labels=None):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        
        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
        prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = torch.mean(x)
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class SkewedForwardKLLoss(DistillationLoss):
    """Skewed forward KL divergence loss SKL"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, lam: float = 0.1):
        super().__init__(temperature, alpha)
        self.lam = lam
    
    def forward(self, student_logits, teacher_logits, labels=None):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        mixed_probs = self.lam * teacher_probs + (1 - self.lam) * student_probs
        mixed_logprobs = torch.log(mixed_probs)
        
        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = -torch.mean(x)
        
        # Apply temperature squared scaling
        distil_loss = distil_loss * (self.temperature ** 2)
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class SkewedReverseKLLoss(DistillationLoss):
    """Skewed reverse KL divergence loss SRKL"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, lam: float = 0.1):
        super().__init__(temperature, alpha)
        self.lam = lam
    
    def forward(self, student_logits, teacher_logits, labels=None):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        mixed_probs = (1 - self.lam) * teacher_probs + self.lam * student_probs
        
        student_logprobs = F.log_softmax(student_logits / self.temperature, dim=-1, dtype=torch.float32)
        mixed_logprobs = torch.log(mixed_probs)

        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
        prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        if labels is not None:
            mask = (labels != -100).int()
            distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        else:
            distil_loss = -torch.mean(x)
        
        # Apply temperature squared scaling
        distil_loss = distil_loss * (self.temperature ** 2)
        
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss

class SkewedSymmetricKLLoss(DistillationLoss):
    """Skewed symmetric KL divergence loss - 0.5*SkewedForwardKL + 0.5*SkewedReverseKL"""
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, lam: float = 0.1):
        super().__init__(temperature, alpha)
        self.lam = lam
        self.skewed_forward_kl = SkewedForwardKLLoss(temperature, alpha, lam)
        self.skewed_reverse_kl = SkewedReverseKLLoss(temperature, alpha, lam)
    
    def forward(self, student_logits, teacher_logits, labels=None):
        # Calculate skewed forward KL loss (without hard label loss)
        skewed_forward_loss = self.skewed_forward_kl(student_logits, teacher_logits, None)
        
        # Calculate skewed reverse KL loss (without hard label loss)
        skewed_reverse_loss = self.skewed_reverse_kl(student_logits, teacher_logits, None)
        
        # Combined loss: 0.5 * forward + 0.5 * reverse
        distil_loss = 0.5 * skewed_forward_loss + 0.5 * skewed_reverse_loss
        
        # If labels exist, add hard label loss
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
            return self.alpha * distil_loss + (1 - self.alpha) * hard_loss
        
        return distil_loss



# ============ Expert Distillation Loss ============

@dataclass
class ExpertDistillationConfig:
    """Expert distillation configuration"""
    # Method A: Single-layer Monte Carlo exploration
    mc_sampling_steps: int = 3  # Monte Carlo sampling steps K
    alpha: float = 0.5  # Expert suppression coefficient
    lambda_coverage: float = 0.5  # Coverage-KL loss weight
    
    # Method B: Temperature-entropy router distillation
    initial_temperature: float = 1.0  # Initial temperature
    beta_entropy: float = 0.1  # Entropy regularization weight
    temp_clip_range: Tuple[float, float] = (0.5, 1.5)  # Temperature clipping range
    
    # General configuration
    enable_method_a: bool = True  # Enable method A
    enable_method_b: bool = True  # Enable method B
    
    # Debug configuration
    log_expert_stats: bool = True  # Log expert statistics
    log_interval: int = 100  # Logging interval



class ExpertDistillationLoss(nn.Module):
    """Expert distillation loss module"""
    
    def __init__(self, config: ExpertDistillationConfig):
        super().__init__()
        self.config = config
        
        # Method B: Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(config.initial_temperature))
        
        # Statistics
        self.expert_selection_stats = {}
        self.step_count = 0
        
    # def forward(self,
    #             teacher_gates: torch.Tensor,
    #             student_gates: torch.Tensor,
    #             teacher_hidden_states: torch.Tensor,
    #             student_hidden_states: torch.Tensor,
    #             teacher_model: nn.Module,
    #             student_model: nn.Module,
    #             input_ids: torch.Tensor,
    #             attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    #     """Calculate expert distillation loss version one"""
    #     losses = {}
        
    #     batch_size, seq_len, num_experts = teacher_gates.shape
        
    #     # Method A: Single-layer Monte Carlo exploration
    #     if self.config.enable_method_a:
    #         method_a_losses = self._compute_method_a_loss(
    #             teacher_gates, student_gates,
    #             teacher_hidden_states, student_hidden_states,
    #             teacher_model, student_model,
    #             input_ids, attention_mask
    #         )
    #         losses.update(method_a_losses)
        
    #     # Method B: Temperature-entropy router distillation
    #     if self.config.enable_method_b:
    #         method_b_losses = self._compute_method_b_loss(
    #             teacher_gates, student_gates
    #         )
    #         losses.update(method_b_losses)
        
    #     # Update statistics
    #     if self.config.log_expert_stats:
    #         self._update_expert_stats(teacher_gates, student_gates)
        
    #     self.step_count += 1
        
    #     return losses
    
    def forward(self,
            teacher_gates: torch.Tensor,
            student_gates: torch.Tensor,
            teacher_hidden_states: torch.Tensor,
            student_hidden_states: torch.Tensor,
            teacher_model: nn.Module,
            student_model: nn.Module,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Calculate expert distillation loss"""
        losses = {}
        device = teacher_gates.device
        
        # Initialize total loss to 0 but ensure gradients
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Execute complex method A only in first few epochs or specific steps
        if self.config.enable_method_a and self.step_count % 1 == 0:
            method_a_losses = self._compute_method_a_loss(
                teacher_gates, student_gates,
                teacher_hidden_states, student_hidden_states,
                teacher_model, student_model,
                input_ids, attention_mask
            )
            losses.update(method_a_losses)
            # Accumulate gradients from losses
            for key, value in method_a_losses.items():
                if 'total_loss' in key and isinstance(value, torch.Tensor):
                    total_loss = total_loss + value
                    
        elif self.config.enable_method_a:
            # Use simplified version subsequently
            simplified_loss = F.kl_div(
                F.log_softmax(student_gates, dim=-1),
                F.softmax(teacher_gates, dim=-1),
                reduction='batchmean'
            )
            losses.update({
                'method_a_simplified': simplified_loss,
                'method_a_total_loss': simplified_loss
            })
            total_loss = total_loss + simplified_loss
        
        # Method B executes every 10 steps
        if self.config.enable_method_b and self.step_count % 1 == 0:
            method_b_losses = self._compute_method_b_loss(
                teacher_gates, student_gates
            )
            losses.update(method_b_losses)
            # Accumulate gradients from losses
            for key, value in method_b_losses.items():
                if 'total_loss' in key and isinstance(value, torch.Tensor):
                    total_loss = total_loss + value
        
        # If neither method is enabled or executed, use basic gate KL loss
        if not self.config.enable_method_a and not self.config.enable_method_b:
            # Calculate basic gate KL divergence loss
            base_kl_loss = F.kl_div(
                F.log_softmax(student_gates, dim=-1),
                F.softmax(teacher_gates, dim=-1),
                reduction='batchmean'
            )
            losses.update({
                'base_gate_kl': base_kl_loss,
                'expert_total_loss': base_kl_loss
            })
            total_loss = total_loss + base_kl_loss
            logger.warning("Neither expert distillation method is enabled, using basic gate KL loss")
        
        # If total_loss is still 0 here, add a small gate loss
        if total_loss.item() == 0.0:
            fallback_loss = F.kl_div(
                F.log_softmax(student_gates, dim=-1),
                F.softmax(teacher_gates, dim=-1),
                reduction='batchmean'
            )
            losses.update({
                'fallback_gate_kl': fallback_loss,
                'expert_total_loss': fallback_loss
            })
            total_loss = total_loss + fallback_loss
            logger.debug("Added fallback gate KL loss to ensure gradients exist")
        
        # Ensure the returned total loss is included in the losses dictionary
        if 'expert_total_loss' not in losses:
            losses['expert_total_loss'] = total_loss

        self.step_count += 1
        
        return losses
    

    def _compute_method_a_loss(self, 
                                   teacher_gates: torch.Tensor,
                                   student_gates: torch.Tensor,
                                   teacher_hidden_states: torch.Tensor,
                                   student_hidden_states: torch.Tensor,
                                   teacher_model: nn.Module,
                                   student_model: nn.Module,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Method A: Optimized version - batch processing version 2"""
        batch_size, seq_len, num_experts = teacher_gates.shape
        _bs, _sl_, num_student_experts = student_gates.shape
        K = self.config.mc_sampling_steps
        alpha = self.config.alpha
        device = teacher_gates.device
        
        # 1. Pre-compute sampling and weights for all positions (avoid loops)
        # Reshape all position probabilities to [batch_size * seq_len, num_experts]
        flat_teacher_probs = teacher_gates.view(-1, num_experts)
        valid_mask = attention_mask.view(-1) if attention_mask is not None else torch.ones(batch_size * seq_len, device=device)
        
        # 2. Batch sample experts for all positions (complete all sampling at once)
        total_samples = valid_mask.sum().int().item()
        if total_samples == 0:
            return self._get_zero_loss_dict()
        
        # Perform K sampling for valid positions
        valid_probs = flat_teacher_probs[valid_mask.bool()]  # [valid_positions, num_experts]
        
        all_expert_indices = []
        all_importance_weights = []
        current_probs = valid_probs.clone()
        
        for k in range(K):
            # Batch sampling
            expert_indices = torch.multinomial(current_probs, 1).squeeze(-1)  # [valid_positions]
            importance_weights = current_probs.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)
            
            all_expert_indices.append(expert_indices)
            all_importance_weights.append(importance_weights)
            
            # Batch update probabilities (expert suppression)
            if k < K - 1:
                current_probs.scatter_(1, expert_indices.unsqueeze(-1), 
                                    current_probs.gather(1, expert_indices.unsqueeze(-1)) * alpha)
                current_probs = current_probs / current_probs.sum(dim=-1, keepdim=True)
        
        # 3. Batch compute losses (avoid individual forward passes)
        total_feat_loss = 0.0
        total_weight = 0.0
        
        # Count expert usage statistics
        teacher_expert_counts = torch.zeros(num_experts, device=device)
        student_expert_counts = torch.zeros(num_experts, device=device)
        
        for k in range(K):
            expert_indices = all_expert_indices[k]
            importance_weights = all_importance_weights[k]
            
            # 4. Use pre-computed logits (instead of re-forward propagation)
            # Here we simplify by using gate weights to approximate expert influence
            expert_kl_losses = self._compute_batch_expert_kl_fast(
                teacher_gates, student_gates, expert_indices, importance_weights, valid_mask
            )
            
            feat_loss = (expert_kl_losses * importance_weights).sum()
            total_feat_loss += feat_loss
            total_weight += importance_weights.sum()
            
            # Update expert statistics
            for i, (expert_idx, weight) in enumerate(zip(expert_indices, importance_weights)):
                if expert_idx >= num_student_experts:
                    continue
                teacher_expert_counts[expert_idx] += weight
                # Use student gates corresponding to valid positions
                valid_positions = torch.where(valid_mask)[0]
                batch_idx = valid_positions[i] // seq_len
                seq_idx = valid_positions[i] % seq_len
                student_expert_counts[expert_idx] += student_gates[batch_idx, seq_idx, expert_idx]
        
        # Calculate average feature loss
        feat_loss = total_feat_loss / max(total_weight, 1e-8)
        
        # Calculate coverage-KL loss (keep unchanged)
        teacher_avg_activation = teacher_expert_counts / teacher_expert_counts.sum()
        student_avg_activation = student_expert_counts / student_expert_counts.sum()
        
        eps = 1e-8
        teacher_avg_activation = teacher_avg_activation + eps
        student_avg_activation = student_avg_activation + eps
        
        teacher_avg_activation = teacher_avg_activation / teacher_avg_activation.sum()
        student_avg_activation = student_avg_activation / student_avg_activation.sum()
        
        coverage_kl = F.kl_div(
            student_avg_activation.log(),
            teacher_avg_activation,
            reduction='batchmean'
        )
        
        return {
            'method_a_feat_loss': feat_loss,
            'method_a_coverage_kl': coverage_kl,
            'method_a_total_loss': feat_loss + self.config.lambda_coverage * coverage_kl
        }
    

    def _compute_batch_expert_kl_fast(self, 
                                    teacher_gates: torch.Tensor,
                                    student_gates: torch.Tensor,
                                    expert_indices: torch.Tensor,
                                    importance_weights: torch.Tensor,
                                    valid_mask: torch.Tensor) -> torch.Tensor:
        """Fast batch computation of expert KL loss"""
        # Use gate weights as logits proxy
        # This is an approximation but can greatly accelerate computation
        
        # Get valid positions
        valid_positions = torch.where(valid_mask)[0]
        batch_indices = valid_positions // teacher_gates.shape[1]
        seq_indices = valid_positions % teacher_gates.shape[1]
        
        # Extract gate probabilities for corresponding positions
        teacher_probs = teacher_gates[batch_indices, seq_indices]  # [valid_positions, num_experts]
        student_probs = student_gates[batch_indices, seq_indices]
        
        # Calculate weighted KL loss for each expert
        kl_losses = []
        for i, expert_idx in enumerate(expert_indices):
            # Create one-hot expert distribution
            expert_dist = torch.zeros_like(teacher_probs[i])
            expert_dist[expert_idx] = 1.0

            if student_probs[i].shape != expert_dist.shape:
                carry = int(expert_dist.shape[0] / student_probs[i].shape[0])
                acc_expert_dist = expert_dist[::carry]
                kl_loss = F.kl_div(
                    student_probs[i].log(),
                    acc_expert_dist,
                    reduction='sum'
                )
            else:
                # Calculate KL divergence with student distribution
                kl_loss = F.kl_div(
                    student_probs[i].log(),
                    expert_dist,
                    reduction='sum'
                )
            kl_losses.append(kl_loss)
        
        return torch.stack(kl_losses)
    

    def _apply_importance_weight_to_teacher(self, 
                                        teacher_model: nn.Module, 
                                        expert_idx: int, 
                                        importance_weight: float) -> Dict:
        """Apply importance_weight to teacher model's lora_route weights"""
        original_weights = {}
        
        def apply_weight_to_module(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Check if it's a HydraLoRA Linear layer with lora_route
                if hasattr(child_module, 'lora_route') and hasattr(child_module, 'lora_num'):
                    try:
                        # Save original weights
                        if hasattr(child_module.lora_route, 'weight'):
                            original_weights[full_name] = child_module.lora_route.weight.data.clone()
                            
                            # Apply importance_weight to the corresponding expert weights
                            # lora_route output shape is typically [input_dim, num_experts]
                            if expert_idx < child_module.lora_route.weight.shape[-1]:
                                # Create weight mask, apply importance_weight only to the specified expert
                                weight_mask = torch.ones_like(child_module.lora_route.weight)
                                weight_mask[:, expert_idx] *= importance_weight
                                
                                # Apply weight mask
                                child_module.lora_route.weight.data *= weight_mask
                                
                                logger.debug(f"Applied importance_weight {importance_weight:.4f} to expert {expert_idx} of {full_name}")
                            else:
                                logger.warning(f"Expert index {expert_idx} exceeds range {child_module.lora_route.weight.shape[-1]} for {full_name}")
                                
                    except Exception as e:
                        logger.warning(f"Error applying weight to {full_name}: {e}")
                else:
                    # Recursively process child modules
                    apply_weight_to_module(child_module, full_name)
        
        apply_weight_to_module(teacher_model)
        return original_weights

    def _restore_original_weights(self, 
                                teacher_model: nn.Module, 
                                original_weights: Dict):
        """Restore original lora_route weights of teacher model"""
        
        def restore_weights_in_module(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                # Check if weight restoration is needed
                if full_name in original_weights:
                    try:
                        if hasattr(child_module, 'lora_route') and hasattr(child_module.lora_route, 'weight'):
                            child_module.lora_route.weight.data = original_weights[full_name]
                            logger.debug(f"Restored original weights for {full_name}")
                    except Exception as e:
                        logger.warning(f"Error restoring weights for {full_name}: {e}")
                else:
                    # Recursively process child modules
                    restore_weights_in_module(child_module, full_name)
        
        restore_weights_in_module(teacher_model)

    def _forward_with_modified_weights(self, 
                                    model: nn.Module, 
                                    input_ids: torch.Tensor, 
                                    attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with modified weights"""
        try:
            # Ensure model is in evaluation mode
            model.eval()
            
            with torch.no_grad():
                # Prepare input
                if attention_mask is not None:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids)
                
                # Extract logits
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    # If no logits attribute, try to get through lm_head
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    elif isinstance(outputs, tuple):
                        hidden_states = outputs[0] if len(outputs) > 0 else None
                    else:
                        hidden_states = outputs
                    
                    if hidden_states is not None:
                        logits = self._get_logits_from_hidden(model, hidden_states)
                    else:
                        logger.error("Cannot extract logits or hidden_states from model output")
                        return torch.zeros(1, 1, 32000, device=input_ids.device)  # Return zero tensor as default
                
                return logits
                
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            # Return default-shaped zero tensor
            vocab_size = getattr(model.config, 'vocab_size', 32000)
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab_size, device=input_ids.device)

    def _get_logits_from_hidden(self, model: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calculate logits from hidden states (improved version)"""
        try:
            # Get lm_head layer
            lm_head = None
            
            # Find lm_head by priority
            if hasattr(model, 'lm_head'):
                lm_head = model.lm_head
            elif hasattr(model, 'head'):
                lm_head = model.head
            elif hasattr(model, 'classifier'):
                lm_head = model.classifier
            else:
                # Find layers containing lm_head
                for name, module in model.named_modules():
                    if 'lm_head' in name.lower() and isinstance(module, nn.Linear):
                        lm_head = module
                        break
                
                if lm_head is None:
                    # Find the last linear layer
                    linear_layers = []
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            linear_layers.append((name, module))
                    
                    if linear_layers:
                        lm_head = linear_layers[-1][1]  # Use the last linear layer
            
            if lm_head is None:
                logger.warning("lm_head layer not found, returning original hidden states")
                return hidden_states
            
            # Ensure data type and device consistency
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Move lm_head to correct device and data type
            if next(lm_head.parameters()).device != device:
                logger.debug(f"Moving lm_head to device: {device}")
                lm_head = lm_head.to(device)
            
            # Ensure weight data type matches
            weight = lm_head.weight
            if weight.dtype != dtype:
                logger.debug(f"Converting lm_head weight data type: {weight.dtype} -> {dtype}")
                weight = weight.to(dtype)
            
            # Check dimension matching
            input_dim = hidden_states.shape[-1]
            weight_dim = weight.shape[1] if hasattr(lm_head, 'weight') else weight.shape[0]
            
            if weight_dim != input_dim:
                logger.warning(f"lm_head dimension mismatch: input {input_dim}, weight {weight.shape}")
                # Try to adjust
                if weight.shape[0] == input_dim:
                    weight = weight.T
                else:
                    return hidden_states
            
            # Calculate logits
            logits = F.linear(hidden_states, weight, lm_head.bias)
            
            return logits.to(dtype=dtype)
            
        except Exception as e:
            logger.warning(f"Failed to calculate logits: {e}")
            return hidden_states
        
    def _compute_expert_kl_loss(self, 
                           student_logits: torch.Tensor, 
                           teacher_logits: torch.Tensor,
                           temperature: float = 1.0) -> torch.Tensor:
        """Calculate KL divergence loss for expert logits"""
        try:
            # Ensure dimension consistency
            if student_logits.shape != teacher_logits.shape:
                min_vocab_size = min(student_logits.shape[-1], teacher_logits.shape[-1])
                student_logits = student_logits[..., :min_vocab_size]
                teacher_logits = teacher_logits[..., :min_vocab_size]
            
            # Calculate temperature-scaled probability distributions
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
            
            # Calculate KL divergence
            kl_loss = F.kl_div(
                student_log_probs, 
                teacher_probs, 
                reduction='batchmean'
            )
            
            # Apply temperature squared scaling
            kl_loss = kl_loss * (temperature ** 2)
            
            return kl_loss
            
        except Exception as e:
            logger.warning(f"Failed to calculate expert KL loss: {e}")
            # Fall back to MSE loss if KL calculation fails
            return F.mse_loss(student_logits, teacher_logits)
    
    def _get_expert_hidden_state(self, 
                                model: nn.Module,
                                expert_idx: int,
                                hidden_input: torch.Tensor,
                                batch_idx: int,
                                seq_idx: int) -> torch.Tensor:
        """Get hidden state for specific expert (based on Hydralora implementation)"""
        try:
            # Find the last Hydralora module
            lora_layers = []
            
            # Traverse model to find Hydralora Linear layers
            def find_lora_layers(module, name=""):
                for child_name, child_module in module.named_children():
                    full_name = f"{name}.{child_name}" if name else child_name
                    
                    # Check if it's a Hydralora Linear layer
                    if hasattr(child_module, 'lora_route') and hasattr(child_module, 'lora_num'):
                        lora_layers.append((full_name, child_module))
                    else:
                        find_lora_layers(child_module, full_name)
            
            find_lora_layers(model)
            
            if not lora_layers:
                logger.warning("No Hydralora layers found")
                return hidden_input
            
            # Use the last Hydralora layer (typically in the last part of the model)
            last_lora_name, last_lora_layer = lora_layers[-1]
            
            # Simulate forward pass for a specific expert
            with torch.no_grad():
                # Basic linear transformation
                result = F.linear(
                    hidden_input, 
                    last_lora_layer.weight.T if last_lora_layer.fan_in_fan_out else last_lora_layer.weight, 
                    bias=last_lora_layer.bias
                )
                
                if last_lora_layer.r > 0 and not last_lora_layer.merged:
                    # Calculate the output of matrix A
                    lora_A_output = getattr(last_lora_layer, f"lora_A")(
                        last_lora_layer.lora_dropout(hidden_input)
                    )
                    
                    # Use the B matrix of the specific expert
                    if expert_idx < last_lora_layer.lora_num:
                        lora_B_output = getattr(last_lora_layer, f"lora_B{expert_idx}")(lora_A_output)
                        # Apply scaling
                        expert_output = lora_B_output * last_lora_layer.scaling
                        result = result + expert_output
                    else:
                        logger.warning(f"Expert index {expert_idx} exceeds range {last_lora_layer.lora_num}")
                
                return result
                
        except Exception as e:
            logger.warning(f"Failed to get hidden state for expert {expert_idx}: {e}")
            return hidden_input
    
    # def _compute_method_b_loss(self, 
    #                           teacher_gates: torch.Tensor,
    #                           student_gates: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     """Method B: Temperature-Entropy Router Distillation Version 1"""
    #     # Clip temperature parameter
    #     self.temperature.data = torch.clamp(
    #         self.temperature.data, 
    #         self.config.temp_clip_range[0], 
    #         self.config.temp_clip_range[1]
    #     )
        
    #     # Calculate student gate probabilities after temperature scaling
    #     student_gates_temp = F.softmax(student_gates / self.temperature, dim=-1)
        
    #     # Calculate temperature-aligned KL loss
    #     # KL(p_t || p_s^T)
    #     temp_kl_loss = F.kl_div(
    #         student_gates_temp.log(),
    #         teacher_gates,
    #         reduction='batchmean'
    #     )
        
    #     # Calculate entropy promotion regularization
    #     # H(p_s) = -sum(p_s * log(p_s))
    #     student_entropy = -torch.sum(student_gates * student_gates.log(), dim=-1)
    #     entropy_loss = -student_entropy.mean()  # negative sign means maximizing entropy
        
    #     # Total loss
    #     total_loss = temp_kl_loss + self.config.beta_entropy * entropy_loss
        
    #     return {
    #         'method_b_temp_kl': temp_kl_loss,
    #         'method_b_entropy': entropy_loss,
    #         'method_b_total_loss': total_loss / 10.0,
    #         'temperature': self.temperature.item()
    #     }

    def _compute_method_b_loss(self, 
                                   teacher_gates: torch.Tensor,
                                   student_gates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Method B: Optimized version"""
        
        # Reduce temperature update frequency (update every 100 steps)
        if self.step_count % 100 == 0:
            self.temperature.data = torch.clamp(
                self.temperature.data, 
                self.config.temp_clip_range[0], 
                self.config.temp_clip_range[1]
            )
        
        # Precalculate commonly used softmax (avoid repeated calculations)
        student_gates_temp = F.softmax(student_gates / self.temperature, dim=-1)
        teacher_gates_soft = F.softmax(teacher_gates, dim=-1)
        
        # Calculate KL loss (vectorized)
        if student_gates_temp.shape == teacher_gates_soft.shape:
            temp_kl_loss = F.kl_div(
                student_gates_temp.log(),
                teacher_gates_soft,
                reduction='batchmean'
            )
        else:
            carry = int(teacher_gates_soft.shape[2] / student_gates_temp.shape[2])
            acc_teacher_gates_soft = teacher_gates_soft[:, :, ::carry]
            temp_kl_loss = F.kl_div(
                student_gates_temp.log(),
                acc_teacher_gates_soft,
                reduction='batchmean'
            )
        
        # Simplified entropy calculation - Fix: use the correct variable name
        student_entropy = -(student_gates_temp * student_gates_temp.log()).sum(dim=-1).mean()
        entropy_loss = -student_entropy
        
        total_loss = temp_kl_loss + self.config.beta_entropy * entropy_loss
        
        return {
            'method_b_temp_kl': temp_kl_loss,
            'method_b_entropy': entropy_loss,
            'method_b_total_loss': total_loss,
            'temperature': self.temperature.item()
        }
    
    def _update_expert_stats(self, 
                            teacher_gates: torch.Tensor,
                            student_gates: torch.Tensor):
        """Update expert selection statistics"""
        if self.step_count % self.config.log_interval == 0:
            # Calculate the average activation probability of each expert
            teacher_avg_probs = teacher_gates.mean(dim=(0, 1))
            student_avg_probs = student_gates.mean(dim=(0, 1))
            
            # Calculate the selection distribution of Top-1 experts
            teacher_top1 = torch.argmax(teacher_gates, dim=-1)
            student_top1 = torch.argmax(student_gates, dim=-1)
            
            num_experts = teacher_gates.shape[-1]
            teacher_top1_dist = torch.bincount(teacher_top1.flatten(), minlength=num_experts).float()
            student_top1_dist = torch.bincount(student_top1.flatten(), minlength=num_experts).float()
            
            teacher_top1_dist = teacher_top1_dist / teacher_top1_dist.sum()
            student_top1_dist = student_top1_dist / student_top1_dist.sum()
            
            # Calculate the imbalance of the distribution (Gini coefficient)
            def gini_coefficient(probs):
                sorted_probs = torch.sort(probs)[0]
                n = len(sorted_probs)
                index = torch.arange(1, n + 1, device=probs.device)
                return (2 * torch.sum(index * sorted_probs)) / (n * torch.sum(sorted_probs)) - (n + 1) / n
            
            teacher_gini = gini_coefficient(teacher_avg_probs)
            student_gini = gini_coefficient(student_avg_probs)
            
            # Record statistics
            self.expert_selection_stats[self.step_count] = {
                'teacher_avg_probs': teacher_avg_probs.float().cpu().numpy(),
                'student_avg_probs': student_avg_probs.float().cpu().numpy(),
                'teacher_top1_dist': teacher_top1_dist.float().cpu().numpy(),
                'student_top1_dist': student_top1_dist.float().cpu().numpy(),
                'teacher_gini': teacher_gini.float().item(),
                'student_gini': student_gini.float().item(),
                'step': self.step_count
            }
            
            # Log information
            logger.info(f"Hydralora Expert Stats (Step {self.step_count}):")
            logger.info(f"  Teacher Gini: {teacher_gini:.4f}, Student Gini: {student_gini:.4f}")
            logger.info(f"  Top-3 Teacher Experts: {torch.topk(teacher_avg_probs, 3)[1].tolist()}")
            logger.info(f"  Top-3 Student Experts: {torch.topk(student_avg_probs, 3)[1].tolist()}")
    
    def get_expert_stats(self) -> Dict:
        """Get expert statistics"""
        return self.expert_selection_stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.expert_selection_stats = {}
        self.step_count = 0



class CombinedDistillationLoss(DistillationLoss):
    """Combined distillation loss"""
    def __init__(self, 
                 loss_type: str = "forward_kl",
                 temperature: float = 1.0,
                 alpha: float = 0.5,
                 lam: float = 0.9):
        super().__init__(temperature, alpha)
        self.loss_type = loss_type
        self.lam = lam
        
        # Initialize the corresponding loss function
        if loss_type == "forward_kl":
            self.loss_fn = ForwardKLLoss(temperature, alpha)
        elif loss_type == "reverse_kl":
            self.loss_fn = ReverseKLLoss(temperature, alpha)
        elif loss_type == "symmetric_kl":
            self.loss_fn = SymmetricKLLoss(temperature, alpha, lam)
        elif loss_type == "js_distance":
            self.loss_fn = JSDistanceLoss(temperature, alpha, lam)
        elif loss_type == "tv_distance":
            self.loss_fn = TVDistanceLoss(temperature, alpha)
        elif loss_type == "skewed_forward_kl":
            self.loss_fn = SkewedForwardKLLoss(temperature, alpha, lam)
        elif loss_type == "skewed_reverse_kl":
            self.loss_fn = SkewedReverseKLLoss(temperature, alpha, lam)
        elif loss_type == "skewed_symmetric_kl":
            self.loss_fn = SkewedSymmetricKLLoss(temperature, alpha, lam)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, student_outputs, teacher_outputs, labels=None):
        """
        Calculate distillation loss
        Args:
            student_outputs: Student model outputs (logits or dict with 'logits')
            teacher_outputs: Teacher model outputs (logits or dict with 'logits')
            labels: True labels
        """
        # Extract logits
        if isinstance(student_outputs, dict):
            student_logits = student_outputs['logits']
        else:
            student_logits = student_outputs
            
        if isinstance(teacher_outputs, dict):
            teacher_logits = teacher_outputs['logits']
        else:
            teacher_logits = teacher_outputs
        
        # Calculate loss
        loss = self.loss_fn(student_logits, teacher_logits, labels)
        
        return loss, {f"{self.loss_type}_loss": loss.item()}