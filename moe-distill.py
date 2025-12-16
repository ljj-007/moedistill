import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExpertDistillationConfig:
    """Expert Distillation Configuration"""
    # Method A: Single-layer Monte Carlo Exploration
    mc_sampling_steps: int = 3  # Number of Monte Carlo sampling steps K
    alpha: float = 0.5  # Expert suppression coefficient
    lambda_coverage: float = 0.5  # Coverage-KL loss weight
    
    # Method B: Temperature-Entropy Router Distillation
    initial_temperature: float = 1.0  # Initial temperature
    beta_entropy: float = 0.1  # Entropy regularization weight
    temp_clip_range: Tuple[float, float] = (0.5, 1.5)  # Temperature clipping range
    
    # General Configuration
    enable_method_a: bool = True  # Enable Method A
    enable_method_b: bool = True  # Enable Method B
    
    # Debug Configuration
    log_expert_stats: bool = True  # Record expert statistics
    log_interval: int = 100  # Logging interval


class ExpertDistillationLoss(nn.Module):
    """Expert Distillation Loss Module"""
    
    def __init__(self, config: ExpertDistillationConfig):
        super().__init__()
        self.config = config
        
        # Method B: Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(config.initial_temperature))
        
        # Statistics
        self.expert_selection_stats = {}
        self.step_count = 0
        
    def forward(self,
                teacher_gates: torch.Tensor,
                student_gates: torch.Tensor,
                teacher_hidden_states: torch.Tensor,
                student_hidden_states: torch.Tensor,
                teacher_model: nn.Module,
                student_model: nn.Module,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Calculate expert distillation loss
        
        Args:
            teacher_gates: Teacher model gate probabilities [batch_size, seq_len, num_experts]
            student_gates: Student model gate probabilities [batch_size, seq_len, num_experts] 
            teacher_hidden_states: Teacher model hidden states [batch_size, seq_len, hidden_size]
            student_hidden_states: Student model hidden states [batch_size, seq_len, hidden_size]
            teacher_model: Teacher model
            student_model: Student model
            input_ids: Input token ids
            attention_mask: Attention mask
            
        Returns:
            Loss dictionary
        """
        losses = {}
        
        batch_size, seq_len, num_experts = teacher_gates.shape
        
        # Method A: Single-layer Monte Carlo Exploration
        if self.config.enable_method_a:
            method_a_losses = self._compute_method_a_loss(
                teacher_gates, student_gates,
                teacher_hidden_states, student_hidden_states,
                teacher_model, student_model,
                input_ids, attention_mask
            )
            losses.update(method_a_losses)
        
        # Method B: Temperature-Entropy Router Distillation
        if self.config.enable_method_b:
            method_b_losses = self._compute_method_b_loss(
                teacher_gates, student_gates
            )
            losses.update(method_b_losses)
        
        # Update statistics
        if self.config.log_expert_stats:
            self._update_expert_stats(teacher_gates, student_gates)
        
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
        """
        Method A: Single-layer Monte Carlo Exploration
        """
        batch_size, seq_len, num_experts = teacher_gates.shape
        K = self.config.mc_sampling_steps
        alpha = self.config.alpha
        
        # Monte Carlo sampling is performed for each position
        total_feat_loss = 0.0
        total_coverage_loss = 0.0
        total_weight = 0.0
        
        # Statistics for each expert's activation count (for calculating Coverage-KL)
        teacher_expert_counts = torch.zeros(num_experts, device=teacher_gates.device)
        student_expert_counts = torch.zeros(num_experts, device=student_gates.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Skip padding positions
                if attention_mask is not None and attention_mask[b, s] == 0:
                    continue
                
                # Current position's teacher gate probabilities
                teacher_probs = teacher_gates[b, s]  # [num_experts]
                current_probs = teacher_probs.clone()
                
                position_feat_loss = 0.0
                position_weight = 0.0
                
                # Perform K Monte Carlo sampling steps
                for k in range(K):
                    # 1. Sample expert according to current probabilities
                    expert_idx = torch.multinomial(current_probs, 1).item()
                    
                    # 2. Record importance weight (probability before sampling)
                    importance_weight = current_probs[expert_idx].item()
                    
                    # 3. Get hidden state for this expert
                    teacher_expert_hidden = self._get_expert_hidden_state(
                        teacher_model, expert_idx, 
                        teacher_hidden_states[b:b+1, s:s+1], b, s
                    )
                    student_expert_hidden = self._get_expert_hidden_state(
                        student_model, expert_idx,
                        student_hidden_states[b:b+1, s:s+1], b, s
                    )
                    
                    # 4. Calculate feature KD loss
                    feat_loss = F.mse_loss(student_expert_hidden, teacher_expert_hidden)
                    position_feat_loss += importance_weight * feat_loss
                    position_weight += importance_weight
                    
                    # 5. Update expert statistics
                    teacher_expert_counts[expert_idx] += importance_weight
                    student_expert_counts[expert_idx] += student_gates[b, s, expert_idx].item()
                    
                    # 6. Suppress selected expert and renormalize
                    if k < K - 1:  # No need to update on the last step
                        current_probs[expert_idx] *= alpha
                        # Renormalize
                        current_probs = current_probs / current_probs.sum()
                
                # Accumulate position loss
                if position_weight > 0:
                    total_feat_loss += position_feat_loss
                    total_weight += position_weight
        
        # Calculate average feature loss
        feat_loss = total_feat_loss / max(total_weight, 1e-8)
        
        # Calculate Coverage-KL loss
        teacher_avg_activation = teacher_expert_counts / teacher_expert_counts.sum()
        student_avg_activation = student_expert_counts / student_expert_counts.sum()
        
        # Add small epsilon to avoid numerical instability
        eps = 1e-8
        teacher_avg_activation = teacher_avg_activation + eps
        student_avg_activation = student_avg_activation + eps
        
        # Renormalize
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
    
    def _get_expert_hidden_state(self, 
                                model: nn.Module,
                                expert_idx: int,
                                hidden_input: torch.Tensor,
                                batch_idx: int,
                                seq_idx: int) -> torch.Tensor:
        """
        Get hidden state for a specific expert (based on Hydralora implementation)
        
        Args:
            model: Model
            expert_idx: Expert index
            hidden_input: Current position hidden state input [1, 1, hidden_size]
            batch_idx: Batch index
            seq_idx: Sequence index
        """
        try:
            # Find the last layer of Hydralora module
            lora_layers = []
            
            # Iterate through the model to find Hydralora's Linear layers
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
                logger.warning("No Hydralora layer found")
                return hidden_input
            
            # Use the last Hydralora layer (usually in the last part of the model)
            last_lora_name, last_lora_layer = lora_layers[-1]
            
            # Simulate forward pass for a specific expert
            with torch.no_grad():
                # Base linear transformation
                result = F.linear(
                    hidden_input, 
                    last_lora_layer.weight.T if last_lora_layer.fan_in_fan_out else last_lora_layer.weight, 
                    bias=last_lora_layer.bias
                )
                
                if last_lora_layer.r > 0 and not last_lora_layer.merged:
                    # Calculate output of A matrix
                    lora_A_output = getattr(last_lora_layer, f"lora_A")(
                        last_lora_layer.lora_dropout(hidden_input)
                    )
                    
                    # Use the specific expert's B matrix
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
    
    def _compute_method_b_loss(self, 
                              teacher_gates: torch.Tensor,
                              student_gates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Method B: Temperature-Entropy Router Distillation
        """
        # Clip temperature parameter
        self.temperature.data = torch.clamp(
            self.temperature.data, 
            self.config.temp_clip_range[0], 
            self.config.temp_clip_range[1]
        )
        
        # Calculate temperature-scaled student gate probabilities
        student_gates_temp = F.softmax(student_gates / self.temperature, dim=-1)
        
        # Calculate temperature-aligned KL loss
        # KL(p_t || p_s^T)
        temp_kl_loss = F.kl_div(
            student_gates_temp.log(),
            teacher_gates,
            reduction='batchmean'
        )
        
        # Calculate entropy enhancement regularization
        # H(p_s) = -sum(p_s * log(p_s))
        student_entropy = -torch.sum(student_gates * student_gates.log(), dim=-1)
        entropy_loss = -student_entropy.mean()  # Negative sign indicates maximizing entropy
        
        # Total loss
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
            # Calculate average activation probability for each expert
            teacher_avg_probs = teacher_gates.mean(dim=(0, 1))
            student_avg_probs = student_gates.mean(dim=(0, 1))
            
            # Calculate Top-1 expert selection distribution
            teacher_top1 = torch.argmax(teacher_gates, dim=-1)
            student_top1 = torch.argmax(student_gates, dim=-1)
            
            num_experts = teacher_gates.shape[-1]
            teacher_top1_dist = torch.bincount(teacher_top1.flatten(), minlength=num_experts).float()
            student_top1_dist = torch.bincount(student_top1.flatten(), minlength=num_experts).float()
            
            teacher_top1_dist = teacher_top1_dist / teacher_top1_dist.sum()
            student_top1_dist = student_top1_dist / student_top1_dist.sum()
            
            # Calculate distribution imbalance (Gini coefficient)
            def gini_coefficient(probs):
                sorted_probs = torch.sort(probs)[0]
                n = len(sorted_probs)
                index = torch.arange(1, n + 1, device=probs.device)
                return (2 * torch.sum(index * sorted_probs)) / (n * torch.sum(sorted_probs)) - (n + 1) / n
            
            teacher_gini = gini_coefficient(teacher_avg_probs)
            student_gini = gini_coefficient(student_avg_probs)
            
            # Record statistics
            self.expert_selection_stats[self.step_count] = {
                'teacher_avg_probs': teacher_avg_probs.cpu().numpy(),
                'student_avg_probs': student_avg_probs.cpu().numpy(),
                'teacher_top1_dist': teacher_top1_dist.cpu().numpy(),
                'student_top1_dist': student_top1_dist.cpu().numpy(),
                'teacher_gini': teacher_gini.item(),
                'student_gini': student_gini.item(),
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


class HydraloraExpertDistillationTrainer:
    """
    Expert Distillation Trainer combined with Hydralora
    """
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 config: ExpertDistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.distillation_loss = ExpertDistillationLoss(config)
        
        # Ensure teacher model is in evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_distillation_loss(self, 
                                 input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor = None,
                                 labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Calculate distillation loss
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Labels (optional)
            
        Returns:
            Loss dictionary
        """
        # Hook function to store routing information
        teacher_route_weights = []
        student_route_weights = []
        
        def create_hook(route_weights_list):
            def hook_fn(module, input, output):
                if hasattr(module, 'lora_route') and hasattr(module, 'lora_num'):
                    # Get routing weights
                    route_logits = module.lora_route(input[0])
                    route_weight = F.softmax(route_logits, dim=-1)
                    route_weights_list.append(route_weight)
                return output
            return hook_fn
        
        # Register hooks
        teacher_hooks = []
        student_hooks = []
        
        # Register hooks for teacher model
        for name, module in self.teacher_model.named_modules():
            if hasattr(module, 'lora_route') and hasattr(module, 'lora_num'):
                hook = module.register_forward_hook(create_hook(teacher_route_weights))
                teacher_hooks.append(hook)
        
        # Register hooks for student model
        for name, module in self.student_model.named_modules():
            if hasattr(module, 'lora_route') and hasattr(module, 'lora_num'):
                hook = module.register_forward_hook(create_hook(student_route_weights))
                student_hooks.append(hook)
        
        try:
            # Get teacher model output
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True
                )
            
            # Get student model output
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True
            )
            
            # Get gate information (use the last routing weights)
            if teacher_route_weights and student_route_weights:
                teacher_gates = teacher_route_weights[-1]  # Use routing weights from the last layer
                student_gates = student_route_weights[-1]
            else:
                # If no routing weights were obtained, use a default uniform distribution
                batch_size, seq_len = input_ids.shape
                num_experts = getattr(self.teacher_model.config, 'lora_nums', 2)
                device = input_ids.device
                
                teacher_gates = torch.ones(batch_size, seq_len, num_experts, device=device)
                student_gates = torch.ones(batch_size, seq_len, num_experts, device=device)
                teacher_gates = F.softmax(teacher_gates, dim=-1)
                student_gates = F.softmax(student_gates, dim=-1)
                
                logger.warning("Failed to get routing weights, using uniform distribution")
            
            # Use hidden states from the last layer
            teacher_hidden = teacher_outputs.hidden_states[-1]
            student_hidden = student_outputs.hidden_states[-1]
            
            # Calculate expert distillation loss
            distill_losses = self.distillation_loss(
                teacher_gates=teacher_gates,
                student_gates=student_gates,
                teacher_hidden_states=teacher_hidden,
                student_hidden_states=student_hidden,
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            return distill_losses
            
        finally:
            # Remove hooks
            for hook in teacher_hooks:
                hook.remove()
            for hook in student_hooks:
                hook.remove()
    
    def get_expert_stats(self) -> Dict:
        """Get expert statistics"""
        return self.distillation_loss.get_expert_stats()
    
    def reset_stats(self):
        """Reset statistics"""
        self.distillation_loss.reset_stats()


# Case
def create_expert_distillation_trainer(teacher_model, student_model, config_dict=None):
    """
    Convenient function to create expert distillation trainer
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        config_dict: Configuration dictionary
        
    Returns:
        HydraloraExpertDistillationTrainer instance
    """
    # Default configuration
    default_config = {
        'mc_sampling_steps': 3,
        'alpha': 0.5,
        'lambda_coverage': 0.5,
        'initial_temperature': 1.0,
        'beta_entropy': 0.1,
        'temp_clip_range': (0.5, 1.5),
        'enable_method_a': True,
        'enable_method_b': True,
        'log_expert_stats': True,
        'log_interval': 100
    }
    
    if config_dict:
        default_config.update(config_dict)
    
    config = ExpertDistillationConfig(**default_config)
    
    return HydraloraExpertDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config
    )


# Examples used in training cycles
def training_step_with_expert_distillation(batch, 
                                          expert_trainer: HydraloraExpertDistillationTrainer,
                                          base_loss_fn,
                                          alpha_distill: float = 0.5):
    """
    Example training step combined with expert distillation
    
    Args:
        batch: Training batch
        expert_trainer: Expert distillation trainer
        base_loss_fn: Base loss function
        alpha_distill: Distillation loss weight
        
    Returns:
        Total loss and loss details
    """
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask', None)
    labels = batch.get('labels', None)
    
    # Calculate base loss (e.g. language model loss)
    student_outputs = expert_trainer.student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True
    )
    base_loss = student_outputs.loss
    
    # Calculate expert distillation loss
    distill_losses = expert_trainer.compute_distillation_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    # Combine losses
    total_distill_loss = 0.0
    loss_details = {'base_loss': base_loss.item()}
    
    for loss_name, loss_value in distill_losses.items():
        if 'total_loss' in loss_name:
            total_distill_loss += loss_value
        loss_details[loss_name] = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
    
    # Total loss
    total_loss = (1 - alpha_distill) * base_loss + alpha_distill * total_distill_loss
    loss_details['total_loss'] = total_loss.item()
    loss_details['total_distill_loss'] = total_distill_loss.item() if isinstance(total_distill_loss, torch.Tensor) else total_distill_loss
    
    return total_loss, loss_details


if __name__ == "__main__":
    # Test code
    print("Hydralora Expert Distillation module loaded successfully")
    print("Main components:")
    print("1. ExpertDistillationConfig - Configuration class")
    print("2. ExpertDistillationLoss - Loss calculation module")
    print("3. HydraloraExpertDistillationTrainer - Trainer")
    print("4. create_expert_distillation_trainer - Convenient creation function")
    print("5. training_step_with_expert_distillation - Training step example")