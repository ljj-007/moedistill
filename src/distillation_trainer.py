import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Union, Any
import logging
from src.losses import CombinedDistillationLoss, ExpertDistillationConfig, ExpertDistillationLoss
from peft import PeftModel

logger = logging.getLogger(__name__)

class DistillationTrainer(Trainer):
    """Distillation trainer"""
    
    def __init__(self,
                 student_model,
                 teacher_model,
                 teacher_tokenizer=None,
                 loss_type: str = "forward_kl",
                 temperature: float = 1.0,
                 alpha: float = 0.5,
                 lam: float = 0.9,
                 distill_weight: float = 0.1,  # Distillation loss weight, default 0.1
                 sft_weight: float = 0.9,      # SFT loss weight, default 0.9
                 **kwargs):
        super().__init__(model=student_model, **kwargs)
        
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer or self.tokenizer
        self.distill_weight = 0.1
        self.sft_weight = 0.9
        
        # Set distillation loss
        self.distillation_loss = CombinedDistillationLoss(
            loss_type=loss_type,
            temperature=temperature,
            alpha=alpha,
            lam=lam
        )
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Move loss function to the same device
        if hasattr(self.model, 'device'):
            self.distillation_loss = self.distillation_loss.to(self.model.device)
        
        logger.info(f"Using distillation loss type: {loss_type}")
        logger.info(f"Temperature: {temperature}, Alpha: {alpha}, Lambda: {lam}")
        logger.info(f"Distillation weight: {distill_weight}, SFT weight: {sft_weight}")

    def _save(self, output_dir: str, state_dict=None):
        """Override save method to handle tensor sharing issues in PEFT models"""
        if getattr(self.model, "module", None) is not None:
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # If it's a PEFT model, use PEFT's save method
        if isinstance(model_to_save, PeftModel):
            logger.info("Saving PEFT model...")
            model_to_save.save_pretrained(output_dir)
            
            # Save tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
                
        else:
            # Normal model save
            logger.info("Saving normal model...")
            super()._save(output_dir, state_dict=state_dict)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Calculate combined loss: distillation loss + SFT loss
        """
        labels = inputs.get("labels")
        
        # Student model forward pass
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
            return_dict=True
        )
        
        # Calculate SFT loss (standard cross-entropy loss)
        sft_loss = student_outputs.loss if student_outputs.loss is not None else 0.0


        # Teacher model forward pass (for distillation)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True
            )
        
        # Calculate distillation loss (excluding hard label loss, as we calculate SFT loss separately)
        distill_loss, distill_loss_dict = self.distillation_loss(
            student_outputs, 
            teacher_outputs, 
            labels=None  # Don't pass labels, only calculate distillation part
        )
        
        # Combined loss
        total_loss = self.sft_weight * sft_loss + self.distill_weight * distill_loss
        
        # Combined loss dictionary
        combined_loss_dict = {
            **distill_loss_dict,
            'sft_loss': sft_loss.item() if isinstance(sft_loss, torch.Tensor) else sft_loss,
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'sft_weight': self.sft_weight,
            'distill_weight': self.distill_weight
        }
        
        # Log losses
        self.log(combined_loss_dict)
        
        return (total_loss, student_outputs) if return_outputs else total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Prediction step
        """
        model.eval()
        
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            
        if prediction_loss_only:
            return (loss, None, None)
        
        # Get student model output for evaluation
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                return_dict=True
            )
        
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        return (loss, logits, labels)
    

class ExpertDistillationTrainer(Trainer):
    """
    Expert distillation trainer - combining Hydralora's expert distillation
    """
    
    def __init__(self,
                 student_model,
                 teacher_model,
                 teacher_tokenizer=None,
                 expert_distill_config: Optional[Dict] = None,
                 logits_loss_type: str = "forward_kl",
                 logits_temperature: float = 1.0,
                 logits_alpha: float = 0.7,
                 logits_lam: float = 0.9,
                 expert_distill_weight: float = 0.1,  # Expert distillation weight
                 logits_distill_weight: float = 0.1,  # Logits distillation weight
                 sft_weight: float = 0.8,             # SFT loss weight
                 **kwargs):
        super().__init__(model=student_model, **kwargs)
        
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer or self.tokenizer
        self.expert_distill_weight = 0.1
        self.logits_distill_weight = 0.1
        self.sft_weight = 0.8
        
        # Set logits distillation loss
        self.logits_distillation_loss = CombinedDistillationLoss(
            loss_type=logits_loss_type,
            temperature=logits_temperature,
            alpha=logits_alpha,
            lam=logits_lam
        )
        
        # Set expert distillation loss
        if expert_distill_config is None:
            expert_distill_config = {}
        
        expert_config = ExpertDistillationConfig(**expert_distill_config)
        self.expert_distillation_loss = ExpertDistillationLoss(expert_config)
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Move loss functions to the same device
        if hasattr(self.model, 'device'):
            self.logits_distillation_loss = self.logits_distillation_loss.to(self.model.device)
            self.expert_distillation_loss = self.expert_distillation_loss.to(self.model.device)
        
        logger.info(f"Using logits distillation loss type: {logits_loss_type}")
        logger.info(f"Expert distillation weight: {expert_distill_weight}")
        logger.info(f"Logits distillation weight: {logits_distill_weight}")
        logger.info(f"SFT weight: {sft_weight}")
        logger.info(f"Expert distillation config: {expert_config}")

    def _extract_gates_and_hidden_states(self, model, inputs):
        """Extract routing weights and hidden states"""
        route_weights = []
        
        def create_hook(route_weights_list):
            def hook_fn(module, input, output):
                if hasattr(module, 'lora_route') and hasattr(module, 'lora_num'):
                    # Get routing weights
                    route_logits = module.lora_route(input[0])
                    route_weight = F.softmax(route_logits, dim=-1)
                    route_weights_list.append(route_weight)
                return output
            return hook_fn
        
        # Register hook functions
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'lora_route') and hasattr(module, 'lora_num'):
                hook = module.register_forward_hook(create_hook(route_weights))
                hooks.append(hook)
        
        try:
            # Ensure inputs are on the correct device
            device = next(model.parameters()).device
            inputs_on_device = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs_on_device[key] = value.to(device)
                else:
                    inputs_on_device[key] = value
            
            # Forward pass
            with torch.no_grad():  # Add no_grad to avoid gradient computation
                outputs = model(
                    input_ids=inputs_on_device["input_ids"],
                    attention_mask=inputs_on_device.get("attention_mask"),
                    labels=inputs_on_device.get("labels"),  # Add labels to compute loss
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get gate information (use the last routing weight)
            if route_weights:
                gates = route_weights[-1]  # Use the routing weight of the last layer
            else:
                # If no routing weights are obtained, use default uniform distribution
                batch_size, seq_len = inputs_on_device["input_ids"].shape
                num_experts = getattr(model.config, 'lora_nums', 2)
                
                gates = torch.ones(batch_size, seq_len, num_experts, device=device)
                gates = F.softmax(gates, dim=-1)
                
                logger.warning("Could not obtain routing weights, using uniform distribution")
            
            # Use the hidden states of the last layer
            hidden_states = outputs.hidden_states[-1]
            
            return outputs, gates, hidden_states
            
        finally:
            # Remove hook functions
            for hook in hooks:
                hook.remove()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Calculate combined distillation loss (SFT + logits distillation + expert distillation)
        """
        labels = inputs.get("labels")
        
        # Ensure inputs are on the correct device
        model_device = next(model.parameters()).device
        teacher_device = next(self.teacher_model.parameters()).device
        
        # Move input to model device
        inputs_for_student = {}
        inputs_for_teacher = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs_for_student[key] = value.to(model_device)
                inputs_for_teacher[key] = value.to(teacher_device)
            else:
                inputs_for_student[key] = value
                inputs_for_teacher[key] = value
        
        # Student model forward pass and extract information
        student_outputs, student_gates, student_hidden = self._extract_gates_and_hidden_states(model, inputs_for_student)
        
        # Calculate SFT loss separately (requires gradients)
        student_sft_outputs = model(
            input_ids=inputs_for_student["input_ids"],
            attention_mask=inputs_for_student.get("attention_mask"),
            labels=labels.to(model_device) if labels is not None else None,
            return_dict=True
        )
        sft_loss = student_sft_outputs.loss if student_sft_outputs.loss is not None else 0.0
        
        # Teacher model forward pass and extract information
        with torch.no_grad():
            teacher_outputs, teacher_gates, teacher_hidden = self._extract_gates_and_hidden_states(self.teacher_model, inputs_for_teacher)
        
        # Ensure all tensors are on the same device (use student model's device)
        if teacher_gates.device != student_gates.device:
            teacher_gates = teacher_gates.to(student_gates.device)
        if teacher_hidden.device != student_hidden.device:
            teacher_hidden = teacher_hidden.to(student_hidden.device)
        if teacher_outputs.logits.device != student_outputs.logits.device:
            teacher_outputs.logits = teacher_outputs.logits.to(student_outputs.logits.device)
        
        # Calculate logits distillation loss (excluding hard label loss)
        logits_loss, logits_loss_dict = self.logits_distillation_loss(
            student_outputs, 
            teacher_outputs, 
            labels=None  # Don't pass labels, only calculate distillation part
        )
        
        # Calculate expert distillation loss
        expert_loss_dict = self.expert_distillation_loss(
            teacher_gates=teacher_gates,
            student_gates=student_gates,
            teacher_hidden_states=teacher_hidden,
            student_hidden_states=student_hidden,
            teacher_model=self.teacher_model,
            student_model=model,
            input_ids=inputs_for_student["input_ids"],
            attention_mask=inputs_for_student.get("attention_mask")
        )
        
        # Combine expert losses
        expert_loss = 0.0
        for loss_name, loss_value in expert_loss_dict.items():
            if 'total_loss' in loss_name:
                expert_loss += loss_value
        
        # Total loss: SFT + logits distillation + expert distillation
        total_loss = (
            self.sft_weight * sft_loss + 
            self.logits_distill_weight * logits_loss + 
            self.expert_distill_weight * expert_loss
        )
        
        # Merge loss dictionary
        combined_loss_dict = {
            **logits_loss_dict,
            **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in expert_loss_dict.items()},
            'sft_loss': sft_loss.item() if isinstance(sft_loss, torch.Tensor) else sft_loss,
            'logits_distill_loss': logits_loss.item() if isinstance(logits_loss, torch.Tensor) else logits_loss,
            'expert_loss': expert_loss.item() if isinstance(expert_loss, torch.Tensor) else expert_loss,
            'total_loss': total_loss.item(),
            'sft_weight': self.sft_weight,
            'logits_distill_weight': self.logits_distill_weight,
            'expert_distill_weight': self.expert_distill_weight
        }
        
        # Log losses
        self.log(combined_loss_dict)
        
        return (total_loss, student_outputs) if return_outputs else total_loss

    def get_expert_stats(self) -> Dict:
        """Get expert statistics"""
        return self.expert_distillation_loss.get_expert_stats()

    def reset_expert_stats(self):
        """Reset expert statistics"""
        self.expert_distillation_loss.reset_stats()

    def _save(self, output_dir: str, state_dict=None):
        """Override save method to handle tensor sharing issues in PEFT models"""
        if getattr(self.model, "module", None) is not None:
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # If it's a PEFT model, use PEFT's save method
        if isinstance(model_to_save, PeftModel):
            logger.info("Saving PEFT model...")
            model_to_save.save_pretrained(output_dir)
            
            # Save tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
                
            # Save expert statistics
            import json
            import os
            expert_stats = self.get_expert_stats()
            if expert_stats:
                stats_path = os.path.join(output_dir, "expert_stats.json")
                with open(stats_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_stats = {}
                    for step, stats in expert_stats.items():
                        serializable_stats[str(step)] = {
                            k: v.tolist() if hasattr(v, 'tolist') else v
                            for k, v in stats.items()
                        }
                    json.dump(serializable_stats, f, indent=2)
                logger.info(f"Expert statistics saved to: {stats_path}")
                
        else:
            # Normal model save
            logger.info("Saving normal model...")
            super()._save(output_dir, state_dict=state_dict)