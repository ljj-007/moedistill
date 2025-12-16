#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script: Test the performance of trained models on multiple datasets using Rouge-L score
Supported datasets: sinst, uinst, vicuna, self-inst, dolly
"""

import os
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import time
from rouge_score import rouge_scorer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path, lora_path=None, model_type="auto", device="cuda"):
        """Initialize model evaluator"""
        self.device = device
        self.model_type = model_type
        logger.info(f"Loading model: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True if model_type == "qwen" else False
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True if model_type == "qwen" else False,
            device_map="auto" if torch.cuda.device_count() > 1 else None
        )
        
        # If only one GPU or a specific device is specified, move to that device
        if torch.cuda.device_count() == 1 or device != "auto":
            self.model = self.model.to(device)
        
        # Load LoRA weights (if provided)
        if lora_path and os.path.exists(lora_path):
            logger.info(f"Loading LoRA weights: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            
            # Ensure all PEFT layers have consistent data types
            logger.info("Unifying PEFT layer data types...")
            target_dtype = torch.bfloat16
            peft_layers_fixed = 0
            
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype != target_dtype:
                        module.weight.data = module.weight.data.to(target_dtype)
                        peft_layers_fixed += 1
                        
                # Check bias (if exists)
                if hasattr(module, 'bias') and module.bias is not None:
                    if module.bias.dtype != target_dtype:
                        module.bias.data = module.bias.data.to(target_dtype)
                        peft_layers_fixed += 1
            
            logger.info(f"Fixed data type mismatches in {peft_layers_fixed} layers")
        
        self.model.eval()
        
        # Initialize Rouge scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, do_sample=True):
        """Generate model response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Ensure input data type is correct
            if hasattr(self.model, 'dtype'):
                target_dtype = self.model.dtype
            else:
                target_dtype = torch.bfloat16
                
            # Convert inputs to the correct data type
            for key in inputs:
                if inputs[key].dtype in [torch.float32, torch.float64]:
                    inputs[key] = inputs[key].to(target_dtype)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Only return newly generated tokens
            generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Return empty response to avoid interrupting evaluation
            return ""

class DatasetLoader:
    """Dataset loader"""
    
    @staticmethod
    def load_sinst_dataset(data_path):
        """Load sinst dataset"""
        datasets = []
        for subset in ["0_2", "3_5", "6_10", "11_"]:
            subset_path = os.path.join(data_path, subset, "valid.jsonl")
            if os.path.exists(subset_path):
                with open(subset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        datasets.append({
                            'subset': subset,
                            'prompt': data['prompt'],
                            'reference': str(data['output'][0]) if isinstance(data['output'], list) else str(data['output']),
                            'instruction': data.get('instruction', ''),
                            'input': data.get('input', ''),
                            'topic': data.get('topic', '')
                        })
        return datasets
    
    @staticmethod
    def load_uinst_dataset(data_path):
        """Load uinst dataset"""
        datasets = []
        for subset in ["0_2", "3_5", "6_10", "11_"]:
            subset_path = os.path.join(data_path, subset, "valid.jsonl")
            if os.path.exists(subset_path):
                with open(subset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        datasets.append({
                            'subset': subset,
                            'prompt': data['prompt'],
                            'reference': str(data['output']),
                            'instruction': data.get('instruction', ''),
                            'input': data.get('input', '')
                        })
        return datasets
    
    @staticmethod
    def load_vicuna_dataset(data_path):
        """Load vicuna dataset"""
        datasets = []
        file_path = os.path.join(data_path, "valid.jsonl")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                datasets.append({
                    'subset': 'all',
                    'prompt': data['prompt'],
                    'reference': str(data['output']),
                    'instruction': data.get('instruction', ''),
                    'category': data.get('category', '')
                })
        return datasets
    
    @staticmethod
    def load_self_inst_dataset(data_path):
        """Load self-inst dataset"""
        datasets = []
        file_path = os.path.join(data_path, "valid.jsonl")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                datasets.append({
                    'subset': 'all',
                    'prompt': data['prompt'],
                    'reference': str(data['output']),
                    'instruction': data.get('instruction', ''),
                    'input': data.get('input', ''),
                    'topic': data.get('topic', '')
                })
        return datasets
    
    @staticmethod
    def load_dolly_dataset(data_path):
        """Load dolly dataset"""
        datasets = []
        file_path = os.path.join(data_path, "valid.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            for data in data_list:
                datasets.append({
                    'subset': 'all',
                    'prompt': data['prompt'],
                    'reference': str(data['output']),
                    'instruction': data.get('instruction', ''),
                    'input': data.get('input', ''),
                    'topic': data.get('topic', '')
                })
        return datasets

def evaluate_dataset(evaluator, dataset_name, dataset, output_dir, max_samples=None):
    """Evaluate a single dataset"""
    logger.info(f"Starting evaluation of dataset: {dataset_name}, sample count: {len(dataset)}")
    
    if max_samples:
        dataset = dataset[:max_samples]
        logger.info(f"Limiting evaluation samples to: {max_samples}")
    
    results = []
    rouge_scores = []
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
        try:
            # Generate response
            generated = evaluator.generate_response(sample['prompt'])
            
            # Skip this sample if generation failed
            if not generated:
                logger.warning(f"Sample {i} generated empty response, skipping")
                continue
            
            # Calculate Rouge-L score
            rouge_score = evaluator.rouge_scorer.score(sample['reference'], generated)
            rouge_l_score = rouge_score['rougeL'].fmeasure
            rouge_scores.append(rouge_l_score)
            
            # Record results
            result = {
                'dataset': dataset_name,
                'subset': sample.get('subset', 'all'),
                'sample_id': i,
                'prompt': sample['prompt'],
                'reference': sample['reference'],
                'generated': generated,
                'rouge_l_score': rouge_l_score,
                'instruction': sample.get('instruction', ''),
                'input': sample.get('input', ''),
                'topic': sample.get('topic', ''),
                'category': sample.get('category', '')
            }
            results.append(result)
            
            # Save every 50 samples (reduce save frequency)
            if (i + 1) % 50 == 0:
                logger.info(f"{dataset_name}: Completed {i+1}/{len(dataset)} samples, current average Rouge-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            logger.debug(f"Error sample content: {sample.get('prompt', '')[:100]}...")
            continue
    
    # Calculate average Rouge-L score
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    logger.info(f"{dataset_name} average Rouge-L score: {avg_rouge_l:.4f}")
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{dataset_name}_detailed_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save CSV format for easy analysis
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, f"{dataset_name}_results.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    return avg_rouge_l, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate model Rouge-L scores on multiple datasets")
    parser.add_argument("--model_path", required=True, help="Base model path")
    parser.add_argument("--lora_path", help="LoRA weights path (optional)")
    parser.add_argument("--model_type", default="auto", choices=["auto", "llama", "qwen"], help="Model type")
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--output_dir", default="./eval_results", help="Results output directory")
    parser.add_argument("--datasets", nargs='+', default=["sinst", "uinst", "vicuna", "self-inst", "dolly"], 
                       help="Datasets to evaluate")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples per dataset (for quick testing)")
    parser.add_argument("--device", default="cuda", help="Computing device")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        lora_path=args.lora_path,
        model_type=args.model_type,
        device=args.device
    )
    
    # Dataset loader mapping
    dataset_loaders = {
        "sinst": DatasetLoader.load_sinst_dataset,
        "uinst": DatasetLoader.load_uinst_dataset,
        "vicuna": DatasetLoader.load_vicuna_dataset,
        "self-inst": DatasetLoader.load_self_inst_dataset,
        "dolly": DatasetLoader.load_dolly_dataset
    }
    
    # Summary of evaluation results
    summary_results = {}
    all_detailed_results = []
    
    # Evaluate each dataset
    for dataset_name in args.datasets:
        if dataset_name not in dataset_loaders:
            logger.warning(f"Unknown dataset: {dataset_name}, skipping")
            continue
            
        dataset_path = os.path.join(args.data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path does not exist: {dataset_path}, skipping")
            continue
        
        try:
            # Load dataset
            dataset = dataset_loaders[dataset_name](dataset_path)
            if not dataset:
                logger.warning(f"Dataset {dataset_name} is empty, skipping")
                continue
            
            # Evaluate dataset
            avg_rouge_l, detailed_results = evaluate_dataset(
                evaluator, dataset_name, dataset, args.output_dir, args.max_samples
            )
            
            summary_results[dataset_name] = {
                'avg_rouge_l': avg_rouge_l,
                'num_samples': len(dataset) if not args.max_samples else min(len(dataset), args.max_samples)
            }
            
            all_detailed_results.extend(detailed_results)
            
        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
            continue
    
    # Save summary results
    summary_file = os.path.join(args.output_dir, "summary_results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)
    
    # Save all detailed results
    all_results_file = os.path.join(args.output_dir, "all_detailed_results.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_detailed_results, f, ensure_ascii=False, indent=2)
    
    # Display summary results
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results Summary:")
    logger.info("="*50)
    for dataset_name, result in summary_results.items():
        logger.info(f"{dataset_name:15} | Rouge-L: {result['avg_rouge_l']:.4f} | Samples: {result['num_samples']}")
    
    # Calculate overall average score
    if summary_results:
        overall_avg = sum(r['avg_rouge_l'] for r in summary_results.values()) / len(summary_results)
        logger.info("="*50)
        logger.info(f"Overall average Rouge-L score: {overall_avg:.4f}")
    
    logger.info(f"\nDetailed results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
