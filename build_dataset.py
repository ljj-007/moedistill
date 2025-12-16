import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
import transformers
import json
import tempfile


IGNORE_INDEX = -100

logger = logging.getLogger('__name__')

PROMPT_TEMPLATE = (
        "{instruction}</s>"
)

def load_dataset_with_fallback(file_path, cache_dir=None):
    """Try to load dataset, manually process data if fails"""
    try:
        return load_dataset("json", data_files=file_path, cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"Direct loading failed: {e}")
        logger.info("Trying manual data processing...")
        
        # Manually read and clean data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Clean data, ensure field types are consistent
        cleaned_data = []
        for item in data:
            cleaned_item = {
                'instruction': str(item.get('instruction', '')),
                'input': str(item.get('input', '')),
                'output': str(item.get('output', '')),
                'task_type': str(item.get('task_type', ''))  # Force conversion to string
            }
            cleaned_data.append(cleaned_item)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            json.dump(cleaned_data, tmp_file, ensure_ascii=False, indent=2)
            tmp_file_path = tmp_file.name
        
        try:
            dataset = load_dataset("json", data_files=tmp_file_path, cache_dir=cache_dir)
            return dataset, tmp_file_path
        except Exception as e2:
            logger.error(f"Even manual processing failed: {e2}")
            raise e2



def build_instruction_dataset(
    data_path: Union[List[str],str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int, 
    data_cache_dir = None,
    preprocessing_num_workers = None,
    use_cache = False,  # Add this parameter, default not to use cache
):

    def tokenization(examples):
        sources = []
        targets = []
        task_types = []
        prompt = PROMPT_TEMPLATE
        
        # Safely handle potentially missing fields
        instructions = examples.get('instruction', [''] * len(examples.get('input', examples.get('context', []))))
        
        # Handle different names for input fields
        if 'input' in examples:
            inputs = examples['input']
        elif 'context' in examples:
            inputs = examples['context']
        else:
            inputs = [''] * len(instructions)
        
        # Handle different names for output fields
        if 'output' in examples:
            outputs = examples['output']
        elif 'response' in examples:
            outputs = examples['response']
        else:
            outputs = [''] * len(instructions)
        
        # Handle task type field (optional)
        task_types_raw = examples.get('task_type', [''] * len(instructions))
        
        for instruction, input_text, output, task_type in zip(instructions, inputs, outputs, task_types_raw):
            if input_text is not None and input_text != "":
                instruction = str(instruction) + '\n' + str(input_text)
            source = prompt.format_map({'instruction': str(instruction)})
            target = f"{str(output)}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)
            task_types.append(str(task_type) if task_type is not None else '')  # Ensure it's a string

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids': all_input_ids, 'labels': all_labels, 'task_types': task_types}
        return results

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]
    
    for file in data_path:
        logger.info(f"Processing file: {file}")
        
        if use_cache:
            # Only use cache logic when explicitly enabled
            if data_cache_dir is None:
                data_cache_dir = str(os.path.dirname(file))
            cache_path = os.path.join(data_cache_dir, os.path.basename(file).split('.')[0] + f"_{max_seq_length}")
            os.makedirs(cache_path, exist_ok=True)
            
            try:
                processed_dataset = datasets.load_from_disk(cache_path)
                logger.info(f'training datasets-{file} has been loaded from disk')
                processed_dataset.set_format('torch')
                all_datasets.append(processed_dataset['train'])
                continue
            except Exception:
                logger.info(f'Cache not found for {file}, processing from scratch')
        
        # Process data directly, without using cache
        tmp_file_path = None
        try:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=None)
            num_samples = None
            if num_samples is not None:
                raw_dataset['train'] = raw_dataset['train'].select(range(min(num_samples, len(raw_dataset['train']))))
        except Exception:
            result = load_dataset_with_fallback(file, cache_dir=None)
            if isinstance(result, tuple):
                raw_dataset, tmp_file_path = result
            else:
                raw_dataset = result
        
        # Check dataset columns, only remove existing ones
        available_columns = raw_dataset['train'].column_names
        # print(available_columns)
        # exit()
        columns_to_remove = [col for col in ["instruction", "input", "output", "task_type"] if col in available_columns]
        
        tokenized_dataset = raw_dataset.map(
            tokenization,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=columns_to_remove,
            keep_in_memory=False,
            desc="preprocessing on dataset",
        )
        processed_dataset = tokenized_dataset

        # Check the processed dataset
        logger.info(f"Columns after processing: {processed_dataset['train'].column_names}")
        if 'task_types' not in processed_dataset['train'].column_names:
            logger.warning("task_types field is missing, will add default values")
            # Add default task_types
            def add_default_task_types(examples):
                examples['task_types'] = [''] * len(examples['input_ids'])
                return examples
            
            processed_dataset = processed_dataset.map(
                add_default_task_types,
                batched=True,
                desc="Adding default task_types"
            )

        # # Print information about the processed dataset
        # print(f"\n=== Processed Dataset Information ===")
        # print(f"Dataset size: {len(processed_dataset['train'])}")
        # print(f"Dataset columns: {processed_dataset['train'].column_names}")
        # print(f"Dataset features: {processed_dataset['train'].features}")
        # print(processed_dataset['train']["labels"][0])
        # print(processed_dataset["train"]["input_ids"][0])
        # exit()
        
        # Only save when cache is enabled
        if use_cache:
            processed_dataset.save_to_disk(cache_path)
        
        # Clean up temporary files
        if tmp_file_path:
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Safely extract required fields
        try:
            input_ids = [instance["input_ids"] for instance in instances]
            labels = [instance["labels"] for instance in instances]
        except KeyError as e:
            logger.error(f"Missing required field in dataset: {e}")
            logger.error(f"Available keys in first instance: {list(instances[0].keys())}")
            raise
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )