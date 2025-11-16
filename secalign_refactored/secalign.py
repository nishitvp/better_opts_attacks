from copy import deepcopy
import torch
import transformers
from peft import PeftModel
import os
import dataclasses
from enum import Enum
from typing import Union, List, Dict
import sys

sys.path.append("..")
from utils import attack_utility
from . import config

DEFENDED_MODEL_COMMON_PATH = "secalign_refactored/secalign_models"
MODEL_REL_PATHS = {
    
    ("mistralai", "undefended"): 'mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08',
    ("mistralai", "struq"): "mistralai/Mistral-7B-v0.1_SpclSpclSpcl_NaiveCompletion_2025-03-15-03-25-16",
    ("mistralai", "secalign"): "mistralai/Mistral-7B-v0.1_SpclSpclSpcl_None_2025-03-12-01-02-08_dpo_NaiveCompletion_2025-03-14-18-26-14",
    
    ("meta-llama", "undefended"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14",
    ("meta-llama", "struq"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_NaiveCompletion_2025-03-18-06-16-46-lr4e-6",
    ("meta-llama", "secalign"): "meta-llama/Meta-Llama-3-8B_SpclSpclSpcl_None_2025-03-12-01-02-14_dpo_NaiveCompletion_2025-03-12-05-33-03",
    
    ("huggyllama", "undefended"): "huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20",
    ("huggyllama", "struq"): "huggyllama/llama-7b_SpclSpclSpcl_NaiveCompletion_2025-03-12-01-02-37",
    ("huggyllama", "secalign"): "huggyllama/llama-7b_SpclSpclSpcl_None_2025-03-12-01-01-20_dpo_NaiveCompletion_2025-03-12-05-33-03",
    
    ("mistralai-instruct", "undefended"): "mistralai/Mistral-7B-Instruct-v0.1",
    ("mistralai-instruct", "secalign"): "mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-03-12-12-01-27",
    
    ("meta-llama-instruct", "undefended"): "meta-llama/Meta-Llama-3-8B-Instruct",
    ("meta-llama-instruct", "secalign"): "meta-llama/Meta-Llama-3-8B-Instruct_dpo_NaiveCompletion_2024-11-12-17-59-06-resized",
}

def load_model_and_tokenizer(model_path, tokenizer_path=None, device="cuda:0", **kwargs):
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        )
        .to(device)
        .eval()
    )
    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if "oasst-sft-6-llama-30b" in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if "guanaco" in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if "llama-2" in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "falcon" in tokenizer_path:
        tokenizer.padding_side = "left"
    if "mistral" in tokenizer_path:
        tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def _form_chat_template_from_frontend_delimiters(frontend_delimiters):
    formatter = config.PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    return (
        formatter
        .replace("{instruction}", "{{ messages[0]['content'] }}")
        .replace("{input}", "{{ messages[1]['content'] }}")
    )

def load_lora_model(model_name_or_path, device='0', load_model=True, **kwargs):
    configs = model_name_or_path.split('/')[-1].split('_') + ['Frontend-Delimiter-Placeholder', 'None']
    for alignment in ['dpo', 'kto', 'orpo']:
        base_model_index = model_name_or_path.find(alignment) - 1
        if base_model_index > 0: break
        else: base_model_index = False

    base_model_path = model_name_or_path[:base_model_index] if base_model_index else model_name_or_path
    frontend_delimiters = configs[1] if configs[1] in config.DELIMITERS else base_model_path.split('/')[-1]
    training_attacks = configs[2]
    if not load_model: return base_model_path, None, frontend_delimiters, None
    model, tokenizer = load_model_and_tokenizer(base_model_path, low_cpu_mem_usage=True, use_cache=False, device="cuda:" + device, **kwargs)
    
    try:
        _ = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "ABC"
            },
            {
                "role": "user",
                "content": "DEF"
            }
        ])
    except Exception:
        tokenizer.chat_template = _form_chat_template_from_frontend_delimiters(frontend_delimiters)
        _ = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "ABC"
            },
            {
                "role": "user",
                "content": "DEF"
            }
        ])

    if 'Instruct' in model_name_or_path: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    if base_model_index: model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=False)
    return model, tokenizer, frontend_delimiters, training_attacks


def maybe_load_secalign_defended_model(model_name, defence, **kwargs):

    if (model_name, defence) in MODEL_REL_PATHS:
        model_path = os.path.join(DEFENDED_MODEL_COMMON_PATH, MODEL_REL_PATHS[(model_name, defence)])
        return load_lora_model(model_path, **kwargs)
    else:

        if "Meta-SecAlign" in model_name:
            if "8B" in model_name:
                base_model_name = "secalign_refactored/secalign_models/meta-llama/Llama-3.1-8B-Instruct"
            else:
                raise ValueError(f"Meta-SecAlign model {model_name} is not supported")
            
            device = str(kwargs.pop("device", "0"))
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            load_model = kwargs.pop("load_model", True)
            if not load_model:
                return None, tokenizer, None, None
            
            if device == "cpu":
                device_str = device
            else:
                device_str = "cuda:" + device

            base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False,
                device_map=device_str,
                **kwargs)
            model = PeftModel.from_pretrained(base_model, "secalign_refactored/secalign_models/Meta-SecAlign-8B", is_trainable=False)
            return model, tokenizer, None, None

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer, None, None

def secalign_filter(token_ids, **kwargs):

    masks_data = kwargs.get("masks_data", None)
    tokenizer = kwargs.get("tokenizer", None)

    if tokenizer is None:
        raise ValueError(f"SecAlign filter function needs a tokenizer to be sent through")

    is_invertible = attack_utility.invertibility_filter(token_ids, tokenizer=tokenizer)

    if masks_data is None:
        decoded_string = tokenizer.decode(token_ids)
        return not any([spec_token_id in decoded_string for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])
    prefix_mask = masks_data["prefix_mask"]
    suffix_mask = masks_data["suffix_mask"]
    decoded_prefix = tokenizer.decode(token_ids[prefix_mask])
    decoded_suffix = tokenizer.decode(token_ids[suffix_mask])
    prefix_contains_specs = any([spec_token_id in decoded_prefix for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])
    suffix_contains_specs = any([spec_token_id in decoded_suffix for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])
    
    return (not (prefix_contains_specs or suffix_contains_specs)) and is_invertible

def struq_filter(token_ids, **kwargs):
    masks_data = kwargs.get("masks_data", None)
    tokenizer = kwargs.get("tokenizer", None)

    if tokenizer is None:
        raise ValueError(f"SecAlign filter function needs a tokenizer to be sent through")

    if masks_data is None:
        decoded_string = tokenizer.decode(token_ids)
        return not any([spec_token_id in decoded_string for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])
    prefix_mask = masks_data["prefix_mask"]
    suffix_mask = masks_data["suffix_mask"]
    decoded_prefix = tokenizer.decode(token_ids[prefix_mask])
    decoded_suffix = tokenizer.decode(token_ids[suffix_mask])
    prefix_contains_specs = any([spec_token_id in decoded_prefix for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])
    suffix_contains_specs = any([spec_token_id in decoded_suffix for spec_token_id in list(tokenizer.get_added_vocab().keys()) + config.FILTERED_TOKENS])

    return not (prefix_contains_specs or suffix_contains_specs)

def meta_secalign_filter(token_ids, **kwargs):

    masks_data = kwargs.get("masks_data", None)
    tokenizer = kwargs.get("tokenizer", None)

    if tokenizer is None:
        raise ValueError(f"SecAlign filter function needs a tokenizer to be sent through")

    is_invertible = attack_utility.invertibility_filter(token_ids, tokenizer=tokenizer)

    if masks_data is None:
        decoded_string = tokenizer.decode(token_ids)
        return not any([spec_token_id in decoded_string for spec_token_id in list(tokenizer.get_added_vocab().keys())])
    prefix_mask = masks_data["prefix_mask"]
    suffix_mask = masks_data["suffix_mask"]
    decoded_prefix = tokenizer.decode(token_ids[prefix_mask])
    decoded_suffix = tokenizer.decode(token_ids[suffix_mask])
    prefix_contains_specs = any([spec_token_id in decoded_prefix for spec_token_id in list(tokenizer.get_added_vocab().keys())])
    suffix_contains_specs = any([spec_token_id in decoded_suffix for spec_token_id in list(tokenizer.get_added_vocab().keys())])
    
    return (not (prefix_contains_specs or suffix_contains_specs)) and is_invertible

def _convert_to_secalign_format(
    input_conv,
    prompt_template,
    tokenizer,
    harmful_inst
):
    assert isinstance(input_conv, list) and all([isinstance(conv_part, dict) for conv_part in input_conv])
    inst_str = deepcopy(input_conv[0]["content"])
    data_str = deepcopy(input_conv[1]["content"])
    data_str += " " + attack_utility.ADV_PREFIX_INDICATOR + " " + harmful_inst + " " + attack_utility.ADV_SUFFIX_INDICATOR + " "
    static_string = prompt_template.format_map({"instruction": inst_str, "input": data_str})
    input_conv = tokenizer.batch_decode(tokenizer([static_string])["input_ids"], clean_up_tokenization_spaces=False)[0]
    return input_conv