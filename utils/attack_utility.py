import transformers
import torch
import typing
import random
import gc
import time
import copy
import peft
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import utils.experiment_logger as experiment_logger

def invertibility_filter(token_ids, **kwargs):
    tokenizer = kwargs.get("tokenizer", None)
    if tokenizer is None:
        raise ValueError(f"Tokenizer required for evaluating invertibility, you complete dingus.")
    
    try:
        bool_val = all(tokenizer.encode(tokenizer.decode(token_ids, clean_up_tokenization_spaces=False), add_special_tokens=False, return_tensors="pt")[0] == token_ids)
        return bool_val
    except Exception:
        return False

def analyze_conversation_tokens(conversation, tokenizer):
    """
    Analyzes tokenization of a conversation using the tokenizer's built-in chat template,
    separating content tokens from control tokens. Includes generation prompt in the analysis.
    
    Args:
        conversation: List of dictionaries with 'role' and 'content' keys
        tokenizer: HuggingFace tokenizer instance with chat_template
    
    Returns:
        dict: Contains lists of content token indices and control token indices,
              along with the full token list and their string representations
    """
    # Get the formatted text using the tokenizer's chat template
    formatted_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    if tokenizer.chat_template.endswith("\n") and not formatted_text.endswith("\n"):
        formatted_text = formatted_text + "\n"
    
    # Track the original content positions
    content_char_ranges = []
    for msg in conversation:
        # Find each content occurrence in the formatted text
        content = msg["content"]
        content_start = formatted_text.find(content)
        while content_start != -1:
            content_char_ranges.append(
                (content_start, content_start + len(content))
            )
            # Look for any additional occurrences
            content_start = formatted_text.find(
                content,
                content_start + 1
            )
    
    # Tokenize the full text
    tokens = tokenizer(formatted_text, return_offsets_mapping=True)
    
    # Separate content tokens from control tokens
    content_token_indices = []
    control_token_indices = []
    generation_prompt_indices = []
    
    # Get the offset mapping
    offset_mapping = tokens["offset_mapping"]
    
    # Find generation prompt location
    generation_start = len(formatted_text)
    # Any tokens that start after or at the generation prompt position
    # should be considered part of the generation prompt
    
    # Analyze each token
    for i, (start, end) in enumerate(offset_mapping):
        # Special tokens have (0,0) offset
        if start == end == 0:
            control_token_indices.append(i)
            continue
            
        # Check if token is part of the generation prompt
        if start >= generation_start:
            generation_prompt_indices.append(i)
            continue
            
        # Check if token falls within any content range
        is_content = False
        for content_start, content_end in content_char_ranges:
            # Token is content if it overlaps with content range
            if not (end <= content_start or start >= content_end):
                content_token_indices.append(i)
                is_content = True
                break
        
        if not is_content:
            control_token_indices.append(i)
    
    # Get the actual tokens for reference
    
    return {
        "content_token_indices": content_token_indices,
        "control_token_indices": control_token_indices,
        "generation_prompt_indices": generation_prompt_indices,
        "formatted_text": formatted_text
    }

ADV_SUFFIX_INDICATOR = "<ADV_SUFFIX>"
ADV_PREFIX_INDICATOR = "<ADV_PREFIX>"
def string_masks(
    tokenizer: "transformers.AutoTokenizer",
    input_string_template: str,
    adv_pre_init: str,
    adv_suf_init: str,
    target_string: str,
    prefix_placeholder: str = "<ADV_PREFIX>",
    suffix_placeholder: str = "<ADV_SUFFIX>",
):
    """
    Create masks for different parts of a tokenized string.
    
    Args:
        tokenizer: HuggingFace tokenizer
        input_string_template: Template string with placeholders for adversarial prefix and suffix
        adv_pre_init: String to replace prefix_placeholder
        adv_suf_init: String to replace suffix_placeholder
        target_string: Target string to be appended after the input string
        prefix_placeholder: Placeholder for prefix in the template
        suffix_placeholder: Placeholder for suffix in the template
    
    Returns:
        Dictionary containing tokens and various masks
    """
    # Replace placeholders in the template with the initial strings
    if prefix_placeholder in input_string_template and suffix_placeholder in input_string_template:
        prefix_pos = input_string_template.find(prefix_placeholder)
        suffix_pos = input_string_template.find(suffix_placeholder)
        
        # Extract the payload string between placeholders
        payload_string = input_string_template[prefix_pos + len(prefix_placeholder):suffix_pos]
        
        # Create the full text by replacing placeholders
        full_text = (
            input_string_template[:prefix_pos] + 
            adv_pre_init + 
            payload_string + 
            adv_suf_init + 
            input_string_template[suffix_pos + len(suffix_placeholder):]
        ) + target_string
    else:
        raise ValueError(f"Template must contain both {prefix_placeholder} and {suffix_placeholder}")
    
    # Tokenize the full text
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = encoding.input_ids
    char_spans = encoding.offset_mapping
    
    # Convert to torch tensor for consistency with the existing code
    final_tokens = torch.tensor(tokens)
    seq_length = len(final_tokens)
    
    # Find spans for different components
    prefix_span = find_clean_token_span(tokenizer, full_text, adv_pre_init, final_tokens)
    suffix_span = find_clean_token_span(tokenizer, full_text, adv_suf_init, final_tokens)
    
    # Create masks using clean token spans
    prefix_mask = torch.zeros(seq_length, dtype=torch.bool)
    suffix_mask = torch.zeros(seq_length, dtype=torch.bool)
    target_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    if prefix_span:
        prefix_mask[prefix_span["start"]:prefix_span["end"]] = True
    if suffix_span:
        suffix_mask[suffix_span["start"]:suffix_span["end"]] = True
    
    # Create payload mask (between prefix and suffix)
    payload_span = find_containing_token_span(tokenizer, full_text, payload_string, final_tokens)

    payload_mask = torch.zeros(seq_length, dtype=torch.bool)
    payload_mask[payload_span["start"]:payload_span["end"]] = True

    # Create target mask
    target_start = len(full_text) - len(target_string)
    for i, (start, end) in enumerate(char_spans):
        if start >= target_start:
            target_mask[i] = True
    
    # Create input mask (everything before target)
    input_mask = torch.ones(seq_length, dtype=torch.bool)
    input_mask[target_mask.nonzero()] = False
    
    # Convert boolean masks to index tensors for compatibility with existing code
    prefix_indices = torch.where(prefix_mask)[0]
    suffix_indices = torch.where(suffix_mask)[0]
    payload_indices = torch.where(payload_mask)[0]
    target_indices = torch.where(target_mask)[0]
    input_indices = torch.where(input_mask)[0]
    
    if len(prefix_indices) > 0:
        assert min(payload_indices) > max(prefix_indices)
    if len(suffix_indices) > 0:
        assert max(payload_indices) < min(suffix_indices)

    # Create the optim_mask as the combination of prefix and suffix indices
    optim_mask = torch.cat([prefix_indices, suffix_indices])
    
    return {
        "tokens": final_tokens,
        "masks": {
            "optim_mask": optim_mask,
            "prefix_mask": prefix_indices,
            "suffix_mask": suffix_indices,
            "target_mask": target_indices,
            "input_mask": input_indices,
            "payload_mask": payload_indices
        }
    }


def find_containing_token_span(tokenizer: transformers.PreTrainedTokenizer,
                               full_text: str,
                               target_text: str,
                               full_tokens: typing.List[int]) -> None | typing.Dict[str, typing.Any]:
    """
    Find the smallest contiguous sequence of tokens that cleanly contains the target_text.
    """
    # Get the token ids and offsets for the full text
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    char_spans = encoding.offset_mapping
    
    # Find the character positions of target_text in full_text
    start_pos = full_text.find(target_text)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(target_text)
    
    # Find the smallest token span that contains the target text
    token_start = None
    token_end = None
    
    # Find the first token that contains or is after the start position
    for i, (start, end) in enumerate(char_spans):
        # Token that contains or is after the start position
        if end > start_pos:
            token_start = i
            break
    
    # Find the last token that contains or is before the end position
    for i in range(len(char_spans) - 1, -1, -1):
        start, end = char_spans[i]
        # Token that contains or is before the end position
        if start < end_pos:
            token_end = i + 1  # +1 because end is exclusive
            break
    
    if token_start is None or token_end is None:
        return None
    
    # Verify the decoded tokens contain the target_text
    span_tokens = full_tokens[token_start:token_end]
    decoded = tokenizer.decode(span_tokens, clean_up_tokenization_spaces=False)
    
    # assert target_text in decoded, f"Target text: {target_text} not found in decoded string: {decoded}"
    
    return {
        "start": token_start,
        "end": token_end,
        "text": decoded
    }

def find_clean_token_span(tokenizer: transformers.PreTrainedTokenizer, 
                         full_text: str,
                         target_text: str,
                         full_tokens: typing.List[int]) -> None | typing.Dict[str, typing.Any]:
    """
    Find the largest contiguous sequence of tokens that cleanly maps to a substring of target_text.
    """
    # Get the token ids and offsets for the full text
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    char_spans = encoding.offset_mapping
    
    # Find the character positions of target_text in full_text
    start_pos = full_text.find(target_text)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(target_text)
    
    # Find token spans that fall within these character positions
    token_start = None
    token_end = None
    
    for i, (start, end) in enumerate(char_spans):
        # Token completely within target_text
        if start >= start_pos and end <= end_pos:
            if token_start is None:
                token_start = i
            token_end = i + 1
    
    if token_start is None:
        return None
        
    # Verify the decoded tokens match a substring of target_text
    span_tokens = full_tokens[token_start:token_end]
    decoded = tokenizer.decode(span_tokens, clean_up_tokenization_spaces=False)
    
    assert decoded in target_text, f"Decoded string: {decoded} not a subset of target_text: {target_text}"
    return {
        "start": token_start,
        "end": token_end,
        "text": decoded
    }

def conversation_masks(
    tokenizer: transformers.PreTrainedTokenizer,
    conversation: typing.List[typing.Dict[str, str]],
    adv_prefix_init: str,
    adv_suffix_init: str,
    target_string: str,
    prefix_placeholder: str = "<ADV_PREFIX>",
    suffix_placeholder: str = "<ADV_SUFFIX>"
) -> typing.Dict[str, typing.Any]:
    """
    Create masks for different parts of a tokenized conversation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        conversation: List of dictionaries with 'role' and 'content' keys
        adv_prefix_init: String to replace prefix_placeholder
        adv_suffix_init: String to replace suffix_placeholder
        target_string: Target string to be appended after the conversation
        prefix_placeholder: Placeholder for prefix in the content
        suffix_placeholder: Placeholder for suffix in the content
    
    Returns:
        Dictionary containing tokens and various masks
    """
    # First, process the conversation by replacing placeholders
    processed_conversation = []
    for turn in conversation:
        content = turn['content']
        
        # Replace placeholders in content
        if prefix_placeholder in content and suffix_placeholder in content:
            prefix_pos = content.find(prefix_placeholder)
            suffix_pos = content.find(suffix_placeholder)
            
            payload_string = content[prefix_pos + len(prefix_placeholder):suffix_pos]

            content = (
                content[:prefix_pos] + 
                adv_prefix_init + 
                content[prefix_pos + len(prefix_placeholder):suffix_pos] + 
                adv_suffix_init + 
                content[suffix_pos + len(suffix_placeholder):]
            )
        
        processed_conversation.append({
            "role": turn['role'],
            "content": content
        })
    
    full_text = tokenizer.apply_chat_template(
        processed_conversation,  # Exclude the target string message
        tokenize=False,
        add_generation_prompt=True,    # Let the tokenizer handle the generation prompt
    ) 
    if tokenizer.chat_template.endswith("\n") and not full_text.endswith("\n"):
        full_text = full_text + "\n"

    full_text = full_text + target_string  # Add target string separately to maintain control over its position
    
    # Tokenize the conversation without the target string first
    
    conversation_tokens_input = tokenizer.apply_chat_template(
        processed_conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    if tokenizer.chat_template.endswith("\n") and not conversation_tokens_input.endswith("\n"):
        conversation_tokens_input = conversation_tokens_input + "\n"

    
    conversation_tokens = tokenizer(
        conversation_tokens_input,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    # Tokenize the target string separately
    target_tokens = tokenizer(
        target_string,
        return_offsets_mapping=True,
        add_special_tokens=False  # No special tokens for target as they're already in the conversation
    )
    
    # Combine tokens
    final_tokens = conversation_tokens['input_ids'] + target_tokens['input_ids']
    final_tokens = torch.tensor(final_tokens)

    # Combine offset mappings, adjusting target offsets
    last_offset = len(full_text) - len(target_string)
    target_offsets = [(start + last_offset, end + last_offset) 
                     for start, end in target_tokens.offset_mapping]
    char_spans = conversation_tokens.offset_mapping + target_offsets
    
    # Initialize masks
    seq_length = len(final_tokens)
    prefix_mask = torch.zeros(seq_length, dtype=torch.bool)
    suffix_mask = torch.zeros(seq_length, dtype=torch.bool)
    content_mask = torch.zeros(seq_length, dtype=torch.bool)
    target_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    # Find clean token spans for prefix and suffix
    prefix_span = find_clean_token_span(tokenizer, full_text, adv_prefix_init, final_tokens)
    suffix_span = find_clean_token_span(tokenizer, full_text, adv_suffix_init, final_tokens)
    
    if prefix_span:
        prefix_mask[prefix_span["start"]:prefix_span["end"]] = True
    if suffix_span:
        suffix_mask[suffix_span["start"]:suffix_span["end"]] = True
    
    # Create content mask - we'll identify content by finding non-template parts in each message
    for turn in processed_conversation:  # Exclude target message
        turn_text = turn['content']
        # Find this content in the full text and create mask for its tokens
        start_pos = full_text.find(turn_text)
        if start_pos != -1:
            end_pos = start_pos + len(turn_text)
            for i, (start, end) in enumerate(char_spans):
                if start >= start_pos and end <= end_pos:
                    content_mask[i] = True
    
    # Create payload mask (between prefix and suffix)
    payload_span = find_containing_token_span(tokenizer, full_text, payload_string, final_tokens)
    payload_mask = torch.zeros(seq_length, dtype=torch.bool)
    payload_mask[payload_span["start"]:payload_span["end"]] = True
    
    # Create target mask
    target_start = len(full_text) - len(target_string)
    for i, (start, end) in enumerate(char_spans):
        if start >= target_start:
            target_mask[i] = True
    
    # Create input mask (everything before target)
    input_mask = torch.ones(seq_length, dtype=torch.bool)
    input_mask[target_mask.nonzero()] = False
    
    control_mask = ~(content_mask | target_mask)

    prefix_indices = torch.where(prefix_mask)[0]
    suffix_indices = torch.where(suffix_mask)[0]
    payload_indices = torch.where(payload_mask)[0]
    target_indices = torch.where(target_mask)[0]
    input_indices = torch.where(input_mask)[0]
    content_indices = torch.where(content_mask)[0]
    control_indices = torch.where(control_mask)[0]

    return {
        "tokens": final_tokens,
        "masks": {
            "optim_mask": torch.cat([prefix_indices, suffix_indices]),
            "prefix_mask": prefix_indices,
            "suffix_mask": suffix_indices,
            "payload_mask": payload_indices,
            "target_mask": target_indices,
            "input_mask": input_indices,
            "content_mask": content_indices,
            "control_mask": control_indices
        }
    }

def DEFAULT_FILTER_FUNCTION(tokens: torch.tensor, **kwargs):
    return True

INITIAL_PREFIX_LENGTH = 40
INITIAL_SUFFIX_LENGTH = 40
DEFAULT_INIT_TOKEN = " And"
def initialize_adversarial_strings(tokenizer: transformers.AutoTokenizer, init_config: typing.Dict):

    adv_prefix_init: str
    adv_suffix_init: str

    try:
        init_strategy_type = init_config["strategy_type"]
    except KeyError:
        raise ValueError("strategy_type needs to be in initialization strategy.")

    if init_strategy_type == "random":
        try:
            random_seed = init_config["seed"]
            random.seed(random_seed)
        except KeyError:
            pass

        try:
            prefix_filter: typing.Callable[..., bool] = init_config["prefix_filter"]
        except KeyError:
            prefix_filter = DEFAULT_FILTER_FUNCTION
        
        try:
            suffix_filter: typing.Callable[..., bool] = init_config["suffix_filter"]
        except KeyError:
            suffix_filter = DEFAULT_FILTER_FUNCTION
        
        try:
            filter_metadata = init_config["filter_metadata"]
        except KeyError:
            filter_metadata = None

        try:
            prefix_length = init_config["prefix_length"]
        except KeyError:
            prefix_length = INITIAL_PREFIX_LENGTH
        
        try:
            suffix_length = init_config["suffix_length"]
        except KeyError:
            suffix_length = INITIAL_SUFFIX_LENGTH

        while True:
            prefix_random_tokens = []
            for _ in range(prefix_length):
                rand_token = random.randint(0, tokenizer.vocab_size)
                prefix_random_tokens.append(rand_token)
            prefix_random_tokens = torch.tensor(prefix_random_tokens)
            if prefix_filter(prefix_random_tokens, **(filter_metadata or {})):
                break
        
        while True:
            suffix_random_tokens = []
            for _ in range(suffix_length):
                rand_token = random.randint(0, tokenizer.vocab_size)
                suffix_random_tokens.append(rand_token)
            suffix_random_tokens = torch.tensor(suffix_random_tokens)
            if suffix_filter(suffix_random_tokens, **(filter_metadata or {})):
                break
        
        adv_prefix_init = tokenizer.decode(prefix_random_tokens)
        adv_suffix_init = tokenizer.decode(suffix_random_tokens)

    elif init_strategy_type == "fixed_string":
        try:
            adv_prefix_init = init_config["adv_prefix_init"]
            adv_suffix_init = init_config["adv_suffix_init"]
        except KeyError:
            raise ValueError("Both adv_prefix_init and adv_suffix_init need to be if your initialization strategy is fixed_string")

    elif init_strategy_type == "fixed_length_const_init":
        try:
            prefix_length = init_config["prefix_length"]
        except KeyError:
            prefix_length = INITIAL_PREFIX_LENGTH
        
        try:
            prefix_token = init_config["prefix_token"]
        except KeyError:
            prefix_token = DEFAULT_INIT_TOKEN

        try:
            suffix_length = init_config["suffix_length"]
        except KeyError:
            suffix_length = INITIAL_SUFFIX_LENGTH
        
        try:
            suffix_token = init_config["suffix_token"]
        except KeyError:
            suffix_token = DEFAULT_INIT_TOKEN
        
        adv_prefix_init = prefix_token * prefix_length
        adv_suffix_init = suffix_token * suffix_length
    
    else:
        raise ValueError("Initialization Strategy not recognized")
    
    return adv_prefix_init, adv_suffix_init

BULK_FORWARD_DEFAULT_BSZ = 512
DEFAULT_GENERATION_PARAMS = {
    "logprobs": True,
}

def bulk_logits_iter(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=8,
    generation_params=DEFAULT_GENERATION_PARAMS,
):
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            data_piece = data[i:i + batch_size]
            try:
                logits = model(input_ids=data_piece.to(model.device)).logits
                yield logits
                gc.collect()
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                gc.collect()
                torch.cuda.empty_cache()
                sub_iterator = bulk_logits_iter(
                    model, 
                    data_piece, 
                    batch_size // 2, 
                    generation_params,
                )
                for sub_result in sub_iterator:
                    yield sub_result
                    del sub_result
                    gc.collect()
                    torch.cuda.empty_cache()
                del sub_iterator
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

BULK_ATT_FORWARD_DEFAULT_SIZE=128
def bulk_forward_iter(
    model: transformers.AutoModelForCausalLM,
    data: torch.tensor,
    batch_size=BULK_ATT_FORWARD_DEFAULT_SIZE,
    generation_params=DEFAULT_GENERATION_PARAMS,
) -> typing.Iterator[typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, ...]]]:
    """
    Iterator that yields both logits and attentions one batch at a time.
    Now supports past_key_values for prefix caching.
    
    Returns:
        Iterator yielding tuples of (logits, attentions) where:
        - logits: tensor of shape (batch_size, sequence_length, vocab_size)
        - attentions: tuple of attention tensors for each layer
    """
    if batch_size <= 32:
        raise ValueError(f"Can't with smaller sizes. Moving on.")

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            current_batch_size = min(batch_size, len(data) - i)
            data_piece = data[i:i + batch_size]
            
            # If past_key_values is provided, expand it to match the current batch size
            try:
                output = model(
                    input_ids=data_piece.to(model.device),
                    output_attentions=True
                )
                yield output.logits, output.attentions
                
            except torch.cuda.OutOfMemoryError:
                # If OOM occurs, recursively process with smaller batch size
                gc.collect()
                torch.cuda.empty_cache()
                sub_iterator = bulk_forward_iter(
                    model, 
                    data_piece, 
                    batch_size // 2, 
                    generation_params,
                )
                for sub_logits, sub_attentions in sub_iterator:
                    yield sub_logits, sub_attentions
                    del sub_logits, sub_attentions
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Clean up memory after each batch
            gc.collect()
            torch.cuda.empty_cache()


UNREDUCED_CE_LOSS = torch.nn.CrossEntropyLoss(reduction="none")
def target_logprobs(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    input_points: torch.tensor,
    masks_data: typing.Dict[str, torch.tensor],
    target_tokens: torch.tensor,
    logger: experiment_logger.ExperimentLogger = None,
    **kwargs
):
    target_mask = masks_data["target_mask"]
    losses_list = []
    for logit_piece in bulk_logits_iter(model, input_points):
        loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, -(len(target_mask) + 1):- 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
        losses_list.append(loss_tensor)
    loss_tensor = torch.cat(losses_list)
    return loss_tensor


DEFAULT_TEXT_GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 20
}

def default_best_choice_function(model, tokenizer, input_tokenized_data, best_tokens_sequences, logger, **kwargs):
    masks_data = input_tokenized_data["masks"]
    best_index = torch.argmin(target_logprobs(model, tokenizer, torch.stack(best_tokens_sequences), masks_data, input_tokenized_data["tokens"][masks_data["target_mask"]], logger))
    return {
        "tokens": best_tokens_sequences[best_index],
        "masks": masks_data
    }


def generate_valid_input_tokenized_data(
    tokenizer,
    input_template,
    target_output_str,
    init_config,
    logger: experiment_logger.ExperimentLogger,
    *,
    max_attempts = 10000
):
    new_init_config = copy.deepcopy(init_config)
    num_init_tries = 0
    # if new_init_config["prefix_length"] > 0:
    #     new_init_config["prefix_length"] = new_init_config["prefix_length"] - 1
    # if new_init_config["suffix_length"] > 0:
    #     new_init_config["suffix_length"] = new_init_config["suffix_length"] - 1

    while num_init_tries < max_attempts:
        try:
            adv_prefix_init, adv_suffix_init = initialize_adversarial_strings(tokenizer, new_init_config)
            if isinstance(input_template, str):
                input_tokenized_data = string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
            elif isinstance(input_template, list):
                input_tokenized_data = conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
            
            masks_data = input_tokenized_data["masks"]
            if len(masks_data["prefix_mask"]) > new_init_config.get("prefix_length", 10000):
                raise ValueError(f"Prefix is too long.")
            if len(masks_data["suffix_mask"]) > new_init_config.get("suffix_length", 10000):
                raise ValueError(f"Suffix is too long.")

        except Exception as e:
            INIT_TOKENIZATION_FAILED = f"The given initialization failed due to the following reasons - {str(e)}"
            # logger.log(INIT_TOKENIZATION_FAILED)
            if new_init_config["strategy_type"] != "random":
                raise ValueError(f"{INIT_TOKENIZATION_FAILED}")
            new_seed = int(time.time())
            RETRYING_STRING = f"Retrying with another random seed: {str(new_seed)}"
            new_init_config["seed"] = new_seed
            # logger.log(RETRYING_STRING)
        else:
            break
        num_init_tries += 1
            
    logger.log(new_init_config, num_init_tries=num_init_tries)
    return input_tokenized_data, new_init_config

def generate_bulk_valid_input_tokenized_data(
    tokenizer,
    input_templates,
    target_output_str,
    init_config,
    logger: experiment_logger.ExperimentLogger,
    *,
    max_attempts = 10000
):
    new_init_config = copy.deepcopy(init_config)
    num_init_tries = 0
    # if new_init_config["prefix_length"] > 0:
    #     new_init_config["prefix_length"] = new_init_config["prefix_length"] - 1
    # if new_init_config["suffix_length"] > 0:
    #     new_init_config["suffix_length"] = new_init_config["suffix_length"] - 1

    input_tokenized_data_list = []
    while num_init_tries < max_attempts:
        try:
            adv_prefix_init, adv_suffix_init = initialize_adversarial_strings(tokenizer, new_init_config)
            for input_template in input_templates:
                if isinstance(input_template, str):
                    input_tokenized_data = string_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                elif isinstance(input_template, list):
                    input_tokenized_data = conversation_masks(tokenizer, input_template, adv_prefix_init, adv_suffix_init, target_output_str)
                
                masks_data = input_tokenized_data["masks"]
                if len(masks_data["prefix_mask"]) > new_init_config.get("prefix_length", 10000):
                    raise ValueError(f"Prefix is too long.")
                if len(masks_data["suffix_mask"]) > new_init_config.get("suffix_length", 10000):
                    raise ValueError(f"Suffix is too long.")
                input_tokenized_data_list.append(input_tokenized_data)
        except Exception as e:
            INIT_TOKENIZATION_FAILED = f"The given initialization failed due to the following reasons - {str(e)}"
            # logger.log(INIT_TOKENIZATION_FAILED)
            if new_init_config["strategy_type"] != "random":
                raise ValueError(f"{INIT_TOKENIZATION_FAILED}")
            new_seed = int(time.time())
            RETRYING_STRING = f"Retrying with another random seed: {str(new_seed)}"
            new_init_config["seed"] = new_seed
            input_tokenized_data_list = []
            # logger.log(RETRYING_STRING)
        else:
            break
        num_init_tries += 1

    logger.log(new_init_config, num_init_tries=num_init_tries)
    return input_tokenized_data_list, new_init_config


def _get_layer_obj(model):
    if isinstance(model, peft.PeftModel):
        return model.base_model.model.model.layers
    elif isinstance(model, transformers.LlamaPreTrainedModel):
        return model.model.layers
    elif isinstance(model, transformers.MistralPreTrainedModel):
        return model.model.layers


DEFAULT_MAXIMUM_BATCH_SIZE = 512
class CachedTargetLogprobs:

    def _cache_init(self, model, tokenizer, input_tokenized_data):
        tokens = input_tokenized_data["tokens"]
        masks_data = input_tokenized_data["masks"]
        optim_mask = masks_data["optim_mask"]
        static_index = min(optim_mask) - 1
        static_tokens = tokens[:static_index]
        past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
        self.cache_object = {
            "past_key_values": past_key_values,
            "static_index": static_index
        }

    def _batch_size_init(self, model, tokenizer, input_tokenized_data):
        tokens = input_tokenized_data["tokens"]
        batch_size = DEFAULT_MAXIMUM_BATCH_SIZE
        with torch.no_grad():
            if self.to_cache:
                past_key_values_true = self.cache_object["past_key_values"]
                static_index_true = self.cache_object["static_index"]

                while batch_size > 1:
                    input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index_true:]
                    batched_kv_cache = []
                    for keys_cached, values_cached in past_key_values_true:
                        keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                        values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                        batched_kv_cache.append((keys_cached_new, values_cached_new))
                    try:
                        dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                        output = model(
                            input_ids = input_ids_sliced_batch.to(model.device),
                            past_key_values = dynamic_cache
                        ).logits
                        self.batch_size = batch_size // 2
                        for pair in batched_kv_cache:
                            del pair
                        del batched_kv_cache
                        del output, dynamic_cache
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        break
                    except torch.cuda.OutOfMemoryError:
                        del dynamic_cache, batched_kv_cache
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size //= 2

    def __init__(self, to_cache=True):
        self.to_cache = to_cache
        self.is_inited = False
        self.cache_object = None
        self.batch_size = None

    def __call__(self, model, tokenizer, input_points, masks_data, target_tokens, logger, **kwargs):

        if not self.is_inited:
            input_tokenized_data = {
                "tokens": input_points[0],
                "masks": masks_data
            }
            self._cache_init(model, tokenizer, input_tokenized_data)
            self._batch_size_init(model, tokenizer, input_tokenized_data)
            self.is_inited = True
        
        gc.collect()
        torch.cuda.empty_cache()
        input_points_sliced = input_points[:, self.cache_object["static_index"]:]
        target_mask = masks_data["target_mask"]
        data_split = torch.split(input_points_sliced, self.batch_size, dim=0)
        losses_list = []
        with torch.no_grad():
            for data_batch in data_split:
                    new_legacy_cache = []
                    for key_cache, value_cache in self.cache_object["past_key_values"]:
                        new_legacy_cache.append((key_cache.expand(data_batch.shape[0], -1, -1, -1).clone(), value_cache.expand(data_batch.shape[0], -1, -1, -1).clone()))

                    dynamic_cache = transformers.DynamicCache.from_legacy_cache(new_legacy_cache)
                    output = model(input_ids=data_batch.to(model.device), past_key_values=dynamic_cache)
                    logit_piece = output.logits
                    loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, -(len(target_mask) + 1):- 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
                    losses_list.append(loss_tensor.detach())
                    for pair in new_legacy_cache:
                        del pair
                    del new_legacy_cache, dynamic_cache, output
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
        losses_tensor = torch.cat(losses_list)
        return losses_tensor

class CachedBulkForward:
    def _cache_init(self, model, tokenizer, input_tokenized_data):
        tokens = input_tokenized_data["tokens"]
        masks_data = input_tokenized_data["masks"]
        optim_mask = masks_data["optim_mask"]
        static_index = min(optim_mask) - 1
        static_tokens = tokens[:static_index]
        past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
        self.cache_object = {
            "past_key_values": past_key_values,
            "static_index": static_index
        }

    def _batch_size_init(self, model, tokenizer, input_tokenized_data):
        tokens = input_tokenized_data["tokens"]
        batch_size = DEFAULT_MAXIMUM_BATCH_SIZE
        with torch.no_grad():
            past_key_values_true = self.cache_object["past_key_values"]
            static_index_true = self.cache_object["static_index"]

            while batch_size > 1:
                input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index_true:]
                batched_kv_cache = []
                for keys_cached, values_cached in past_key_values_true:
                    keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                    keys_cached_new.requires_grad = False
                    values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                    values_cached_new.requires_grad = False
                    batched_kv_cache.append((keys_cached_new, values_cached_new))
                try:
                    dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                    output = model(
                        input_ids = input_ids_sliced_batch.to(model.device),
                        past_key_values = dynamic_cache,
                        output_attentions = True
                    )
                    self.batch_size = batch_size // 2
                    for pair in batched_kv_cache:
                        del pair
                    del batched_kv_cache
                    del output, dynamic_cache
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    del batched_kv_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2

    def __init__(self, to_cache=True):
        self.is_inited = False
        self.cache_object = None
        self.batch_size = None

    def __call__(self, model, tokenizer, input_points, masks_data, logger, **kwargs):
        
        if not self.is_inited:
            input_tokenized_data = {
                "tokens": input_points[0],
                "masks": masks_data
            }
            self._cache_init(model, tokenizer, input_tokenized_data)
            self._batch_size_init(model, tokenizer, input_tokenized_data)
            self.is_inited = True
        
        input_points_sliced = input_points[:, self.cache_object["static_index"]:]
        data_split = torch.split(input_points_sliced, self.batch_size, dim=0)
        for data_batch in data_split:
            gc.collect()
            torch.cuda.empty_cache()
            new_legacy_cache = []
            for key_cache, value_cache in self.cache_object["past_key_values"]:
                new_legacy_cache.append((key_cache.expand(data_batch.shape[0], -1, -1, -1).clone(), value_cache.expand(data_batch.shape[0], -1, -1, -1).clone()))
            with torch.no_grad():
                output = model(input_ids=data_batch.to(model.device), past_key_values=transformers.DynamicCache.from_legacy_cache(new_legacy_cache), output_attentions=True)
                logits = output.logits
                attentions = output.attentions
                yield logits, attentions
                for pair in new_legacy_cache:
                    del pair
                del output, logits, attentions, new_legacy_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

class CachedAverageLogprobs:

    def _cache_init(self, models, tokenizer, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _cache_object = []
        _static_indices = []   
        for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches, strict=True):
            cache_object_batch = []
            static_index_batch = []
            for input_tokenized_data in input_tokenized_data_list_batch:
                tokens = input_tokenized_data["tokens"]
                masks_data = input_tokenized_data["masks"]
                optim_mask = masks_data["optim_mask"]
                static_index = min(optim_mask) - 1
                static_tokens = tokens[:static_index]
                past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
                cache_object_batch.append(past_key_values)
                static_index_batch.append(static_index)
            _cache_object.append(cache_object_batch)
            _static_indices.append(static_index_batch)
        
        self.cache_object = _cache_object
        self.static_indices = _static_indices

    def _find_single_element_batch_size(self, model, input_tokenized_data, past_key_values, static_index):
        with torch.no_grad():
            tokens = input_tokenized_data["tokens"]
            batch_size = DEFAULT_MAXIMUM_BATCH_SIZE
            while batch_size > 1:
                input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index:]
                batched_kv_cache = []
                for keys_cached, values_cached in past_key_values:
                    keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                    values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                    batched_kv_cache.append((keys_cached_new, values_cached_new))
                try:
                    dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                    output = model(
                        input_ids = input_ids_sliced_batch.to(model.device),
                        past_key_values = dynamic_cache
                    ).logits
                    for pair in batched_kv_cache:
                        del pair
                    del batched_kv_cache
                    del output, dynamic_cache
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    del dynamic_cache, batched_kv_cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
        
        return batch_size // 2

    def _batch_size_init(self, models, input_tokenized_data_list):

        num_elements_per_batch = len(input_tokenized_data_list) // len(models)
        input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        _batch_sizes = []
        for model, input_tokenized_data_list_batch, cache_object_batch, static_index_batch in zip(models, input_tokenized_data_list_batches, self.cache_object, self.static_indices, strict=True):
            per_device_batch_sizes = []
            for input_tokenized_data, cache_object, static_index in zip(input_tokenized_data_list_batch, cache_object_batch, static_index_batch, strict=True):
                single_example_batch_size = self._find_single_element_batch_size(model, input_tokenized_data, cache_object, static_index)
                per_device_batch_sizes.append(single_example_batch_size)
            _batch_sizes.append(per_device_batch_sizes)
        
        self.batch_sizes = _batch_sizes

    def _single_thread_call(self,
        model,
        tokenizer,
        batch_id,
        input_points_list_batch,
        masks_data_list_batch,
        logger,
        **kwargs
    ):
        cache_object_batch = self.cache_object[batch_id]
        static_index_batch = self.static_indices[batch_id]
        batch_size_batch = self.batch_sizes[batch_id]

        losses_list_batch = []
        for input_points, masks_data, past_key_values, static_index, batch_size in zip(input_points_list_batch, masks_data_list_batch, cache_object_batch, static_index_batch, batch_size_batch, strict=True):
            if input_points.dim() == 1:
                input_points = torch.unsqueeze(input_points, dim=0)

            input_points_sliced = input_points[:, static_index:]
            target_mask = masks_data["target_mask"]
            target_tokens = input_points[0, target_mask]
            data_split = torch.split(input_points_sliced, batch_size, dim=0)
            losses_list = []
            with torch.no_grad():
                for data_batch in data_split:
                        new_legacy_cache = []
                        for key_cache, value_cache in past_key_values:
                            new_legacy_cache.append((key_cache.expand(data_batch.shape[0], -1, -1, -1).clone(), value_cache.expand(data_batch.shape[0], -1, -1, -1).clone()))

                        dynamic_cache = transformers.DynamicCache.from_legacy_cache(new_legacy_cache)
                        output = model(input_ids=data_batch.to(model.device), past_key_values=dynamic_cache)
                        logit_piece = output.logits
                        loss_tensor = UNREDUCED_CE_LOSS(torch.transpose(logit_piece[:, -(len(target_mask) + 1):- 1, :], 1, 2), target_tokens.repeat((logit_piece.shape[0], 1)).to(logit_piece.device)).sum(dim=1)
                        losses_list.append(loss_tensor.detach())
                        for pair in new_legacy_cache:
                            del pair
                        del new_legacy_cache, dynamic_cache, output
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
            losses_tensor = torch.cat(losses_list)
            losses_list_batch.append(losses_tensor.to("cpu"))
        
        torch.cuda.synchronize()
        return losses_list_batch

    def _input_matches_expected_pattern(self, input_points_list):

        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        for batch_idx, input_batch_list in enumerate(input_points_list_batches):
            try:
                expected_batch_list = self.current_data_structure[batch_idx]
            except IndexError:
                return False

            try:
                assert len(input_batch_list) == len(expected_batch_list)
            except AssertionError:
                return False
        
        return True


    def _set_current_data_pattern(self, input_points_list):
        
        num_elements_per_batch = len(input_points_list) // self.num_models
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(self.num_models)]

        _current_data_structure = [None] * self.num_models

        for batch_idx, input_point_batch in enumerate(input_points_list_batches):
            _current_data_structure[batch_idx] = [1] * len(input_point_batch)
        
        self.current_data_structure = _current_data_structure

    def __init__(self):
        self.num_models = None
        self.current_data_structure = []
        self.cache_object = []
        self.batch_sizes = []
        self.static_indices = []

    def __call__(self, models, tokenizer, input_points_list, masks_data_list, logger, *, canonical_device_idx=0, **kwargs):
        
        if self.num_models is None:
            self.num_models = len(models)
        
        if len(models) != self.num_models:
            raise ValueError(f"How can you mess up the models parameter? Please do something useful.")

        if not self._input_matches_expected_pattern(input_points_list):
            input_tokenized_data_list = [
                {
                    "tokens": input_points[0],
                    "masks": masks_data
                }
                for (input_points, masks_data) in zip(input_points_list, masks_data_list, strict=True)
            ]
            self._cache_init(models, tokenizer, input_tokenized_data_list)
            self._batch_size_init(models, input_tokenized_data_list)
            self._set_current_data_pattern(input_points_list)

        num_elements_per_batch = len(input_points_list) // len(models)
        input_points_list_batches = [input_points_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]
        masks_data_list_batches = [masks_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

        results = []

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            future_to_models = [
                executor.submit(self._single_thread_call, model, tokenizer, batch_id, input_points_list_batch, masks_data_list_batch, logger, **kwargs)
                for batch_id, (model, input_points_list_batch, masks_data_list_batch) in enumerate(zip(models, input_points_list_batches, masks_data_list_batches, strict=True))
            ]

            for idx, future in enumerate(future_to_models):
                try:
                    result = future.result()  # 5 minute timeout
                    results.append((idx, result))
                except Exception as exc:
                    MODEL_EXCEPTION_STRING = f"Model {idx} generated an exception: {exc}"
                    logger.log(MODEL_EXCEPTION_STRING)
                    results.append((idx, None))  # or handle differently
                    raise RuntimeError(MODEL_EXCEPTION_STRING)

        results.sort(key = lambda x: x[0])
        
        stacked_results = []
        for _, model_result in results:
            stacked_results.append(torch.stack(model_result).to(f"cuda:{str(canonical_device_idx)}"))
        final_stacked_results = torch.cat(stacked_results)
        return final_stacked_results.mean(dim=0)
        
def normalize_mask(input_tokenized_data_list, mask_key):
    # Assumes masks are contiguous
    token_to_index_map_list = [
        {
            x: idx for x, idx in zip(input_tokenized_data["tokens"][input_tokenized_data["masks"][mask_key]], input_tokenized_data["masks"][mask_key], strict=True)
        }
        for input_tokenized_data in input_tokenized_data_list
    ]

    common_masked_tokens = set.intersection(*[set(input_tokenized_data["tokens"][input_tokenized_data["masks"][mask_key]].tolist()) for input_tokenized_data in input_tokenized_data_list])
    mask_normalized_input_tokenized_data_list = [
        {
            "tokens": input_tokenized_data["tokens"],
            "masks": (m := copy.deepcopy(input_tokenized_data["masks"])) or m.update({mask_key: torch.tensor(sorted(list({x: token_to_index_map_list[list_idx][x] for x in common_masked_tokens}.values())))}) or m
        }
        for list_idx, input_tokenized_data in enumerate(input_tokenized_data_list)
    ]
    return mask_normalized_input_tokenized_data_list

def normalize_input_tokenized_data_list(input_tokenized_data_list, *, keys_to_normalize = ["prefix_mask", "suffix_mask", "payload_mask"]):
    for mask_key in keys_to_normalize:
        input_tokenized_data_list = normalize_mask(input_tokenized_data_list, mask_key)
    return input_tokenized_data_list

def update_all_tokens(best_output_tokens_dict, input_tokenized_data_list):
    new_input_tokenized_data_list = []
    for input_tokenized_data in input_tokenized_data_list:
        new_input_tokenized_data = copy.deepcopy(input_tokenized_data)
        new_input_tokenized_data["tokens"][new_input_tokenized_data["masks"]["prefix_mask"]] = best_output_tokens_dict["prefix_tokens"]
        new_input_tokenized_data["tokens"][new_input_tokenized_data["masks"]["suffix_mask"]] = best_output_tokens_dict["suffix_tokens"]
        new_input_tokenized_data_list.append(new_input_tokenized_data)
    return new_input_tokenized_data_list

def form_best_tokens_dict(input_tokenized_data_list):
    return {
        "prefix_tokens": input_tokenized_data_list[0]["tokens"][input_tokenized_data_list[0]["masks"]["prefix_mask"]],
        "suffix_tokens": input_tokenized_data_list[0]["tokens"][input_tokenized_data_list[0]["masks"]["suffix_mask"]]
    }

def default_best_universal_choice_function(
    models,
    tokenizer,
    input_tokenized_data_list,
    all_best_tokens_dicts_list,
    logger,
    *,
    all_average_logprobs_list,
    **kwargs
):
    best_index = torch.argmin(torch.tensor(all_average_logprobs_list))
    best_output_tokens_dict = all_best_tokens_dicts_list[best_index]
    new_input_tokenized_data_list = update_all_tokens(best_output_tokens_dict, input_tokenized_data_list)
    return new_input_tokenized_data_list

def _cache_init(models, tokenizer, input_tokenized_data_list):

    num_elements_per_batch = len(input_tokenized_data_list) // len(models)
    input_tokenized_data_list_batches = [input_tokenized_data_list[x * num_elements_per_batch: (x+1) * num_elements_per_batch] for x in range(len(models))]

    _cache_object = []
    _static_indices = []    
    for model, input_tokenized_data_list_batch in zip(models, input_tokenized_data_list_batches, strict=True):
        cache_object_batch = []
        static_index_batch = []
        for input_tokenized_data in input_tokenized_data_list_batch:
            tokens = input_tokenized_data["tokens"]
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            static_index = min(optim_mask) - 1
            static_tokens = tokens[:static_index]
            past_key_values = model(input_ids=torch.unsqueeze(static_tokens, dim=0).to(model.device), use_cache=True).past_key_values
            cache_object_batch.append(past_key_values)
            static_index_batch.append(static_index)
        _cache_object.append(cache_object_batch)
        _static_indices.append(static_index_batch)

    gc.collect()
    torch.cuda.empty_cache()
    return _cache_object, _static_indices

def _find_single_element_batch_size(model, input_tokenized_data, past_key_values, static_index):
    with torch.no_grad():
        tokens = input_tokenized_data["tokens"]
        batch_size = DEFAULT_MAXIMUM_BATCH_SIZE
        while batch_size > 1:
            input_ids_sliced_batch = torch.unsqueeze(tokens, dim=0).expand(batch_size, -1)[:, static_index:]
            batched_kv_cache = []
            for keys_cached, values_cached in past_key_values:
                keys_cached_new = keys_cached.expand(batch_size, -1, -1, -1)
                values_cached_new = values_cached.expand(batch_size, -1, -1, -1)
                batched_kv_cache.append((keys_cached_new, values_cached_new))
            try:
                dynamic_cache = transformers.DynamicCache.from_legacy_cache(batched_kv_cache)
                output = model(
                    input_ids = input_ids_sliced_batch.to(model.device),
                    past_key_values = dynamic_cache,
                    output_attentions = True
                )
                for pair in batched_kv_cache:
                    del pair
                del batched_kv_cache
                del output, dynamic_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError:
                del dynamic_cache, batched_kv_cache
                gc.collect()
                torch.cuda.empty_cache()
                batch_size //= 2
    
    return batch_size // 2
