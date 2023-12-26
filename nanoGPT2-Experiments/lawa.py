import torch
import os
import re
import pickle
from tqdm import tqdm
from model import GPTConfig, GPT
import gc

# Assuming you have already defined the necessary components of your model and other required functions
# Step 1: Make a list of Models.
model_location = '/scratch/07946/ss95332/out_far/'
NUM_MODELS = 5
selected_files = []
model_paths = []
models = []
interval = 5  # Size of the moving window
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
block_size = 1024  # seq length
scale_attn_by_inverse_layer_idx = True
dataset = 'openwebtext'
data_dir = os.path.join('data', dataset)
# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx)  # start with model_args from command line


def extract_number(filename):
    """Extracts the number from the filename."""
    match = re.search(r'ckpt_(\d+).pt', filename)
    if match:
        return int(match.group(1))
    else:
        return None  # or raise an error if no match is found


# Load all the models
model_paths = [os.path.join(model_location, file) for file in os.listdir(model_location) if file.endswith('.pt')]
model_paths.sort(key=extract_number)
# print(model_paths)

# Taking K=5 models for uniform averaging
for i in range(0, len(model_paths), interval):
    selected_files.append(model_paths[i:i + interval][-NUM_MODELS:])


def is_float_dtype(dtype):
    # This method is used to check if a dtype is a float dtype.
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
            torch.float64,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        )
        ]
    )

@torch.no_grad()
def uni_update(model, avg_model, param_or_buffer_names_no_ema, num_avg_models):
    def avg_fn(averaged_model_parameter, model_parameter, num_avg_models):
        model_parameter = model_parameter.to(averaged_model_parameter.device)
        return (
                averaged_model_parameter
                + (model_parameter - averaged_model_parameter) / num_avg_models
        )

    for (current_params), (ma_params) in zip(
            list(model.parameters()), list(avg_model.parameters())
    ):
        if not is_float_dtype(current_params.dtype):
            continue

        ma_params.data.copy_(
            avg_fn(ma_params.data, current_params.data, num_avg_models)
        )

    for (name, current_buffer), (_, ma_buffer) in zip(
            list(model.named_buffers()), list(avg_model.named_buffers())
    ):
        if not is_float_dtype(current_buffer.dtype):
            continue

        if name in param_or_buffer_names_no_ema:
            ma_buffer.data.copy_(current_buffer.data)
            continue

        ma_buffer.data.copy_(
            avg_fn(ma_buffer.data, current_buffer.data, num_avg_models)
        )
    return avg_model


# Processing the selected model paths
count = 0
for models in tqdm(selected_files, desc="Outer Loop"):
    param_or_buffer_names_not_to_be_averaged = (
        set()
    )  # in case you don't want to average certain params
    print(models)
    last_ckpt_number = int(extract_number(models[-1]))
    # print(models[2:])
    num_avg_models = 1
    checkpoint = torch.load(models[0], map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    avg_model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    avg_model.load_state_dict(state_dict)
    # print(f'last_ckpt_number: {last_ckpt_number}')
    for j, model_path in enumerate(models[1:]):

        assert os.path.exists(model_path)


        assert os.path.exists(model_path)
        number_pattern = re.compile(r'ckpt_(\d+).pt')
        string = str(model_path)
        match = number_pattern.search(string)
        if match:
            ckpt_number = int(match.group(1))
        else:
            print("Pattern not found in string.")
            continue  # Skip to next iteration

        checkpoint1 = torch.load(model_path, map_location='cpu')
        checkpoint_model_args1 = checkpoint1['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args1[k]
        # create the model
        gptconf1 = GPTConfig(**model_args)
        model = GPT(gptconf1)
        state_dict1 = checkpoint1['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict1.items()):
            if k.startswith(unwanted_prefix):
                state_dict1[k[len(unwanted_prefix):]] = state_dict1.pop(k)
        model.load_state_dict(state_dict1)
        ckpt1 = checkpoint1['model']
        model.load_state_dict(ckpt1, strict=False)
        num_avg_models += 1
        avg_model = uni_update(
            model,
            avg_model,
            param_or_buffer_names_not_to_be_averaged,
            num_avg_models,
        )
        # # Save the final averaged model only if the current ckpt_number matches the last_ckpt_number
        if int(ckpt_number) == int(last_ckpt_number):

            save_obj = {
                'net': avg_model.state_dict(),
                'optimizer': checkpoint1['optimizer'],
                'model_args': checkpoint1['model_args'],
                'iter_num': checkpoint1['iter_num'],
                'best_val_loss': checkpoint1['best_val_loss'],
                'config': checkpoint1['config']

            }
            del state_dict
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            torch.save(save_obj, os.path.join("/scratch/07946/ss95332/avg-small2/", f'checkpoint_k5_{ckpt_number}.pt'))
            print(f'checkpoint_k5_{ckpt_number}.pt')
