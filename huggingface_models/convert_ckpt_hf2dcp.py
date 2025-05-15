from pathlib import Path
import torch
import torch.distributed.checkpoint as DCP

NUM_HIDDEN_LAYERS = 8
FIRST_K_DENSE_REPLACE = 3
N_ROUTED_EXPERTS = 256
# Suppose we have one layer MTP
MTP_LAYER_ID = 61


def list_model_fns_in_dir(folder):
    import os
    fns =  [fn for fn in os.listdir(folder) if os.path.isfile(os.path.join(folder, fn))]
    model_fns = [
        os.path.join(folder, fn)
        for fn in fns
        if (fn.endswith(".bin") and fn.startswith("pytorch_model"))
        or (fn.endswith(".safetensors") and fn.startswith("model"))
    ]
    model_fns.sort()
    return model_fns


def load_safetensors(fn):
    from safetensors import safe_open
    model = safe_open(fn, framework="pt")
    state_dict = {}
    for k in model.keys():
        state_dict[k] = model.get_tensor(k)
    return state_dict


def remove_higher_layers(state_dict, max_layer_id):
    """
    Remove all keys from a state_dict that start with "model.layers.{LAYER_ID}" 
    where LAYER_ID is greater than or equal to max_layer_id.
    
    Args:
        state_dict (dict): The model state dictionary
        max_layer_id (int): The maximum layer ID to keep
        
    Returns:
        dict: A new state dict with higher layer keys removed
    """
    import re
    # Create a new state dict to store the filtered keys
    filtered_state_dict = {}
    
    # Regular expression to match model.layers.{LAYER_ID}
    layer_pattern = re.compile(r'model\.layers\.(\d+)\.')
    
    # Iterate through all keys in the state dict
    for key in state_dict:
        # Check if this key matches the pattern
        match = layer_pattern.match(key)
        
        if match:
            # Extract the layer ID and convert to int
            layer_id = int(match.group(1))
            
            # Only keep the key if layer_id <= max_layer_id
            if layer_id <= max_layer_id:
                filtered_state_dict[key] = state_dict[key]
        else:
            # This key doesn't match the pattern, so keep it
            filtered_state_dict[key] = state_dict[key]
    
    return filtered_state_dict


@torch.inference_mode()
def convert_checkpoint_hf2dcp(
    *,
    input_dir: Path,
    output_dir: Path,
) -> None:
    # get model checkpoint fns
    model_fns = list_model_fns_in_dir(input_dir)
    # get full state_dict
    state_dict = {}
    for model_fn in model_fns:
        state_dict.update(load_safetensors(model_fn))

    # mtp
    state_dict["mtp.mtp_layers.0.hnorm.weight"] = state_dict.pop(f"model.layers.{MTP_LAYER_ID}.hnorm.weight")
    state_dict["mtp.mtp_layers.0.enorm.weight"] = state_dict.pop(f"model.layers.{MTP_LAYER_ID}.enorm.weight")
    state_dict["mtp.mtp_layers.0.eh_proj.weight"] = state_dict.pop(f"model.layers.{MTP_LAYER_ID}.eh_proj.weight")
    mtp_layerwise_expert_gate_proj_weights = []
    mtp_layerwise_up_gate_proj_weights = []
    mtp_layerwise_down_gate_proj_weights = []
    for expert_id in range(N_ROUTED_EXPERTS):
        mtp_layerwise_expert_gate_proj_weights.append(state_dict.pop(f"model.layers.{MTP_LAYER_ID}.mlp.experts.{expert_id}.gate_proj.weight"))
        mtp_layerwise_up_gate_proj_weights.append(state_dict.pop(f"model.layers.{MTP_LAYER_ID}.mlp.experts.{expert_id}.up_proj.weight"))
        mtp_layerwise_down_gate_proj_weights.append(state_dict.pop(f"model.layers.{MTP_LAYER_ID}.mlp.experts.{expert_id}.down_proj.weight"))
    state_dict[f"mtp.mtp_layers.0.layer.mlp.experts.gate_proj.weight"] = torch.cat(mtp_layerwise_expert_gate_proj_weights, dim=1).transpose(0,1)
    state_dict[f"mtp.mtp_layers.0.layer.mlp.experts.up_proj.weight"] = torch.cat(mtp_layerwise_up_gate_proj_weights, dim=1).transpose(0,1)
    state_dict[f"mtp.mtp_layers.0.layer.mlp.experts.down_proj.weight"] = torch.cat(mtp_layerwise_down_gate_proj_weights, dim=1).transpose(0,1)
    state_dict[f"mtp.mtp_layers.0.layer.mlp.e_score_correction_bias"] = state_dict.pop(f"model.layers.{MTP_LAYER_ID}.mlp.gate.e_score_correction_bias")
    for key in list(state_dict.keys()):
        if f"model.layers.{MTP_LAYER_ID}." in key:
            new_key = key.replace(f"model.layers.{MTP_LAYER_ID}.", "mtp.mtp_layers.0.layer.", 1)
            state_dict[new_key] = state_dict.pop(key)

    # concat separate expert gemm into one grouped gemm
    expert_gate_proj_weights = {}
    expert_up_proj_weights = {}
    expert_down_proj_weights = {}
    for layer_id in range(NUM_HIDDEN_LAYERS):
        state_dict.pop(f"model.layers.{layer_id}.self_attn.rotary_emb.inv_freq", None)
        if layer_id >= FIRST_K_DENSE_REPLACE:
            layerwise_expert_gate_proj_weights = []
            layerwise_up_gate_proj_weights = []
            layerwise_down_gate_proj_weights = []
            for expert_id in range(N_ROUTED_EXPERTS):
                # Each tensor is of shape [2048, 7168]
                layerwise_expert_gate_proj_weights.append(state_dict.pop(f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"))
                layerwise_up_gate_proj_weights.append(state_dict.pop(f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"))
                layerwise_down_gate_proj_weights.append(state_dict.pop(f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"))
            # Concatenate along dimension 1 (the second dimension)
            # Need to tranpose if the weight matrix of grouped gemm is not the same as nn.Linear transposed one
            expert_gate_proj_weights[f"model.layers.{layer_id}.mlp.experts.gate_proj.weight"] = torch.cat(layerwise_expert_gate_proj_weights, dim=1).transpose(0,1)
            expert_up_proj_weights[f"model.layers.{layer_id}.mlp.experts.up_proj.weight"] = torch.cat(layerwise_up_gate_proj_weights, dim=1).transpose(0,1)
            expert_down_proj_weights[f"model.layers.{layer_id}.mlp.experts.down_proj.weight"] = torch.cat(layerwise_down_gate_proj_weights, dim=1).transpose(0,1)
            state_dict[f"model.layers.{layer_id}.mlp.e_score_correction_bias"] = state_dict.pop(f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias")
    state_dict.update(expert_gate_proj_weights)
    state_dict.update(expert_up_proj_weights)
    state_dict.update(expert_down_proj_weights)

    filtered_state_dict = remove_higher_layers(state_dict, NUM_HIDDEN_LAYERS)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(output_dir)
    DCP.save({"model": filtered_state_dict}, storage_writer=storage_writer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--input', type=Path, required=False, default=Path("/mnt/afs/jiaopenglong/DeepSeek-V3-hf-ckpt-bf16"))
    parser.add_argument('--output', type=Path, required=False, default=Path("/mnt/afs/caizheng/DeepSeek-V3-hf-ckpt-bf16-fixed"))
    args = parser.parse_args()
    convert_checkpoint_hf2dcp(
        input_dir=args.input,
        output_dir=args.output,
    )
