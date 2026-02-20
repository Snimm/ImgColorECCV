import torch
from safetensors.torch import load_file, save_file
from musubi_tuner.flux_2.flux2_models import AutoEncoder, AutoEncoderParams

def convert_key(k):
    k = k.replace("group_norm", "norm")
    
    # Attention matches
    # to_q -> q, to_k -> k, to_v -> v
    k = k.replace("to_q", "q")
    k = k.replace("to_k", "k")
    k = k.replace("to_v", "v")
    k = k.replace("to_out.0", "proj_out")

    # Resnet matches
    k = k.replace("conv_shortcut", "nin_shortcut")

    # Top level BN
    if k.startswith("bn."):
        return k
    
    # Quant convs (moved inside encoder/decoder in Musubi)
    if k.startswith("quant_conv."):
        return "encoder." + k
    if k.startswith("post_quant_conv."):
        return "decoder." + k
    
    # Encoder
    if k.startswith("encoder."):
        # conv_norm_out -> norm_out
        k = k.replace("encoder.conv_norm_out.", "encoder.norm_out.")
        # mid_block -> mid
        k = k.replace("encoder.mid_block.resnets.0.", "encoder.mid.block_1.")
        k = k.replace("encoder.mid_block.attentions.0.", "encoder.mid.attn_1.")
        k = k.replace("encoder.mid_block.resnets.1.", "encoder.mid.block_2.")
        
        # down_blocks -> down
        # format: encoder.down_blocks.0.resnets.0. ...
        parts = k.split(".")
        if "down_blocks" in parts:
            idx = int(parts[2])
            # down_blocks.0 -> down.0
            # resnets.0 -> block.0
            new_parts = ["encoder", "down", str(idx)]
            
            rest = parts[3:]
            if rest[0] == "resnets":
                new_parts.append("block")
                new_parts.append(rest[1])
                new_parts.extend(rest[2:])
            elif rest[0] == "attentions":
                new_parts.append("attn")
                new_parts.append(rest[1])
                new_parts.extend(rest[2:])
            elif rest[0] == "downsamplers":
                new_parts.append("downsample")
                # downsamplers.0.conv -> downsample.conv (musubi wrapper has .conv)
                # Musubi `Downsample` has `self.conv`
                # Diffusers `Downsample2D` has `conv`
                new_parts.extend(rest[2:])
            return ".".join(new_parts)

        return k

    # Decoder
    if k.startswith("decoder."):
        # conv_norm_out -> norm_out
        k = k.replace("decoder.conv_norm_out.", "decoder.norm_out.")
        # mid_block -> mid
        k = k.replace("decoder.mid_block.resnets.0.", "decoder.mid.block_1.")
        k = k.replace("decoder.mid_block.attentions.0.", "decoder.mid.attn_1.")
        k = k.replace("decoder.mid_block.resnets.1.", "decoder.mid.block_2.")
        
        # up_blocks -> up
        parts = k.split(".")
        if "up_blocks" in parts:
            # decoder.up_blocks.0. ...
            # Musubi Decoder has `self.up` list.
            # Musubi Up block has `block`, `attn`, `upsample`.
            idx = int(parts[2])
            # Reverse index: 0 -> 3, 1 -> 2, 2 -> 1, 3 -> 0
            new_idx = 3 - idx
            new_parts = ["decoder", "up", str(new_idx)]
            
            rest = parts[3:]
            if rest[0] == "resnets":
                new_parts.append("block")
                new_parts.append(rest[1])
                new_parts.extend(rest[2:])
            elif rest[0] == "attentions":
                new_parts.append("attn")
                new_parts.append(rest[1])
                new_parts.extend(rest[2:])
            elif rest[0] == "upsamplers":
                new_parts.append("upsample")
                new_parts.extend(rest[2:])
            return ".".join(new_parts)
            
        return k

    return k

diffusers_path = "/data/swarnim/ImgColorECCV/models/flux_klein_4b/vae/diffusion_pytorch_model.safetensors"
sd = load_file(diffusers_path)

new_sd = {}
for k, v in sd.items():
    new_k = convert_key(k)
    new_sd[new_k] = v

    # Reshape Attention weights (Linear -> Conv2d 1x1)
    # Target keys: *.q.weight, *.k.weight, *.v.weight, *.proj_out.weight
    if (new_k.endswith(".q.weight") or new_k.endswith(".k.weight") or 
        new_k.endswith(".v.weight") or new_k.endswith(".proj_out.weight")):
        if len(v.shape) == 2:
            new_sd[new_k] = v.unsqueeze(-1).unsqueeze(-1)

# Verify against Musubi model
params = AutoEncoderParams()
ae = AutoEncoder(params)
target_keys = set(ae.state_dict().keys())
converted_keys = set(new_sd.keys())

missing = target_keys - converted_keys
unexpected = converted_keys - target_keys

print(f"Total target keys: {len(target_keys)}")
print(f"Total converted keys: {len(converted_keys)}")
print(f"Missing keys: {len(missing)}")
if len(missing) > 0:
    print(f"Example missing: {list(missing)[:5]}")
print(f"Unexpected keys: {len(unexpected)}")
if len(unexpected) > 0:
    print(f"Example unexpected: {list(unexpected)[:5]}")

if len(missing) == 0 and len(unexpected) == 0:
    print("Conversion looks perfect! Saving...")
    save_file(new_sd, "/data/swarnim/ImgColorECCV/models/flux_klein_4b/ae.safetensors")
    print("Saved to models/flux_klein_4b/ae.safetensors")
else:
    print("Conversion incomplete. Not saving.")
