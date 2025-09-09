import os
import json
import numpy as np
import tensorflow as tf
import torch
from gpt_from_scratch import GPT2

"""
Loads Pretrained OpenAI weights onto
the custom model created

"""


class GPT2WeightLoader:
    def __init__(self, model_dir, config, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.settings = json.load(open(os.path.join(model_dir, "hparams.json")))
        self.tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        self.params = self._load_gpt2_params_from_tf_ckpt(
            self.tf_ckpt_path, self.settings
        )
        self.gpt = GPT2(config).to(device)

    def _assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(torch.tensor(right, dtype=torch.float32))

    def _load_gpt2_params_from_tf_ckpt(self, ckpt_path, settings):
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}
        for name, _ in tf.train.list_variables(ckpt_path):
            arr = np.squeeze(tf.train.load_variable(ckpt_path, name))
            name_parts = name.split("/")[1:]
            ptr = params
            if name_parts[0].startswith("h"):
                ptr = params["blocks"][int(name_parts[0][1:])]
            for k in name_parts[1:-1]:
                ptr = ptr.setdefault(k, {})
            ptr[name_parts[-1]] = arr
        return params

    def load_weights(self):
        gpt = self.gpt
        params = self.params

        gpt.pos_emb.weight = self._assign(gpt.pos_emb.weight, params["wpe"])
        gpt.tok_emb.weight = self._assign(gpt.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):
            layer = gpt.transformer_layers[b]
            block_params = params["blocks"][b]

            # Attention: split into Q, K, V
            q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
            layer.attn.w_Q.weight = self._assign(layer.attn.w_Q.weight, q_w.T)
            layer.attn.w_K.weight = self._assign(layer.attn.w_K.weight, k_w.T)
            layer.attn.w_V.weight = self._assign(layer.attn.w_V.weight, v_w.T)

            q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
            layer.attn.w_Q.bias = self._assign(layer.attn.w_Q.bias, q_b)
            layer.attn.w_K.bias = self._assign(layer.attn.w_K.bias, k_b)
            layer.attn.w_V.bias = self._assign(layer.attn.w_V.bias, v_b)

            layer.attn.out_proj.weight = self._assign(
                layer.attn.out_proj.weight, block_params["attn"]["c_proj"]["w"].T
            )
            layer.attn.out_proj.bias = self._assign(
                layer.attn.out_proj.bias, block_params["attn"]["c_proj"]["b"]
            )

            # Feed-forward
            layer.ff.layers[0].weight = self._assign(
                layer.ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T
            )
            layer.ff.layers[0].bias = self._assign(
                layer.ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"]
            )

            layer.ff.layers[2].weight = self._assign(
                layer.ff.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T
            )
            layer.ff.layers[2].bias = self._assign(
                layer.ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"]
            )

            # LayerNorms
            layer.norm1.scale = self._assign(
                layer.norm1.scale, block_params["ln_1"]["g"]
            )
            layer.norm1.shift = self._assign(
                layer.norm1.shift, block_params["ln_1"]["b"]
            )
            layer.norm2.scale = self._assign(
                layer.norm2.scale, block_params["ln_2"]["g"]
            )
            layer.norm2.shift = self._assign(
                layer.norm2.shift, block_params["ln_2"]["b"]
            )

        # Final normalization + LM head
        gpt.final_norm.scale = self._assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = self._assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = self._assign(gpt.out_head.weight, params["wte"])

        return gpt

    def get_model(self):
        return self.gpt
