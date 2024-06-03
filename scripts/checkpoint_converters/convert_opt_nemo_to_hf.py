# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from transformers.tokenization_utils import AddedToken
from tokenizers.models import BPE

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging

"""
Script to convert an OPT checkpoint in nemo into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_nemo_opt_to_hf.py \
    --in-file /path/to/file.nemo or /path/to/extracted_folder \
    --out-file /path/to/pytorch_model.bin

2) Generate the full HF model folder

    python convert_nemo_opt_to_hf.py \
    --in-file /path/to/file.nemo or /path/to/extracted_folder \
    --out-file /path/to/pytorch_model.bin \
    --hf-in-file /path/to/input_hf_folder \
    --hf-out-file /path/to/output_hf_folder

    Use the --cpu-only flag if the model cannot fit in the GPU.
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to .nemo file",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--hf-in-path",
        type=str,
        default=None,
        help="A HF model path, " "e.g. a folder containing https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main",
    )
    parser.add_argument(
        "--hf-out-path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--llama-tokenizer-path",
        type=str,
        default=None,
        help="Path to the llama tokenizer that is taken as a starting point in case the default OPT tokenizer is not used."
    )
    parser.add_argument(
        "--slopt-tokenizer-path",
        type=str,
        default=None,
        help="Path to the Slovene tokenizer in case the default OPT tokenizer is not used."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
             "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
             "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    if cpu_only:
        map_location = torch.device('cpu')
        model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
        model_config.use_cpu_initialization = True
        model_config.tensor_model_parallel_size = 1
    else:
        map_location, model_config = None, None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )
    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    hidden_size = model.cfg.hidden_size
    head_num = model.cfg.num_attention_heads
    num_layers = model.cfg.num_layers

    head_size = hidden_size // head_num
    qkv_total_dim = 3 * head_num

    mcore_gpt = model.cfg.mcore_gpt

    ##################
    # Word embeddings
    ##################
    if mcore_gpt:
        embed_weight = model.state_dict()['model.embedding.word_embeddings.weight']
    else:
        embed_weight = model.state_dict()['model.language_model.embedding.word_embeddings.weight']

    embed_weights_base_name = 'model.decoder.embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    ######################
    # Position embeddings
    ######################
    if mcore_gpt:
        position_embed_weight = model.state_dict()['model.embedding.position_embeddings.weight']
    else:
        position_embed_weight = model.state_dict()['model.language_model.embedding.position_embeddings.weight']

    position_embed_weights_base_name = 'model.decoder.embed_positions.weight'
    checkpoint[position_embed_weights_base_name] = param_to_weights(position_embed_weight)

    #####################
    # Transformer blocks
    #####################
    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        ########################################
        # Layer norm before the self-attention
        ########################################
        if mcore_gpt:
            input_ln_weight = model.state_dict()[
                f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
            input_ln_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_bias']
        else:
            input_ln_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.input_layernorm.weight']
            input_ln_bias = model.state_dict()[f'model.language_model.encoder.layers.{l}.input_layernorm.bias']

        input_ln_weight_base_name = f'model.decoder.layers.{l}.self_attn_layer_norm.weight'
        checkpoint[input_ln_weight_base_name] = param_to_weights(input_ln_weight)

        input_ln_bias_base_name = f'model.decoder.layers.{l}.self_attn_layer_norm.bias'
        checkpoint[input_ln_bias_base_name] = param_to_weights(input_ln_bias)

        ##################
        # Self-Attention
        ##################
        if mcore_gpt:
            qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
            qkv_biases = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.bias']
        else:
            qkv_weights = model.state_dict()[
                f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight']
            qkv_biases = model.state_dict()[
                f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.bias']

        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])
        qkv_biases = qkv_biases.reshape([qkv_total_dim, head_size])
        q_slice = torch.cat(
            [
                torch.arange(3 * i, 3 * i + 1)
                for i in range(head_num)
            ]
        )
        k_slice = torch.arange(1, qkv_total_dim, 3)
        v_slice = torch.arange(2, qkv_total_dim, 3)

        q_weight_base_name = f'model.decoder.layers.{l}.self_attn.q_proj.weight'
        k_weight_base_name = f'model.decoder.layers.{l}.self_attn.k_proj.weight'
        v_weight_base_name = f'model.decoder.layers.{l}.self_attn.v_proj.weight'

        checkpoint[q_weight_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weight_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weight_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        q_bias_base_name = f'model.decoder.layers.{l}.self_attn.q_proj.bias'
        k_bias_base_name = f'model.decoder.layers.{l}.self_attn.k_proj.bias'
        v_bias_base_name = f'model.decoder.layers.{l}.self_attn.v_proj.bias'

        checkpoint[q_bias_base_name] = param_to_weights(qkv_biases[q_slice].reshape(hidden_size, ))
        checkpoint[k_bias_base_name] = param_to_weights(qkv_biases[k_slice].reshape(hidden_size, ))
        checkpoint[v_bias_base_name] = param_to_weights(qkv_biases[v_slice].reshape(hidden_size, ))

        ##########################################
        # Attention dense layer (out projection)
        ##########################################
        if mcore_gpt:
            o_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj.weight']
            o_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj.bias']
        else:
            o_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.self_attention.dense.weight']
            o_bias = model.state_dict()[f'model.language_model.encoder.layers.{l}.self_attention.dense.bias']

        o_weight_base_name = f'model.decoder.layers.{l}.self_attn.out_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        o_bias_base_name = f'model.decoder.layers.{l}.self_attn.out_proj.bias'
        checkpoint[o_bias_base_name] = param_to_weights(o_bias)

        ########################
        # Layer norm before FC1
        ########################
        if mcore_gpt:
            post_attn_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']
            post_attn_ln_bias = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_bias']
        else:
            post_attn_ln_weight = model.state_dict()[
                f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight']
            post_attn_ln_bias = model.state_dict()[
                f'model.language_model.encoder.layers.{l}.post_attention_layernorm.bias']

        post_attn_ln_weight_base_name = f'model.decoder.layers.{l}.final_layer_norm.weight'
        checkpoint[post_attn_ln_weight_base_name] = param_to_weights(post_attn_ln_weight)

        post_attn_ln_bias_base_name = f'model.decoder.layers.{l}.final_layer_norm.bias'
        checkpoint[post_attn_ln_bias_base_name] = param_to_weights(post_attn_ln_bias)

        #######
        # FC1
        #######
        if mcore_gpt:
            fc1_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
            fc1_bias = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.bias']
        else:
            fc1_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight']
            fc1_bias = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.bias']

        fc1_weight_base_name = f'model.decoder.layers.{l}.fc1.weight'
        checkpoint[fc1_weight_base_name] = param_to_weights(fc1_weight)

        fc1_bias_base_name = f'model.decoder.layers.{l}.fc1.bias'
        checkpoint[fc1_bias_base_name] = param_to_weights(fc1_bias)

        #######
        # FC2
        #######
        if mcore_gpt:
            fc2_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2.weight']
            fc2_bias = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2.bias']
        else:
            fc2_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight']
            fc2_bias = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.bias']

        fc2_weight_base_name = f'model.decoder.layers.{l}.fc2.weight'
        checkpoint[fc2_weight_base_name] = param_to_weights(fc2_weight)

        fc2_bias_base_name = f'model.decoder.layers.{l}.fc2.bias'
        checkpoint[fc2_bias_base_name] = param_to_weights(fc2_bias)

        print(f"done layer {l}")

    ###################
    # Final layer norm
    ###################
    if mcore_gpt:
        final_ln_weight = model.state_dict()['model.decoder.final_layernorm.weight']
        final_ln_bias = model.state_dict()['model.decoder.final_layernorm.bias']
    else:
        final_ln_weight = model.state_dict()['model.language_model.encoder.final_layernorm.weight']
        final_ln_bias = model.state_dict()['model.language_model.encoder.final_layernorm.bias']

    final_ln_weight_base_name = 'model.decoder.final_layer_norm.weight'
    checkpoint[final_ln_weight_base_name] = param_to_weights(final_ln_weight)

    final_ln_bias_base_name = 'model.decoder.final_layer_norm.bias'
    checkpoint[final_ln_bias_base_name] = param_to_weights(final_ln_bias)

    ###############
    # Output layer
    ###############
    if mcore_gpt:
        output_layer_weight = model.state_dict()['model.output_layer.weight']
    else:
        output_layer_weight = model.state_dict()['model.language_model.output_layer.weight']

    output_layer_base_name = f'lm_head.weight'
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    ###################
    # Save the weights
    ###################
    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights saved to {output_hf_file}")


def replace_hf_weights_and_tokenizer(weights_file, input_hf_path, output_hf_path, llama_tokenizer_path, slopt_tokenizer_path):
    nemo_exported = torch.load(weights_file, map_location=torch.device('cpu'))

    # Convert the tokenizer
    if llama_tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(input_hf_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path)

    # In case of changed tokenizer, the backend model and special tokens have to be changed
    if slopt_tokenizer_path is not None:
        bpe_tokenizer = BPE.from_file(vocab=os.path.join(slopt_tokenizer_path, "vocab.json"),
                                      merges=os.path.join(slopt_tokenizer_path, "merges.txt"))
        tokenizer.backend_tokenizer.model = bpe_tokenizer

        def update_existing_special_token(content, token):
            new_token = AddedToken(
                content,
                lstrip=token.lstrip,
                normalized=token.normalized,
                rstrip=token.rstrip,
                single_word=token.single_word
            )

            return new_token

        def create_new_special_token(content):
            new_token = AddedToken(
                content,
                lstrip=False,
                normalized=True,
                rstrip=False,
                single_word=False
            )

            return new_token

        special_tokens = {
            "bos_token": update_existing_special_token("<s>", tokenizer._bos_token),
            "eos_token": update_existing_special_token("</s>", tokenizer._eos_token),
            "unk_token": update_existing_special_token("<unk>", tokenizer._unk_token),
            "pad_token": create_new_special_token("<pad>"),
            "sep_token": create_new_special_token("<sep>"),
            "cls_token": create_new_special_token("<cls>"),
            "mask_token": create_new_special_token("<mask>")
        }

        tokenizer.add_special_tokens(special_tokens)

        # Fix the tokens in init kwargs
        tokenizer.init_kwargs["unk_token"] = special_tokens["unk_token"]
        tokenizer.init_kwargs["bos_token"] = special_tokens["bos_token"]
        tokenizer.init_kwargs["eos_token"] = special_tokens["eos_token"]
        tokenizer.init_kwargs["pad_token"] = special_tokens["pad_token"]

        # Fix the path in init kwargs
        tokenizer.init_kwargs["name_or_path"] = output_hf_path

    tokenizer.add_bos_token = False
    tokenizer.init_kwargs["add_bos_token"] = False
    tokenizer.save_pretrained(output_hf_path)

    # Convert the model
    model = AutoModelForCausalLM.from_pretrained(input_hf_path, local_files_only=True)

    # Model's vocabulary size and token IDs need to be updated if tokenizer was changed
    if slopt_tokenizer_path is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.sep_token_id = tokenizer.sep_token_id
        model.config.vocab_size = tokenizer.vocab_size

        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Model needs to be reloaded with tie word embeddings set to false, otherwise word embeddings and output layer are
    # the same parameter
    model.config.tie_word_embeddings = False
    model.config.prefix = None
    model = OPTForCausalLM(model.config)
    model.load_state_dict(nemo_exported)

    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")


if __name__ == '__main__':
    args = get_args()
    convert(args.in_file, args.out_file, precision=args.precision, cpu_only=args.cpu_only)
    if args.hf_in_path and args.hf_out_path:
        replace_hf_weights_and_tokenizer(args.out_file, args.hf_in_path, args.hf_out_path, args.llama_tokenizer_path,
                           args.slopt_tokenizer_path)
    else:
        logging.info("`hf-in-path` and/or `hf-out-path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.out_file}")