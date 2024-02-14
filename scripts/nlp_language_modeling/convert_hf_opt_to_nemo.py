import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.trainer.trainer import Trainer
from transformers import OPTForCausalLM, AutoTokenizer
from sentencepiece import SentencePieceProcessor

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging

DEFAULT_NEMO_PATH = "/workspace/nemo/"


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-file", type=str, default=None, required=True, help="Path to Huggingface OPT checkpoints",
    )
    parser.add_argument("--out-file", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument("--nemo-path", type=str, default=DEFAULT_NEMO_PATH, help="Path to the folder containing nemo scripts")
    parser.add_argument("--tokenizer-path", type=str, default="", help="Path to the .model file of sentencepiece tokenizer")
    args = parser.parse_args()
    return args


def load_model(cls, checkpoint, strict, **kwargs):
    try:
        if 'cfg' in kwargs:
            model = ptl_load_state(cls, checkpoint, strict=strict, **kwargs)
        else:
            model = cls(cfg=checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY], **kwargs)
            for name, module in model.named_parameters():
                if name in checkpoint['state_dict']:
                    module.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)
                else:
                    print(f"Unexpected key: {name} not in checkpoint but in model.")

            for name, buffer in model.named_buffers():
                if name in checkpoint['state_dict']:
                    buffer.data = checkpoint['state_dict'][name]
                    checkpoint['state_dict'].pop(name)

            if len(checkpoint['state_dict'].keys()) != 0:
                raise RuntimeError(
                    f"Additional keys: {checkpoint['state_dict'].keys()} in checkpoint but not in model."
                )

    finally:
        cls._set_model_restore_state(is_being_restored=False)
    return model


def load_config(opt_config, args):
    nemo_config = OmegaConf.load(
        os.path.join(args.nemo_path, 'examples/nlp/language_modeling/conf/megatron_gpt_config.yaml')
    ).model
    nemo_config.activation = 'relu'
    nemo_config.bias_activation_fusion = False
    nemo_config.bias_dropout_add_fusion = False
    nemo_config.encoder_seq_length = opt_config['max_position_embeddings']
    # OPT uses an offset of length 2
    nemo_config.data.position_offset = 2
    nemo_config.max_position_embeddings = nemo_config.encoder_seq_length + 2
    nemo_config.num_layers = int(opt_config['num_hidden_layers'])
    nemo_config.hidden_size = opt_config['word_embed_proj_dim']
    nemo_config.ffn_hidden_size = opt_config['ffn_dim']
    nemo_config.num_attention_heads = opt_config['num_attention_heads']
    nemo_config.init_method_std = opt_config['init_std']
    nemo_config.hidden_dropout = opt_config['dropout']
    nemo_config.ffn_dropout = opt_config['dropout']
    nemo_config.attention_dropout = opt_config['attention_dropout']
    nemo_config.share_embeddings_and_output_weights = False
    if args.tokenizer_path != "":
        # Reset the vocab size
        sp_tokenizer = SentencePieceProcessor(model_file=args.tokenizer_path)
        opt_config["vocab_size"] = sp_tokenizer.vocab_size()

        tokenizer_dict = {
            "library": "sentencepiece",
            "model": args.tokenizer_path,
            "type": None,
            "vocab_file": None,
            "merge_file": None,
            "delimiter": None,
            "sentencepiece_legacy": False
        }
    else:
        tokenizer_dict = {
            "library": "huggingface",
            "type": args.in_file,
            "use_fast": True
        }
    nemo_config.tokenizer = OmegaConf.create(tokenizer_dict)
    nemo_config.use_cpu_initialization = True

    base = 128
    while opt_config['vocab_size'] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base

    return nemo_config


def convert(args):
    logging.info(f"loading checkpoint {args.in_file}")
    model = OPTForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    print(f"hf_config: {hf_config}")
    print("State dict:")
    for key in model.state_dict().keys():
        print(f"- {key}")

    nemo_config = load_config(hf_config, args)

    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=nemo_config.get('native_amp_init_scale', 2 ** 32),
                growth_interval=nemo_config.get('native_amp_growth_interval', 1000),
                hysteresis=nemo_config.get('hysteresis', 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if nemo_config.get('megatron_amp_O2', False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    nemo_config.precision = precision
    print(f"nemo_config: {nemo_config}")

    trainer = Trainer(plugins=plugins, accelerator='cpu', precision=precision, strategy=NLPDDPStrategy())

    num_layers = hf_config["num_hidden_layers"]
    num_heads = hf_config["num_attention_heads"]
    hidden_size = hf_config["hidden_size"]
    head_size = hidden_size // num_heads

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get(
        'transformer_engine', False
    ), "mcore_gpt transformer_engine must be enabled (or disabled) together."

    param_to_weights = lambda param: param.float()

    checkpoint = OrderedDict()
    checkpoint['state_dict'] = OrderedDict()

    ##################
    # Word embeddings
    ##################
    if mcore_gpt:
        embed_weights_base_name = 'model.embedding.word_embeddings.weight'
    else:
        embed_weights_base_name = 'model.language_model.embedding.word_embeddings.weight'

    # If tokenizer was changed, the word embeddings are randomly initialized
    if args.tokenizer_path == "":
        embed_weight = model.state_dict()['model.decoder.embed_tokens.weight']
    else:
        embed_weight = torch.empty((hf_config["vocab_size"], nemo_config.hidden_size))
        torch.nn.init.normal_(embed_weight, mean=0.0, std=nemo_config.init_method_std)

    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(embed_weight)


    ######################
    # Position embeddings
    ######################
    position_embed = model.state_dict()['model.decoder.embed_positions.weight']
    if mcore_gpt:
        embed_weights_base_name = 'model.embedding.position_embeddings.weight'
    else:
        embed_weights_base_name = 'model.language_model.embedding.position_embeddings.weight'
    checkpoint['state_dict'][embed_weights_base_name] = param_to_weights(position_embed)

    #####################
    # Transformer blocks
    #####################
    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        ########################################
        # Layer norm before the self-attention
        ########################################
        input_norm_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attn_layer_norm.weight']
        input_norm_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attn_layer_norm.bias']

        if mcore_gpt:
            input_ln_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight'
            input_ln_bias_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_bias'
        else:
            input_ln_weight_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.weight'
            input_ln_bias_base_name = f'model.language_model.encoder.layers.{l}.input_layernorm.bias'

        checkpoint['state_dict'][input_ln_weight_base_name] = param_to_weights(input_norm_weight)
        checkpoint['state_dict'][input_ln_bias_base_name] = param_to_weights(input_norm_bias)

        ##################
        # Self-Attention
        ##################
        q = model.state_dict()[f'model.decoder.layers.{l}.self_attn.q_proj.weight']
        q_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attn.q_proj.bias']
        k = model.state_dict()[f'model.decoder.layers.{l}.self_attn.k_proj.weight']
        k_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attn.k_proj.bias']
        v = model.state_dict()[f'model.decoder.layers.{l}.self_attn.v_proj.weight']
        v_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attn.v_proj.bias']

        # Weights have to be permuted as nemo stores q, k, v projection weights in a single matrix
        old_shape = q.size()
        new_shape = (num_heads, head_size) + old_shape[1:]
        new_bias_shape = (num_heads, head_size)

        q = q.view(*new_shape)
        q_bias = q_bias.view(*new_bias_shape)
        k = k.view(*new_shape)
        k_bias = k_bias.view(*new_bias_shape)
        v = v.view(*new_shape)
        v_bias = v_bias.view(*new_bias_shape)

        qkv_weights = torch.empty(tuple([0]) + new_shape[1:])
        qkv_biases = torch.empty((0, head_size))

        for i in range(num_heads):
            qkv_weights = torch.cat((qkv_weights, q[i: i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i: i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i: i + 1, :, :]))

            qkv_biases = torch.cat((qkv_biases, q_bias[i: i + 1, :]))
            qkv_biases = torch.cat((qkv_biases, k_bias[i: i + 1, :]))
            qkv_biases = torch.cat((qkv_biases, v_bias[i: i + 1, :]))

        qkv_weights = qkv_weights.reshape([3 * head_size * num_heads, hidden_size])
        qkv_biases = qkv_biases.reshape((-1,))

        if mcore_gpt:
            qkv_weights_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.weight'
            qkv_biases_base_name = f'model.decoder.layers.{l}.self_attention.linear_qkv.bias'
        else:
            qkv_weights_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight'
            qkv_biases_base_name = f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.bias'

        checkpoint['state_dict'][qkv_weights_base_name] = param_to_weights(qkv_weights)
        checkpoint['state_dict'][qkv_biases_base_name] = param_to_weights(qkv_biases)

        ##########################################
        # Attention dense layer (out projection)
        ##########################################
        out_proj_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attn.out_proj.weight']
        out_proj_bias = model.state_dict()[f'model.decoder.layers.{l}.self_attn.out_proj.bias']

        if mcore_gpt:
            output_weight_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.weight'
            output_bias_base_name = f'model.decoder.layers.{l}.self_attention.linear_proj.bias'
        else:
            output_weight_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.weight'
            output_bias_base_name = f'model.language_model.encoder.layers.{l}.self_attention.dense.bias'

        checkpoint['state_dict'][output_weight_base_name] = param_to_weights(out_proj_weight)
        checkpoint['state_dict'][output_bias_base_name] = param_to_weights(out_proj_bias)

        ########################
        # Layer norm before FC1
        ########################
        fc1_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.final_layer_norm.weight']
        fc1_ln_bias = model.state_dict()[f'model.decoder.layers.{l}.final_layer_norm.bias']

        if mcore_gpt:
            post_attn_ln_weight_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight'
            post_attn_ln_bias_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_bias'
        else:
            post_attn_ln_weight_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight'
            post_attn_ln_bias_base_name = f'model.language_model.encoder.layers.{l}.post_attention_layernorm.bias'

        checkpoint['state_dict'][post_attn_ln_weight_base_name] = param_to_weights(fc1_ln_weight)
        checkpoint['state_dict'][post_attn_ln_bias_base_name] = param_to_weights(fc1_ln_bias)

        #######
        # FC1
        #######
        fc1_weight = model.state_dict()[f'model.decoder.layers.{l}.fc1.weight']
        fc1_bias = model.state_dict()[f'model.decoder.layers.{l}.fc1.bias']
        if mcore_gpt:
            mlp_fc1_weight_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.weight'
            mlp_fc1_bias_base_name = f'model.decoder.layers.{l}.mlp.linear_fc1.bias'
        else:
            mlp_fc1_weight_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight'
            mlp_fc1_bias_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.bias'

        checkpoint['state_dict'][mlp_fc1_weight_base_name] = param_to_weights(fc1_weight)
        checkpoint['state_dict'][mlp_fc1_bias_base_name] = param_to_weights(fc1_bias)

        #######
        # FC2
        #######
        fc2_weight = model.state_dict()[f'model.decoder.layers.{l}.fc2.weight']
        fc2_bias = model.state_dict()[f'model.decoder.layers.{l}.fc2.bias']
        if mcore_gpt:
            mlp_fc2_weight_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.weight'
            mlp_fc2_bias_base_name = f'model.decoder.layers.{l}.mlp.linear_fc2.bias'
        else:
            mlp_fc2_weight_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight'
            mlp_fc2_bias_base_name = f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.bias'

        checkpoint['state_dict'][mlp_fc2_weight_base_name] = param_to_weights(fc2_weight)
        checkpoint['state_dict'][mlp_fc2_bias_base_name] = param_to_weights(fc2_bias)

        print(f"done layer {l}")

    ###################
    # Final layer norm
    ###################
    final_ln_weight = model.state_dict()[f'model.decoder.final_layer_norm.weight']
    final_ln_bias = model.state_dict()[f'model.decoder.final_layer_norm.bias']

    if mcore_gpt:
        final_ln_weight_base_name = f'model.decoder.final_layernorm.weight'
        final_ln_bias_base_name = f'model.decoder.final_layernorm.bias'
    else:
        final_ln_weight_base_name = f'model.language_model.encoder.final_layernorm.weight'
        final_ln_bias_base_name = f'model.language_model.encoder.final_layernorm.bias'
    checkpoint['state_dict'][final_ln_weight_base_name] = param_to_weights(final_ln_weight)
    checkpoint['state_dict'][final_ln_bias_base_name] = param_to_weights(final_ln_bias)

    ###############
    # Output layer
    ###############
    if mcore_gpt:
        output_layer_base_name = f'model.output_layer.weight'
    else:
        output_layer_base_name = f'model.language_model.output_layer.weight'

    # If vocabulary was changed, output layer is randomly initialized
    if args.tokenizer_path == "":
        output_layer_weight = model.state_dict()[f'lm_head.weight']
    else:
        output_layer_weight = torch.empty((hf_config["vocab_size"], nemo_config.hidden_size))
        torch.nn.init.normal_(output_layer_weight, mean=0.0, std=nemo_config.init_method_std)

    checkpoint['state_dict'][output_layer_base_name] = param_to_weights(output_layer_weight)


    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config

    del model

    if nemo_config.get('megatron_amp_O2', False):
        keys = list(checkpoint['state_dict'].keys())
        for key in keys:
            checkpoint['state_dict'][key.replace('model.', 'model.module.', 1)] = checkpoint['state_dict'].pop(key)

    model = load_model(MegatronGPTModel, checkpoint, strict=False, trainer=trainer)

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    dtype = torch_dtype_from_precision(precision)
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False

    model.save_to(args.out_file)
    logging.info(f'NeMo model saved to: {args.out_file}')


if __name__ == "__main__":
    args = get_args()
    convert(args)
