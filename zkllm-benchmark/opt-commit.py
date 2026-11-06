import os, sys, time, glob
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='OPT PPGen + Commit')
parser.add_argument('model_size', type=str, help='e.g. 125m, 350m, 1.3b, 2.7b, 6.7b, 13b')
parser.add_argument('--log_scaling_factor', type=int, default=16)
parser.add_argument('--log_off_factor', type=int, default=5)

from transformers import AutoModelForCausalLM


def save_weight_int(int_weight, path):
    int_weight.cpu().detach().numpy().astype(np.int32).tofile(path)


if __name__ == '__main__':
    args = parser.parse_args()
    if os.system('make ppgen commit-param'):
        print('Error compiling ppgen/commit-param')
        exit(1)
    model_card = f'facebook/opt-{args.model_size}'
    scaling_factor = 1 << args.log_scaling_factor
    model = AutoModelForCausalLM.from_pretrained(model_card, local_files_only=True, cache_dir='./model-storage')
    if not model.config.do_layer_norm_before:
        raise ValueError(f'opt-{args.model_size} is post-LN (do_layer_norm_before=False); '
                          'opt-layer.cu only implements pre-LN OPT. Known post-LN size: opt-350m.')
    layers = model.model.decoder.layers
    workdir = f'./zkllm-workdir/opt-{args.model_size}'
    os.makedirs(workdir, exist_ok=True)

    for name, w in layers[0].named_parameters():
        pp_size = w.shape[0] << args.log_off_factor if w.dim() == 2 else w.shape[0]
        if os.system(f'./ppgen {pp_size} {workdir}/{name}-pp.bin'):
            print('Error generating pp for', name)
            exit(1)

    commit_s = 0.0
    for i, layer in enumerate(layers):
        for name, w in layer.named_parameters():
            w_orig = w.float().T if w.dim() == 2 else w.float()
            w_out = torch.round(w_orig * scaling_factor).to(torch.int32)
            pp = f'{workdir}/{name}-pp.bin'
            int_bin = f'{workdir}/layer-{i}-{name}-int.bin'
            com = f'{workdir}/layer-{i}-{name}-commitment.bin'
            save_weight_int(w_out, int_bin)
            rows, cols = (w_out.shape[0], w_out.shape[1]) if w_out.dim() == 2 else (w_out.shape[0], 1)
            t0 = time.time()
            if os.system(f'./commit-param {pp} {int_bin} {com} {rows} {cols} > /dev/null'):
                print('Error committing', i, name)
                exit(1)
            commit_s += time.time() - t0

    com_bytes = sum(os.path.getsize(f) for f in glob.glob(f'{workdir}/layer-*-commitment.bin'))
    c = model.config
    print(f'OPT_COMMIT model=opt-{args.model_size} layers={c.num_hidden_layers} heads={c.num_attention_heads} '
          f'hidden={c.hidden_size} ffn={c.ffn_dim} commit_s={commit_s:.3f} commitment_MB={com_bytes / 1e6:.3f}')
