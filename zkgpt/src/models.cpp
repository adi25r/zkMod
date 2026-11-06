//
// Created by 69029 on 3/16/2021.
//

#include <tuple>
#include <iostream>
#include "models.hpp"
#include "utils.hpp"
#undef USE_VIRGO




LLM::LLM(int depth, int headnum, int headdim, int attn_dim, int linear_dim, int in_len,
         bool output_private, bool input_private, bool input_separate):
    neuralNetwork(0,0,0,0, "./data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv", "", "",
                  true, headnum, headdim, attn_dim, linear_dim, output_private, input_private, input_separate)
{
    len=in_len;
    pic_parallel=1;
    conv_section.clear();
    positive_check=0;
    exp_check=0;
    layer_num=depth;
    for(int i=0;i<depth;i++)
    {
        // Fused QKV: d_model -> 3*d_model
        full_conn.emplace_back(3 * attn_dim, attn_dim); // W_qkv  (produces [Q;K;V])

        // Attention output: d_model -> d_model
        full_conn.emplace_back(attn_dim, attn_dim);     // W_o

        // MLP up/down: d_model -> d_ff -> d_model
        full_conn.emplace_back(linear_dim, attn_dim);   // W_up
        full_conn.emplace_back(attn_dim, linear_dim);   // W_down
 
    }
    
}

LLMSplitBlock::LLMSplitBlock(int depth, int headnum, int headdim, int attn_dim, int linear_dim, int in_len,
                             bool output_private, bool input_private, bool input_separate,
                             prover *pr_prev_block, neuralNetwork *previous_block):
    neuralNetwork(0,0,0,0, "./data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv", "", "",
                  true, headnum, headdim, attn_dim, linear_dim, output_private, input_private, input_separate)
{
    len = in_len;
    pic_parallel = 1;
    conv_section.clear();
    positive_check = 0;
    exp_check = 0;
    layer_num = depth;
    for(int i = 0; i < depth; i++)
    {
        full_conn.emplace_back(3 * attn_dim, attn_dim);
        full_conn.emplace_back(attn_dim, attn_dim);
        full_conn.emplace_back(linear_dim, attn_dim);
        full_conn.emplace_back(attn_dim, linear_dim);
    }
    if (pr_prev_block && previous_block) {
        initStateFromPreviousBlock(*pr_prev_block, *previous_block);
    }
}
