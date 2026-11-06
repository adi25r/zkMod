#ifndef ZKGPT_HPP
#define ZKGPT_HPP

#include "neuralNetwork.hpp"



class LLM: public neuralNetwork {

public:
    explicit LLM(int depth, int headnum=0, int headdim=0, int attn_dim=0, int linear_dim=0, int in_len=30,
                 bool output_private=false, bool input_private=false, bool input_separate=false);
};

class LLMSplitBlock: public neuralNetwork {
public:
    explicit LLMSplitBlock(int depth, int headnum=0, int headdim=0, int attn_dim=0, int linear_dim=0, int in_len=30,
                           bool output_private=false, bool input_private=false, bool input_separate=false,
                           prover *pr_prev_block=nullptr, neuralNetwork *previous_block=nullptr);
};

#endif 
