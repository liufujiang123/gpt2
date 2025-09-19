import numpy as np
import torch
import time
import math
import torch.nn.functional as F
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """

    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))



def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """

    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])


    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)


    x_norm = (x - mean) / torch.sqrt(var + eps)

    return g * x_norm + b # 缩放和平移

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = torch.Tensor(w_b['w']), torch.Tensor(w_b['b'])

    return x @ w + b
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']

    x = linear(x, w_b1)
    x = gelu(x)
    x = linear(x, w_b2)

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    scores = q @ k.transpose(-2, -1)

    d_k = q.size(-1)
    scaled_scores = scores / math.sqrt(d_k)

   

    if mask is not None:
        scaled_scores = scaled_scores + mask

  
    weights = softmax(scaled_scores)

    output = weights @ v
    
    return output

def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']

    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    

    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """

    q, k, v = x.chunk(3, dim=-1)
    
    qkv = [q, k, v]

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  #  [3, n_head, n_seq, n_embd/n_head]?  [n_head, 3, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    n_seq = qkv_heads[0][0].shape[0]
    mask = torch.triu(torch.ones(n_seq, n_seq), diagonal=1)
    causal_mask = mask.masked_fill(mask==1, float('-inf'))

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    

    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1)
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    x = torch.Tensor(x)
  
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling

        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        # 1. 小模型生成K个候选tokens（贪心）
        current_K = min(K, n_tokens_to_generate - len(generated_ids))
        draft_ids = current_inputs.copy()
        for _ in range(current_K):
            logits = gpt2(draft_ids, draft_params, hparams_draft["n_head"])
            next_id = np.argmax(logits[-1])
            draft_ids.append(int(next_id))
        draft_tokens = draft_ids[len(current_inputs):]  # 提取候选tokens
        
        # 2. 大模型并行验证
        verification_input = current_inputs + draft_tokens
        target_logits = gpt2(verification_input, target_params, hparams_target["n_head"])
        
        # 3. 贪心验证：逐个位置比较
        num_accepted = 0
        for i in range(len(draft_tokens)):
            # 正确的logits索引
            logit_idx = len(current_inputs) + i - 1
            target_greedy = np.argmax(target_logits[logit_idx])
            if draft_tokens[i] == target_greedy:
                num_accepted += 1
            else:
                break
        
        # 4. 接受/纠错
        if num_accepted > 0:
            generated_ids.extend(draft_tokens[:num_accepted])
            current_inputs.extend(draft_tokens[:num_accepted])
        else:
            # 完全拒绝时采用大模型的第一个token
            correct_token = int(np.argmax(target_logits[len(current_inputs)-1]))
            generated_ids.append(correct_token)
            current_inputs.append(correct_token)
    
    return generated_ids[:n_tokens_to_generate]



    

def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # 加载编码器（共享）
    encoder, _, _ = load_encoder_hparams_and_params("124M", models_dir)
    
    # 加载双模型
    _, hparams_draft, params_draft = load_encoder_hparams_and_params("124M", models_dir)
    _, hparams_target, params_target = load_encoder_hparams_and_params("1558M", models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams_target["n_ctx"]


    start_greedy = time.time()
    greedy_ids = generate(
        input_ids.copy(),
        params_target,
        n_head=hparams_target["n_head"],
        n_tokens_to_generate=n_tokens_to_generate
    )
    end_greedy = time.time()
    greedy_text = encoder.decode(greedy_ids)
    print(f"[Greedy decoding] Time: {end_greedy - start_greedy:.2f}s")
    print(f"[Greedy decoding] Output:\n{greedy_text}\n")


    # generate output ids
    start = time.time()
    
    output_ids = greedy_speculative_generate(
            input_ids, 
            params_draft, 
            params_target,
            hparams_draft,
            hparams_target,
            n_tokens_to_generate,
            K=4  # 初始K值
        )
    end = time.time()


    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)