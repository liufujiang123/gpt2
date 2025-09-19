import numpy as np
import torch
import time
import math
import torch.nn.functional as F
torch.set_printoptions(8)


# 用于在生成步骤之间传递 K,V 状态
KV_CACHE = None
# 用于在单次 gpt2 调用中构建新缓存的临时变量
NEW_KV_CACHE = None

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

def mha(x, attn, n_head,layer_idx):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """

    global KV_CACHE, NEW_KV_CACHE

    c_attn, c_proj = attn['c_attn'], attn['c_proj']

    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]  

    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """

    q, k, v = x.chunk(3, dim=-1)

    # 如果全局缓存存在，说明是生成阶段
    if KV_CACHE is not None:
        # 1. 从全局缓存中读取对应层的 past_k 和 past_v
        past_k, past_v = KV_CACHE[layer_idx]['k'], KV_CACHE[layer_idx]['v']
        # 2. 拼接
        k = torch.cat((past_k, k), dim=0)
        v = torch.cat((past_v, v), dim=0)

    NEW_KV_CACHE.append({'k': k, 'v': v})

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
    n_seq_q, n_seq_k = q.shape[0], k.shape[0]
    mask = torch.triu(torch.ones(n_seq_q, n_seq_k), diagonal=1 + n_seq_k - n_seq_q)
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


def transformer_block(x, block, n_head, layer_idx):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, layer_idx=layer_idx)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]

    global KV_CACHE, NEW_KV_CACHE

    # 为本次前向传播初始化临时的 NEW_KV_CACHE
    NEW_KV_CACHE = []

    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings

    if KV_CACHE is not None: # 选择输入的范围
        past_length = KV_CACHE[0]['k'].shape[0]
        x = wte[inputs[-1:]] + wpe[past_length : past_length + 1]
    else:
        x = wte[inputs] + wpe[range(len(inputs))]
    # x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    x = torch.Tensor(x)
  
    for i, block in enumerate(blocks):
        # 传递 layer_idx，以便 transformer_block 和 mha 能访问正确的缓存
        x = transformer_block(x, block, n_head=n_head, layer_idx=i)

     # 更新cache
    KV_CACHE = NEW_KV_CACHE

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):

    global KV_CACHE
    # 在每次调用 generate 时，必须重置全局缓存，以防上一次运行的干扰
    KV_CACHE = None
    
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
        pass

    return generated_ids


    

def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)