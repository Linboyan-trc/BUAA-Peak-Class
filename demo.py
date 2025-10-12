import transformers
import torch
import time
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_model import llama_attn_forward_StreamingLLM
from llama_model import llama_sdpa_attn_forward_StreamingLLM
from llama_model import prepare_inputs_for_generation_llama, prepare_inputs_for_generation_llama_new


####################################################################################################
# 1. 加载模型
def load_model(model_name, attn_impl="sdpa", device="cuda:1"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={ "": device },
        attn_implementation=attn_impl,
    )
    return model

# 2. 运行推理
def run_and_log(name, model, tokenizer, prompt, device):
    # 2.1 本次运行任务命名
    print(f"\n=== {name} ===")
    result = benchmark(model, tokenizer, prompt, device=device)

    # 2.2 获取结果
    print(f"输入长度: {result['input_length']} tokens")
    print(f"输出长度: {result['output_length']} tokens")
    print(f"耗时: {result['elapsed']:.2f}s")
    print(f"吞吐量: {result['throughput']:.2f} tokens/s")
    print(f"输出:\n{result['decoded_output']}\n")
    return result

# 3. 运行推理
def benchmark(model, tokenizer, prompt, max_new_tokens=128, device="cuda:1"):
    # 3.1 句子转化为token在字典中的索引
    # 3.1 就是把一句话，转换成一个[1, 128]的张量（这个形状只是一个举例），每个元素的值是其token在字典中的id，然后同时还包含了掩码形状也是[1,128]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    # 3.2 推理一点点，然后丢掉，也不计入任何结果，再开始真正的推理
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)
    torch.cuda.synchronize()

    # 3.3 根据分词器对句子的分词结果，和最大输出长度限制，生成输出
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end_time = time.time()

    # 3.4 得到输出token id序列
    elapsed = end_time - start_time
    output_length = output.shape[1]
    tokens_generated = output_length - input_length
    throughput = tokens_generated / elapsed

    # 3.5 将输出的token id序列转化为句子
    new_tokens = output[0, input_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "elapsed": elapsed,
        "throughput": throughput,
        "input_length": input_length,
        "output_length": tokens_generated,
        "tokens_generated": tokens_generated,
        "decoded_output": decoded_output,
    }

####################################################################################################
# 3. mlp移动到cpu
def move_mlp_to_cpu(model):
    def fwd_cpu(self, x):
        x = x.to("cpu")
        return self._original_forward(x).to('cuda')
    for name, module in model.named_modules():
        if "mlp" in name.lower():
            print("Moving MLP to CPU:", name)
            module._original_forward = module.forward
            module.forward = types.MethodType(fwd_cpu, module)
            module.to(torch.device("cpu"))

# 2. 替换推理
def replace_llama(method, model_name=None):
    # 2.1 将原本的注意力计算，Sdpa注意力计算换成自定义的
    if method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM
        
    # 2.2 将输入的预处理方式也改成自定义的
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new

# 1. 推理
def main():
    # 1. 指定模型和设备
    model_name = "/mtc/longlingkun/models/llama3.1-8b-instruct"
    device = "cuda:3"

    # 2. 加载分词器
    # 2.1 将句子划分成多个有先后顺序的token，并且根据预先训练好的模型中的单词表，将字符转换为单词表中的索引，或者说id
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. 读取长prompt
    with open("long_prompt.txt", "r", encoding="utf-8") as f:
        long_prompt = f.read()

    # 4. 基础推理
    model = load_model(model_name, attn_impl="sdpa", device=device)
    run_and_log("Baseline (SPDA Attention)", model, tokenizer, long_prompt, device)

    # # 5. 做了KV-Cache优化的推理
    # replace_llama("streamingllm")
    # model = load_model(model_name, attn_impl="sdpa", device=device)
    # run_and_log("StreamingLLM + SPDA Attention", model, tokenizer, long_prompt, device)

if __name__ == "__main__":
    main()