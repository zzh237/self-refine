# src/backends/local_hf_api.py (或者你选择的其他路径)
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from typing import Dict, List, Optional, Tuple # 确保导入了 Tuple

# 定义一个简单的停止条件类，用于处理 stop_token
class StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 检查最新生成的token是否是停止token之一
        for stop_id in self.stop_token_ids:
            if input_ids[0, -1] == stop_id: # 只检查最后一个token
                return True
        return False

class LocalHFAPIWrapper:
    _models_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LocalHFAPIWrapper: 使用设备 {_device}")

    @staticmethod
    def get_model_and_tokenizer(model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
        if model_name_or_path not in LocalHFAPIWrapper._models_cache:
            print(f"Hugging Face Transformers: 正在从本地路径加载模型: {model_name_or_path}...")
            try:
                # 如果模型很大，考虑量化加载
                # quantization_config = BitsAndBytesConfig(load_in_8bit=True) # 或 load_in_4bit=True
                # model = AutoModelForCausalLM.from_pretrained(
                #     model_name_or_path,
                #     quantization_config=quantization_config, # 使用量化配置
                #     torch_dtype=torch.bfloat16, # 或 torch.float16
                #     device_map="auto", # 自动分配到GPU或CPU/多GPU
                #     trust_remote_code=trust_remote_code,
                #     **kwargs.get("model_load_kwargs", {})
                # )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.bfloat16 if LocalHFAPIWrapper._device == "cuda" else torch.float32, # 自动选择精度
                     **kwargs.get("model_load_kwargs", {}) # 允许传递其他加载参数
                ).to(LocalHFAPIWrapper._device) # 明确移到设备
                
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                LocalHFAPIWrapper._models_cache[model_name_or_path] = (model, tokenizer)
                print(f"模型 {model_name_or_path} 加载完成。")
            except Exception as e:
                print(f"加载模型 {model_name_or_path} 失败: {e}")
                raise e
        return LocalHFAPIWrapper._models_cache[model_name_or_path]

    @staticmethod
    def call(prompt: str, 
             engine: str, # model_name_or_path
             max_tokens: int, 
             stop_token: Optional[str] = None, 
             temperature: float = 0.0,
             trust_remote_code: bool = True, # 从 get_model_and_tokenizer 传入
             **kwargs) -> dict:
        """
        使用本地加载的Hugging Face模型进行调用。
        """
        model_load_kwargs = kwargs.get("model_load_kwargs", {})
        model, tokenizer = LocalHFAPIWrapper.get_model_and_tokenizer(engine, trust_remote_code, **model_load_kwargs)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False).to(LocalHFAPIWrapper._device) # padding 和 truncation 通常在批处理时更重要

        stopping_criteria = None
        if stop_token:
            # Hugging Face 的 stop token 通常是在生成后通过文本匹配来处理，或者用 StoppingCriteria
            # 对于简单的单个字符串停止标记，可以尝试在生成后截断
            # 如果 stop_token 是tokenizer词汇表中的单个token，可以创建StoppingCriteria
            # 这里为了简化，我们假设 stop_token 是一个字符串，在生成后进行截断
            pass


        # 生成参数设置
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id, # 确保设置了pad_token_id
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 1e-5 : # 避免纯粹的0.0，有些模型可能不喜欢
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
            # 你可能还想在这里加入 top_p, top_k 等参数
            # generation_kwargs["top_p"] = kwargs.get("top_p", 0.9)
            # generation_kwargs["top_k"] = kwargs.get("top_k", 50)
        else: # 贪心解码
            generation_kwargs["do_sample"] = False
            # 对于贪心，通常不需要设置 temperature, top_p, top_k

        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = stopping_criteria
            
        with torch.no_grad():
            # `generate` 方法返回的是包含输入prompt的完整序列
            outputs_ids = model.generate(inputs.input_ids, **generation_kwargs)
        
        # 解码生成的token IDs (只解码新生成的token部分)
        # outputs_ids[0] 的形状是 [1, sequence_length]
        response_ids = outputs_ids[0, inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # 处理 stop_token (如果在生成时没有通过StoppingCriteria精确停止)
        if stop_token and stop_token in response_text:
            response_text = response_text.split(stop_token)[0]

        prompt_tokens = inputs.input_ids.shape[1]
        completion_tokens = len(response_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "choices": [{"text": response_text.strip()}], # strip() 一下确保干净
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

    @staticmethod
    def get_first_response(output: dict) -> str:
        if output and "choices" in output and len(output["choices"]) > 0:
            return output["choices"][0].get("text", "")
        return ""