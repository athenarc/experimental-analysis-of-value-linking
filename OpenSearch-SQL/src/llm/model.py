import requests, time
import dashscope
import torch
import json
import re
from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("VLLM not installed. Please install with 'pip install vllm'")
    LLM, SamplingParams = None, None

def model_chose(step,model="gpt-4 32K"):
    if model.startswith("gpt") and "gpt-oss" not in model:
        return gpt_req(step,model)
    elif model.startswith("claude") or model.startswith("gemini"):
        return gpt_req(step,model)
    elif model.startswith("deepseek"):
        return deep_seek(model)
    elif model.startswith("qwen"):
        return qwenmax(model)
    elif model.startswith("sft"):
        return sft_req()
    else:
        # Default to VLLM for other models like Qwen, Llama, gpt-oss etc.
        return VLLM_Generator(step, model_name=model)


class req:

    def __init__(self,step,model) -> None:
        self.Cost = 0
        self.model=model
        self.step=step

    def log_record(self,prompt_text,output):
        logger=Logger()
        logger.log_conversation(prompt_text, "Human", self.step)
        logger.log_conversation(output, "AI", self.step)

    def fewshot_parse(self, question, evidence, sql):
        s = prompts_fewshot_parse().parse_fewshot.format(question=question,sql=sql)
        ext = self.get_ans(s)
        ext=ext.replace('```','').strip()
        ext = ext.split("#SQL:")[0]
        ans = self.convert_table(ext, sql)
        return ans
    def convert_table(self, s, sql):
        l = re.findall(' ([^ ]*) +AS +([^ ]*)', sql)
        x, v = s.split("#values:")
        t, s = x.split("#SELECT:")
        for li in l:
            s = s.replace(f"{li[1]}.", f"{li[0]}.")
        return t + "#SELECT:" + s + "#values:" + v

def request(url,model,messages,temperature,top_p,n,key,**k):
    res = requests.post(
                url=
                url,
                json={
                    "model":
                    model,
                    "messages": [{
                        "role": "system",
                        "content":
                        "You are an SQL expert, skilled in handling various SQL-related issues."
                    }, {
                        "role": "user",
                        "content": messages
                    }],
                    "max_tokens":
                    800,
                    "temperature":
                    temperature,
                    "top_p":top_p,
                    "n":n,
                    **k
                },
                headers={
                    "Authorization":
                    key
                }).json()

    return res

class gpt_req(req):

    def __init__(self, step,model="gpt-4o-0513") -> None:
        super().__init__(step,model)

    def get_ans(self, messages, temperature=0.0, top_p=None,n=1,single=True,**k):
        count = 0
        while count < 50:
            try:
                res = request(
                url=
                "",
                model=self.model,
                messages= messages,
                temperature=temperature,
                top_p=top_p,
                n=n,key="",
                    **k)
                if n==1 and single:
                    response_clean = res["choices"][0]["message"]["content"]
                else:
                    response_clean = res["choices"]
                if self.step!="prepare_train_queries":
                    self.log_record(messages, response_clean)
                break

            except Exception as e:
                count += 1
                time.sleep(2)
                print(e, count, self.Cost,res)

        self.Cost += res["usage"]['prompt_tokens'] / 1000 * 0.042 + res[
            "usage"]["completion_tokens"] / 1000 * 0.126
        return response_clean
    


class deep_seek(req):

    def __init__(self,model) -> None:
        super().__init__(model)
    def get_ans(self, messages, temperature=0.0, debug=False):
        count = 0

        while count < 8:
            try:
                url = "https://api.deepseek.com/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization":
                    ""
                }

                jsons = {
                    "model":
                    "deepseek-coder",
                    "temperture":
                    temperature,
                    "top_p":
                    0.9,
                    "messages": [{
                        "role": "system",
                        "content": "You are a helpful assistant."
                    }, {
                        "role": "user",
                        "content": messages
                    }]
                }

                response = requests.post(url, headers=headers, json=jsons)
                if debug:
                    print(response.json)
                ans = response.json()['choices'][0]['message']['content']
                break
            except Exception as e:
                count += 1
                time.sleep(2)
                print(e, count, self.Cost, response.json())
        return ans


class qwenmax(req):

    def __init__(self, model) -> None:
        super().__init__(model)
        dashscope.api_key = ""
 

    def get_ans(self, messages, temperature=0.0, debug=False):
        count = 0

        while count < 8:
            try:
                response = dashscope.Generation.call(model=self.model,
                                                     temperature=temperature,
                                                     prompt=messages)
                self.Cost += response.usage.input_tokens / 1000 * 0.04 + response.usage.output_tokens / 1000 * 0.12
                return response.output['text']
            except:
                count += 1
                time.sleep(5)
                print(response.code, response.message)


class sft_req(req):

    def __init__(self,model) -> None:
        super().__init__(model)
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "",
            trust_remote_code=True,
            padding_side="right",
            use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token = "<|EOT|>"
        self.model = AutoModelForCausalLM.from_pretrained(
            "",
            torch_dtype=torch.bfloat16,
            device_map=self.device).eval()

    def get_ans(self, text, temperature=0.0):
        messages = [{
            "role":
            "system",
            "content":
            "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
        }, {
            "role": "user",
            "content": text
        }]
        inputs = self.tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    tokenize=False)
        model_inputs = self.tokenizer([inputs],
                                      return_tensors="pt",
                                      max_length=8000).to("cuda")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=800,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.decode(generated_ids[0][:-1],
                                         skip_special_tokens=True).strip()
        return response


class VLLM_Generator(req):
    _llm_instance = None

    def __init__(self, step, model_name, **kwargs):
        super().__init__(step, model_name)
        if LLM is None:
            raise ImportError("VLLM is not installed. Please run 'pip install vllm'.")
        
        if VLLM_Generator._llm_instance is None:
            print(f"Initializing VLLM model {model_name} for the first time.")
            VLLM_Generator._llm_instance = LLM(model=model_name, tensor_parallel_size=2,gpu_memory_utilization=0.87,download_dir="/data/hdd1/vllm_models/",max_model_len=32768)
        self.llm = VLLM_Generator._llm_instance

    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **kwargs):
        prompt = messages
        
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if temperature > 0.0 else 0.0,
            top_p=top_p if top_p is not None else 1.0,
            max_tokens=kwargs.get("max_tokens", 800),
            stop=kwargs.get("stop", None)
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        
        if single:
            return outputs[0].outputs[0].text
        else:
            return [output.text for output in outputs[0].outputs]

    def batch_generate(self, prompts: list, temperature=0.0, top_p=None, n=1, **kwargs):
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if temperature > 0.0 else 0.0,
            top_p=top_p if top_p is not None else 1.0,
            max_tokens=kwargs.get("max_tokens", 1024),
            stop=kwargs.get("stop", None)
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            generated_texts = [o.text for o in output.outputs]
            results.append(generated_texts)
        
        return results