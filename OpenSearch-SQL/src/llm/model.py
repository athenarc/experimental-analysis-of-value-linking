import requests, time
import dashscope
import torch
import json
import re
from runner.logger import Logger
from llm.prompts import prompts_fewshot_parse
def model_chose(step,model="gpt-4 32K"):
    if model.startswith("gpt") or model.startswith("claude35_sonnet") or model.startswith("gemini"):
        return gpt_req(step,model)
    if model == "deepseek":
        return deep_seek(model)
    if model.startswith("qwen"):
        return qwenmax(model)
    if model.startswith("sft"):
        return sft_req()
    if model == "kosbu/Llama-3.3-70B-Instruct-AWQ": # Add your model identifier
        return VLLM_req(step, model_name=model)
    if model == "Qwen/Qwen2.5-Coder-32B-Instruct":
        return VLLM_req(step, model_name=model)
    if model == "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8":
        return VLLM_req(step, model_name=model)


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
        ext = ext.split("#SQL:")[0]# 防止没按格式生成 至少保留SQL
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
            # print(messages) #保存prompt和答案
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
                # print(self.step)
                if self.step!="prepare_train_queries":
                    self.log_record(messages, response_clean)  # 记录对话内容
                break

            except Exception as e:
                count += 1
                time.sleep(2)
                # print(messages)
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

                # 定义请求体
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

                # 发送POST请求
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
        # drop device_map if running on CPU
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
        # tokenizer.eos_token_id is the id of <|EOT|> token
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




class VLLM_req(req):
    def __init__(self, step, model_name="custom_vllm", api_url="http://localhost:5001/v1/chat/completions"):
        super().__init__(step, model_name)
        self.api_url = api_url
        # You might want to make api_url configurable via pipeline_setup if needed

    def get_ans(self, messages, temperature=0.0, top_p=None, n=1, single=True, **kwargs):
        count = 0
        response_clean = None
        res_json = None

        headers = {
            "Content-Type": "application/json",
        }
        payload_messages = [
            {"role": "system", "content": "You are an SQL expert, skilled in handling various SQL-related issues."},
            {"role": "user", "content": messages}
        ]
        max_tokens = kwargs.get("max_tokens", 800)

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature if temperature > 0 else 0.01,
            "top_p": top_p if top_p is not None else 1.0,
            "n": n,
            "max_tokens": max_tokens,
        }
        if top_p is None:
            del payload["top_p"]
        
        stop_sequences = kwargs.get("stop", None)
        if stop_sequences:
            payload["stop"] = stop_sequences

        while count < 5:
            try:
                http_response = requests.post(self.api_url, headers=headers, json=payload)
                http_response.raise_for_status()
                res_json = http_response.json()

                if n == 1 and single:
                    content_str = res_json["choices"][0]["message"]["content"]
                    try:
                        if content_str.strip().startswith('[') and content_str.strip().endswith(']'):
                            parsed_content_list = json.loads(content_str)
                            if isinstance(parsed_content_list, list) and parsed_content_list:
                                response_clean = parsed_content_list[0]
                            else:
                                response_clean = content_str
                        else:
                            response_clean = content_str
                    except json.JSONDecodeError:
                        response_clean = content_str
                else:
                    temp_responses = []
                    for choice in res_json["choices"]:
                        choice_content_str = choice["message"]["content"]
                        try:
                            if choice_content_str.strip().startswith('[') and choice_content_str.strip().endswith(']'):
                                parsed_list = json.loads(choice_content_str)
                                if isinstance(parsed_list, list) and parsed_list:
                                    temp_responses.append(parsed_list[0])
                                else:
                                    temp_responses.append(choice_content_str)
                            else:
                                temp_responses.append(choice_content_str)
                        except json.JSONDecodeError:
                            temp_responses.append(choice_content_str)
                    response_clean = temp_responses

                if self.step != "prepare_train_queries":
                    self.log_record(messages, response_clean)
                break
            except requests.exceptions.RequestException as e:
                print(f"vLLM request failed: {e}, count: {count+1}, payload: {json.dumps(payload)}")
                count += 1
                time.sleep(2)
            except KeyError as e:
                print(f"vLLM response format error: {e}, response: {res_json}, count: {count+1}")
                count += 1
                time.sleep(2)
            except Exception as e:
                print(f"An unexpected error occurred with vLLM: {e}, response: {res_json}, count: {count+1}")
                count += 1
                time.sleep(2)

        if response_clean is None:
            raise Exception(f"Failed to get a response from vLLM after multiple retries. Last response: {res_json}")

        return response_clean
    
    