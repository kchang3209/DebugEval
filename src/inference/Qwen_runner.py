from openai import OpenAI
import os
import asyncio # CSE247
from vllm import LLM, SamplingParams

# def Qwen_runner(args,messages,model,tokenizer):
def Qwen_runner(args,prompt,model,sampling_params):

    # client = OpenAI(
    #     api_key='',
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # response = client.chat.completions.create(
    #     model=args.model,
    #     messages=messages,
    #     max_tokens=args.max_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     stop=[],
    #     n=args.n
    # )
    # responses = [response.choices[i].message.content for i in range(args.n)]
    # return responses
    

    # if args.mode == 'text_only':
    outputs = model.generate(prompt, sampling_params)
    # num_text_tokens = len(outputs[0].outputs[0].token_ids)
    response = outputs[0].outputs[0].text
    req_token = len(outputs[0].prompt_token_ids)
    return response, req_token


    # =======================
    # CSE247 HF Model Loader
    # =======================
    # if args.mode == 'text_only':
        
    #     text = tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    #     generated_ids = model.generate(
    #         **model_inputs,
    #         max_new_tokens=1024,
    #         do_sample=False
    #     )

    #     num_text_tokens = model_inputs["input_ids"].shape[-1]

    #     output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    #     output = tokenizer.decode(output_ids, skip_special_tokens=True)

    #     return output, num_text_tokens 
    
    # else:
    #     return
