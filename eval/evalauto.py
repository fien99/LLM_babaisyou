'''
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    BitsAndBytesConfig,
    pipeline,
)
import json
import peft
from peft import LoraConfig, get_peft_model, PeftModel
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json


def format_prompt(prompt, system_prompt=""):
    if system_prompt.strip():
        return f"<|endoftext|><|user|> {system_prompt}\n{prompt}<|assistant|>"
    return f"<|endoftext|><|user|> {prompt}<|assistant|>"


if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

#MODEL_NAME ="/data1/s2269651/fine_tune/mistral_nemo_instruct-reasoning"
#base_model ="Migthytwig/mistral_7B_instruct-v3-reasoning" #"mistralai/Mistral-7B-Instruct-v0.3" #"mistralai/Mistral-Nemo-Instruct-2407"
#model_id= "mistralai/Mistral-7B-Instruct-v0.3" #"allenai/OLMo-2-1124-7B-Instruct"#"mistralai/Mixtral-8x7B-Instruct-v0.1"
#peft_model_path1 ="/data1/s2269651/fine_tune/baba_rules/data_l/lora_model_base_7Binstruct_reasoning_datababa" 
#peft_model_path2="/data1/s2269651/fine_tune/M7B/lora_model_mis7x8_reason_baba"    #"/data1/s2269651/fine_tune/output_cot/lora_model_7Binstruct_cot"#"/data1/s2269651/fine_tune/output655/lora_model_7Binstruct_reasoning655"  #"/data1/s2269651/fine_tune/output655/lora_model_7Binstruct_reasoning655"
#peft_model_path3= "/data1/s2269651/fine_tune/baba_rules/data_b/lora_model_base_7Binstruct_reasoning_datababa"
#""#"" #"/data1/s2269651/fine_tune/output654/lora_model_7Binstruct_reasoning654"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    #llm_int8_enable_fp32_cpu_offload=True
)

#model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=compute_dtype,device_map="auto", )#, quantization_config=bnb_config)

#tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#tokenizer.padding_side = 'right'
#tokenizer.pad_token_id = tokenizer.eos_token_id # tokenizer.unk_token
#tokenizer.pad_token = tokenizer.unk_token
#EOS_TOKEN = tokenizer.eos_token
#model = PeftModel.from_pretrained(model, peft_model_path2, adapter_name="reason")
#model.load_adapter(peft_model_path1, adapter_name="baba")
#model.load_adapter(peft_model_path1, adapter_name="levels")

#model.add_weighted_adapter(["reason", "baba"], [1.0,0.5], combination_type="cat", adapter_name="reason_baba")
#model.delete_adapter("reason")
#model.delete_adapter("baba")
#model.delete_adapter("levels")

#model.save_pretrained("./cat_1_1")
#model = PeftModel.from_pretrained(model, "./cat_1_1/reason_baba")
#model = model.merge_and_unload()
#

"""
from huggingface_hub import login
login(token="insert_token")
model.push_to_hub("Migthytwig/olmo_7B_instruct-reasoning")
exit()
"""
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForCausalLM.from_pretrained(
   #MODEL_NAME, device_map="auto",  trust_remote_code=True,)


# Load base model
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3" #"allenai/OLMo-2-1124-7B-Instruct"  
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

# Load LoRA adapter
lora_model_path = "/data1/s2269651/fine_tune/M7B/lora_model_mis7_reason_baba"  # Change this to your actual LoRA path
lora_model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.float16)

# Merge LoRA weights
merged_model = lora_model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
pad_token_id= tokenizer.eos_token_id



'''
generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 2048
generation_config.temperature = 0.0001
generation_config.do_sample = True

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, use_chache = True)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer,
    
    
    
)

'''
#You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps: - Think through the problem step by step within the ‹thinking> tags. - Reflect on your thinking to check for any errors or improvements within the ‹reflection› tags. - Make any necessary adjustments based on your reflection. - Provide your final, concise answer within the ‹output> tags. Important: The <thinking> and ‹reflection› sections are for your internal reasoning process only. Do not include any part of the final answer in these sections. The actual response to the query must be entirely contained within the ‹output› tags. Use the following format for your response: <thinking> [Your initial thought process goes here] </thinking› <reasoning> [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer. You can create as many reasoning steps as necessary in your process.] </reasoning> ‹reflection> [Your reflection on your reasoning, checking for errors or improvements. You can create as many reflection steps as necessary in your process.] </ reflection> <adjustment> [Any adjustments to your thinking based on your reflection] </adjustment> <output> [Your final, concise answer to the query. This is the only part that will be shown to the user.] </output>

SYSTEM_PROMPT = """
You are helping to solve a gridworld game. In Baba Is You, the player can change the game rules by moving text blocks around. The grid contains objects and rules that define how the objects behave. The rules consists of text blocks.

Text Blocks:
Object Text: Words representing game objects.
Property Text: Words that describe actions or properties.

Active Rules:
A rule is formed when the text blocks are aligned in a valid way, either horizontally or vertically. Valid rule formats include:
<object_text> IS <property_text>: Grants a property to an object. 
<object1_text> IS <object2_text>: Changes one object into another. 
<object1_text> IS <object1_text>: Makes an object inmutable.
The goal is to use these rules to solve the level by moving the text blocks and controlling the objects.

The level is displayed in a 2D grid, where each character has the following meaning:
Text blocks in the game which always can be pushed:
<object_text>:
    B = BABA,
    S = SKULL,
    F = FLAG,
    O = FLOOR,
    A = GRASS,
    L = LAVA,
    R = ROCK,
    W = WALL,
    K = KEKE,
    G = GOOP,
    V = LOVE

<property_text>:
    1 = IS
    2 = YOU,
    3 = WIN,
    4 = KILL,
    5 = PUSH,
    6 = STOP,
    7 = MOVE,
    8 = HOT,
    9 = MELT,
    0 = SINK

objects in the game:
<object>:
    b = object baba,
    s = object skull,
    f = object flag,
    o = object floor,
    a = object grass,
    l = object lava,
    r = object rock,
    w = object wall,
    k = object keke,
    g = object goop,
    v = object love,
    _ = border,
    . = empty space
    
<object_text> IS YOU: Makes the object you control in the game. You can move it and push text blocks. 
<object_text> IS WIN: The object you need to reach or to be to win the level.
<object_text> IS STOP: Makes the object impassable (you can't move through it).
<object_text> IS MELT: Makes the object melt when touched by something marked as HOT.
<object_text> IS HOT: Makes the object destroy any object marked as MELT when they touch it. If the same object is both HOT and MELT it self-destructs.
<object_text> IS MOVE: Makes the object move one step in its direction every turn.
<object_text> IS KILL: Makes the object destroy anything you control when touched, but it stays intact.
<object_text> IS PUSH: Lets you push the object or have it pushed by other moving objects.
<object_text> IS SINK: Makes the object destroy itself and anything it touches when it is touched.

A solution exist of one or more actions. An action can be one of the following:
Move_To[ ]:
The player-controlled object moves to a valid target in the grid:
<object>, <object_text>, or <property_text>.
This action is valid only if the specified target is present in the grid.

Brake_Rule[ ]:
Removes an active rule unless it’s at the border. Effects depend on the rule:
[<object_text> IS <property_text>]: Removes the property from the object.
[<object1_text> IS <object2_text>]: No changes occur.
[<object1_text> IS <object1_text>]: Makes the object mutable.

Make_Rule[ ]:
Creates a new rule using valid text blocks in the grid, with effects depending on the rule:
[<object_text> IS <property_text>]: Grants a property to the object.
[<object1_text> IS <object2_text>]: Transforms <object1> into <object2>.
[<object1_text> IS <object1_text>]: Makes <object1> immutable.
This action requires all components of the rule to be present in the grid.

Push[ ]:
The player-controlled object pushes another object or text block in the grid. Effects depend on what is pushed:
Text blocks <object_text>, <property_text>: Can be pushed to arrange or create rules.
Objects <object>: Can be pushed towards other objects if the pushed object is set to "push."
This action is valid only if all elements are present in the grid and not placed at the border.
""".strip()

with open("/data1/s2269651/experiment/levels.json", "r") as f:
    levels = json.load(f)["levels"]

# Process each level
results = []

for level in levels:
    level_id = level["id"]
    ascii_grid = level["ascii"]
    active_rules = level["rules"]
    
    print("processing...level: ", level_id)
    # Format the prompt for the current level
    prompt = f"""

Question: Give a solution for the following grid level:
{ascii_grid}
current active rules are:{active_rules}.
Let's first understand the problem, extract the relevant objects, text blocks and rules, explain the rules in the level and make a plan to solve the problem. Then let's carry out the plan by giving the intermediate actions (using common sense). Solve the problem step by step and show the solution.""".strip()
    temp=format_prompt(prompt, SYSTEM_PROMPT)

    # **TEST PROMPT**
    test_prompt = temp

    # Tokenize prompt
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = merged_model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9, do_sample=False)
        
    # Decode and print response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n### Model Response ###\n")
    print(response)   
    


'''
prompt = """

Question: Give a solution to the following grid level:
__________\n_B12..F13_\n_........_\n_........_\n_........_\n_.b....f._\n_........_\n_........_\n_........_\n__________
Current active rules are: BABA IS YOU and FLAG IS WIN.
Let's first understand the problem, extract the relevant objects, text blocks and rules, explain the rules in the level and make a plan to solve the problem. Then let's carry out the plan by giving the intermediate actions (using common sense). Solve the problem step by step and show the solution.""".strip()

temp=format_prompt(prompt, SYSTEM_PROMPT)

# **TEST PROMPT**
test_prompt = temp

# Tokenize prompt
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Generate response
print(inputs)
with torch.no_grad():
    output = merged_model.generate(**inputs, max_new_tokens=1024, temperature=0.5, top_p=0.8, do_sample=True)

# Decode and print response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n### Model Response ###\n")
print(response)
#print(model_id, peft_model_path1, peft_model_path2 )
#print(MODEL_NAME)
#print("Used prompt: ",temp,"\n")
#result = llm(temp)








'''
