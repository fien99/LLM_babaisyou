from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from huggingface_hub import login
import random


login(token="insert_token")

"""
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"<s> [INST] Question:\n { example['question'][i]}. \n Let’s first understand the problem and devise a plan to solve the problem. Then, let’s carry out the plan give the intermidiate actions. Solve the problem step by step, and show the answer [/NST]\n Reasoning:\n {example['generation'][i]}\n Answer:\n {example['answer'][i]}</s>"
        #text = text + EOS_TOKEN
        output_texts.append(text)
        print(text)
        quit()
    return output_texts
"""
def formatting_prompts_func3(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"<|endoftext|><|user|>\nQuestion:\n { example['prompt'][i]}. \n Let’s first understand the problem and devise a plan to solve the problem. Then, let’s carry out the plan give the intermidiate actions. Solve the problem step by step, and show the answer\n<|assistant|> Answer:\n {example['response'][i]}<|endoftext|>"
        #text = text + EOS_TOKEN
        output_texts.append(text)
    return output_texts




def formatting_prompts_func(example):
    #example =example["data"]

    output_texts= []
    for i in range(len(example["ascii"])):

        text = f"""
<|endoftext|><|user|>
You are helping in solving a gridworld game. In this game (Baba Is You) the player can change the rules of the game by pushing text blocks around <object_text> or <property_text>. charachters in the grid can be objects or text words.
A rule is active if the text blocks are aligned horizontally or vertically and they form a valid rule. A valid rule has the form <object_text > IS <property_text>, <object1_text> IS <object2_text> or <object1_text> IS <object1_text>.
the level is represented in a 2D grid where each character has the following meaning:

_ = border,
. = empty space,

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
    v = object love





<object_text> IS YOU, sets the <object> that the player (you) controls in the grid (moving it left, right, up, down) and is able to push text blocks.
<object_text> IS WIN, sets the <object> that the player needs to reach or needs to be to win the level. 
<object_text> IS STOP, sets the <object> to be impassable.
These words work in tandem. <object1_text> IS MELT sets the <object> that will be destroyed by <object2_text> IS HOT, while other object remains intact. A little reminder that these words have nothing to do with <object> IS YOU nor the objects they modify. This means that ICE can be HOT and LAVA can be MELT! Also note that if something is both HOT and MELT, it will self-destruct.
<object_text> IS MOVE, sets the <object> to move one step in their facing direction per turn of the player (left, right, up, down).
<object_text> IS KILL, sets the <object> to destroy all objects controlled by the player when the player touches them, while remaining intact itself. 
<object_text> IS PUSH, sets the <object> to be pushed by the player or by other moving objects, moving it one tile forward and occupying the new tile if possible.
<object_text> IS SINK, sets the <object> to destroy itself and the player if the player touches it.


A solution exist of one or more actions. An action can be one of the following:
Move_To[<object>], meaning moving the object controlled by the player to the <object> in the grid. This action can only be taken if the <object> is present in the grid level.
Move_To[<object_text>], meaning moving the object controlled by the player to the <object_text> in the grid. This action can only be taken if the <object_text> is present in the grid level.
Move_To[<property_text>], meaning moving the object controled by the player to the <property_text> in the grid. This action can only be taken if the <property_text is present in the grid level.
Brake_Rule[<object_text> IS <property_text>], meaning removing the property of an object. This action can only be taken if the rule is active in the grid level. This action can only be taken if the <property_text> and IS text block are both present in the grid.
Brake_Rule[<object1_text> IS <object2_text>], notting happens after breaking this rule. This action can only be taken if the rule is active in the grid level.
Brake_Rule[<object1_text> IS <object1_text>], meaning the object becomes mutable again. This action can only be taken if the rule is active in the grid level.
Make_Rule[<object_text> IS <property_text>], meaning creating a rule that gives a object an property. This action can only be taken if the <property_text>, IS and <object_text> are both present in the grid level.
Make_Rule[<object1_text> IS <object2_text>], meaning creating a rule that transforms object1 in object2. This action can only be taken if the <object1_text>, IS and <object2_text> are both present in the grid level.
Make_Rule[<object1_text> IS <object1_text>], meaning creating a rule that makes object1 immutable. This action can only be taken if IS and the <object1_text> is twice present in the grid level.
Push[<object_text>_To[<property_text>], meaning pushing, with the object controlled by the player, <object_text> text block towards <property_text> text_block. This action can only be taken if the <property_text> and <object_text> are both present in the grid level and <property_text> or <object_text> are both not placed against the border.
Push[<property_text>_To[<object_text>], meaning pushing, with the object controlled by the player, <property_text> text block towards <object_text> text_block. This action can only be taken if the <property_text> and <object_text> are both present in the grid level and <property_text> or <object_text> are both not placed against the border.
Push[<property1_text>_To[<property2_text>], meaning pushing, with the object controlled by the player, <property1_text> text block towards <property2_text> text_block. This action can only be taken if the <property1_text> and <property2_text> are both present in the grid level and <property1_text> or <property2_text> are both not placed against the border.
Push[<object1>_To[<object2>], meaning pushing, with the object controlled by the player, <object1> towards <object2>. This action can only be taken if <object1> is set to push and <object1> and <object2> are both present in the grid.

Question: Give a solution for the following level: 
{example['ascii'][i]}
With the following active rules: {example['Active Rules'][i]}


<|assistant|>
Answer:
Descripition level:  {example['description level'][i]}

{example['text solution'][i]}<|endoftext|>"""
        
        output_texts.append(text)
    return output_texts

def formatting_prompts_func2(example):

    output_texts= []
    for i in range(len(example["question"])):
        text= f"""
<|endoftext|><|user|> 
Question:{example["question"][i]}
<|assistant|>
Answer:{example["answer"][i]}<|endoftext|>
"""
        output_texts.append(text)
    return output_texts

dataset1 = load_dataset("json", data_files="/data1/s2269651/fine_tune/baba_rules/description.json",field="levels",split="train")
dataset2 = load_dataset("json", data_files="/data1/s2269651/fine_tune/baba_rules/data.json",field="data",split="train")
#dataset3=load_dataset("thesven/gsm8k-reasoning",split="train")

dataset3=load_dataset("isaiahbjork/cot-logic-reasoning",split="train")

formatted_dataset1 = formatting_prompts_func(dataset1)  # Apply the first function
formatted_dataset2 = formatting_prompts_func2(dataset2)  # Apply the second function
formatted_dataset3 = formatting_prompts_func3(dataset3)

# Combine datasets
combined_dataset = formatted_dataset1 + formatted_dataset2 + formatted_dataset3

# Shuffle the combined dataset
random.shuffle(combined_dataset)


hf_dataset = Dataset.from_dict({"text":combined_dataset})



torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"#"allenai/OLMo-2-1124-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    #llm_int8_enable_fp32_cpu_offload=True
)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, quantization_config=bnb_config, trust_remote_code=True,device_map="auto")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    #target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",

)

model = get_peft_model(model, config)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="M7B",      
    #evaluation_strategy="steps",         # Evaluate the model periodically
    save_strategy="steps",               # Save model checkpoints periodically
    logging_dir="./logs",                # Directory for logs
    learning_rate=2e-5,                  # Learning rate for fine-tuning
    per_device_train_batch_size=4,       # Batch size per device
    gradient_accumulation_steps=8,       # Accumulate gradients to simulate larger batches
    num_train_epochs=8,                  # Number of training epochs
    logging_steps=100,                   # Log progress every 100 steps
    save_steps=500,                      # Save a checkpoint every 500 steps
    fp16=True,                           # Enable mixed precision for faster training
    push_to_hub=False                    # Disable auto-push to Hugging Face Hub
)




from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=hf_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=4069,
    packing= False,
    peft_config=config,
    #formatting_func=formatting_prompts_func,

)

trainer.train()
peft_model= os.path.join('M7B', f"lora_model_mis7x8_reason_baba")
trainer.model.save_pretrained(peft_model)
model.config.use_cache = True
model.eval()


#login(token="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")
#trainer.model.push_to_hub(new_model)





#logging.set_verbosity(logging.CRITICAL)
#pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
