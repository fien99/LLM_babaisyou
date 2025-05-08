import torch
import json
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    BitsAndBytesConfig,
    pipeline,
)


if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

def format_prompt(prompt, system_prompt=""):
    if system_prompt.strip():
        return f"<|endoftext|><|user|> {system_prompt}\n{prompt}"
    return f"<|endoftext|><|user|> {prompt}"

MODEL_NAME = ["allenai/OLMo-2-1124-13B-Instruct","mistralai/Mistral-7B-Instruct-v0.3","mistralai/Mixtral-8x7B-Instruct-v0.1","allenai/OLMo-2-1124-7B-Instruct"]
MODEL_NAME2=["mistralai/Mistral-Small-24B-Instruct-2501"]
def create_pipeline(model_id, model, tokenizer):
    generation_config = GenerationConfig.from_pretrained(model_id)
    generation_config.max_new_tokens = 2048
    generation_config.temperature = 0.0001
    generation_config.do_sample = True

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        num_return_sequences=1,
        #eos_token_id=tokenizer.eos_token_id,
        #pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

# Quantization configuration
for name in MODEL_NAME2:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=compute_dtype,
       
        trust_remote_code=True,
    )



    # System prompt
    SYSTEM_PROMPT = """
You are helping to solve a gridworld game. In Baba Is You, the player can change the game rules by moving text blocks around. The grid contains object text and property text.

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

    # Load levels from JSON file
    with open("levels.json", "r") as f:
        levels = json.load(f)["levels"]

    # Process each level
    results = []

    for level in levels:
        level_id = level["id"]
        ascii_grid = level["ascii"]
        active_rules = level["rules"]
        

        # Format the prompt for the current level
        prompt = f"""
Question: Give a solution for the following grid level:
{ascii_grid}
current active rules are: {active_rules}.
Let’s first understand the problem extract the relevant objects, text blocks and rules (explain the rules) in the given level and devise a plan to solve the problem. Then, let’s carry out the plan give the intermidiate actions (pay attention to common sense). Solve the problem step by step, and show the answer.
""".strip()

        # Run the model to generate the solution
        print(f"Processing Level {level_id}...")
        formatted_prompt = format_prompt(prompt, SYSTEM_PROMPT)
        llm = create_pipeline(name, model, tokenizer)
        print(formatted_prompt)
        
        generated_solution = llm(formatted_prompt)

        # Store the result
        results.append({
            "id": level_id,
            "ascii": ascii_grid,
            "rules": active_rules,
            "generated_solution": generated_solution,
        })

    # Save the results to a JSON file
    text=name.split("/")[-1]
    text_name=f"solutions_prompt3_{text}_un.txt"
    with open(text_name, "w") as f:
        for result in results:
            f.write(f"Level ID: {result['id']}\n")
            f.write(f"ASCII Grid:\n{result['ascii']}\n")
            f.write(f"Active Rules: {result['rules']}\n")
            f.write(f"Generated Solution:\n{result['generated_solution']}\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"Processing complete. Solutions saved to {text}.txt.")
    

    del model
    gc.collect()
    torch.cuda.empty_cache()


'''
current active rules are: {active_rules}.


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
'''