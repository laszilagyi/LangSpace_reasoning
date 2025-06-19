import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.data import Dataset, DataLoader
import json

from langspace_core_alt import *

model_id = "llama3_2_3B"

model_name = f"./SFT/{model_id}/lora/epoch_1" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=8,
    ppo_epochs=10,
    target_kl=0.1
)

class SceneGraphDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        system_prompt = item["system_prompt"]
        room_graph = item["room_graph"]
        scene_graph = item["scene_graph"]
        task_description = item["task_description"]
        verifying_step = item["verifying_step"]
        
        query = f"{system_prompt} The room adjacencies to strictly follow: {room_graph} The 3D Scene Graph in JSON format: {scene_graph} The task description: {task_description}"
        return {
            "input_ids": self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)["input_ids"],
            "scene_graph": scene_graph,
            "verifying_step": verifying_step
        }

dataset = SceneGraphDataset("./ppo_trainer_dataset.jsonl", tokenizer)
dataloader = DataLoader(dataset, batch_size=ppo_config.batch_size, shuffle=True)

def validate_plan(plan, scene_graph, verifying_step):
    
    try:
        plan = json.loads(plan)
    except Exception as e:
        return -1.0
        
    try:
        scene_graph = json.loads(scene_graph)
    except Exception as e:
        print("Scenegraph JSON load error.")
        return -1.0

    try:
        verifying_step = json.loads(verifying_step)
    except Exception as e:
        print("Verifying step JSON load error.")
        return -1.0
    
    test_scene_graph = LSSceneGraph.from_json(scene_graph)
    exec_result = test_scene_graph.execute_plan(plan)
    validate_result = test_scene_graph.check_objects_classes(verifying_step)
    
    if exec_result == "Valid plan." and validate_result == "Valid plan.":
        return 1.0
    elif exec_result == "Valid plan.":
        return 0.5
    else:
        return -1.0

def compute_rewards(generated_plans, scene_graphs, verifying_steps):
    rewards = [validate_plan(plan, graph, verifying_step) for plan, graph, verifying_step in zip(generated_plans, scene_graphs, verifying_steps)]
    return torch.tensor(rewards, dtype=torch.float16).to("cuda")

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None, 
    tokenizer=tokenizer,
    dataset=dataset
)

for epoch in range(3):
    for batch in dataloader:
        queries = batch["input_ids"].squeeze(1).to("cuda")
        responses = [ppo_trainer.generate(q, max_new_tokens=100)[0] for q in queries]
        generated_plans = [tokenizer.decode(r, skip_special_tokens=True) for r in responses]
        scene_graphs = batch["scene_graph"]
        verifying_steps = batch["verifying_step"]
        rewards = compute_rewards(generated_plans, scene_graphs)
        ppo_trainer.step(queries, responses, rewards)
        print(f"Epoch {epoch}, Rewards: {rewards.mean().item()}")

model.save_pretrained(f"./PPO/{model_id}/fine_tuned_llama_rl")
tokenizer.save_pretrained(f"./PPO/{model_id}/fine_tuned_llama_rl")