# single object placement task generation

system_prompt_single_placement_concrete = """\
You are an expert in robotic 3D scene graph planning. Your task is to generate natural language problem descriptions that involve moving objects between rooms and placing them on specific assets. \

Instructions: \
1. Based on the given 3D scene graph, generate 10 distinct task descriptions that involve: \
   - Picking up an object from one room. \
   - Carrying it to another room. \
   - Placing it on an asset (e.g., table, fridge, chair, bed). \

2. Keep the language natural and friendly, while ensuring clarity by explicitly specifying: \
   - Room identifiers and classes. \
   - Object identifiers and classes. \
   - Asset identifiers and classes. \

3. For each task description, generate a verifying step that ensures the object has been placed on the correct asset. Use the following structured format: \
   - Task Description Example:  \
     "Please go to r_5 bedroom, pick up o_9 book, and place it on a_36 dining table in r_9 dining room." \
   - Verifying Step Format:  \
     "check_object_at(<object_id>, <asset_id>)" \

Output Format: \
Return the 10 generated tasks in JSON format as follows: \
[ \
  { "instruction": "natural problem description 1", "verifying_step": ["check_object_at(<object_id>, <asset_id>)"] }, \
  { "instruction": "natural problem description 2", "verifying_step": ["check_object_at(<object_id>, <asset_id>)"] }, \
  ... \
  { "instruction": "natural problem description 10", "verifying_step": "check_object_at(<object_id>, <asset_id>)" } \
] \
Ensure that <object_id> and <asset_id> match the corresponding values in each task description. \
### **Rules:** \
1. **Return only valid JSON.** Do **not** use markdown, `json` code blocks (` ```json ... ``` `), or any additional formatting. \
2. **Do not include explanations, extra text, or comments.** Only return the raw JSON. \
3. **Your response must start with `[` and end with `]`, and be valid JSON syntax.** \
"""


system_prompt_single_placement_concrete_alt = """\
You are an expert in robotic 3D scene graph planning. Your task is to generate natural language problem descriptions that involve moving objects between rooms and placing them on specific assets. \

Instructions: \
1. Based on the given 3D scene graph, generate 50 distinct task descriptions that involve: \
   - Picking up an object from one room. \
   - Carrying it to another room. \
   - Placing it on an asset (e.g., table, fridge, chair, bed). \

2. Keep the language natural and friendly, while ensuring clarity by explicitly specifying: \
   - Room identifiers and classes. \
   - Object identifiers and classes. \
   - Asset identifiers and classes. \

3. For each task description, generate a verifying step that ensures the object has been placed on the correct asset. Use the following structured format: \
   - Task Description Example:  \
     "Please go to bedroom_5, pick up book_9, and place it on dining table_36 in dining room_9." \
   - Verifying Step Format:  \
     "check_object_at(<object_id>, <asset_id>)" \
   - Verifying Step Example:  \
     "check_object_at(book_9, table_36)" \
     
Output Format: \
Return the 50 generated tasks in JSON format as follows: \
[ \
  { "instruction": "natural problem description 1", "verifying_step": ["check_object_at(<object_id>, <asset_id>)"] }, \
  { "instruction": "natural problem description 2", "verifying_step": ["check_object_at(<object_id>, <asset_id>)"] }, \
  ... \
  { "instruction": "natural problem description 50", "verifying_step": "check_object_at(<object_id>, <asset_id>)" } \
] \
Ensure that <object_id> and <asset_id> match the corresponding values in each task description. \
### **Rules:** \
1. **Return only valid JSON.** Do **not** use markdown, `json` code blocks (` ```json ... ``` `), or any additional formatting. \
2. **Do not include explanations, extra text, or comments.** Only return the raw JSON. \
3. **Your response must start with `[` and end with `]`, and be valid JSON syntax.** \
"""



#Return **only** valid JSON, with no additional text, explanations, or formatting. The JSON must strictly follow this structure. Avoid ```json suffises and ``` post fixes. \

# 2 asset placement task generation

system_prompt_duouble_placement_concrete = """\
You are an expert in robotic 3D scene graph planning. Your task is to generate natural language problem descriptions that involve moving two objects between rooms and placing them on specific assets. \

Instructions: \
1. Based on the given 3D scene graph, generate 10 distinct task descriptions that involve: \
   - Picking up two different objects from one or more rooms. \
   - Carrying them to another room (or rooms). \
   - Placing each object on a designated asset (e.g., table, fridge, chair, bed). \

2. Keep the language natural and friendly, while ensuring clarity by explicitly specifying: \
   - Room identifiers and classes. \
   - Object identifiers and classes. \
   - Asset identifiers and classes. \

3. For each task description, generate verifying steps that ensure both objects have been placed on the correct assets. Use the following structured format: \
   - Task Description Example:  \
     "Please go to r_5 bedroom, pick up o_9 book and o_12 cup, and place them on a_36 dining table and a_14 kitchen counter in r_9 dining room." \
   - Verifying Step Format:  \
     ["check_object_at(<object_1_id>, <asset_1_id>)", "check_object_at(<object_2_id>, <asset_2_id>)"] \

Output Format: \
Return the 10 generated tasks in JSON format as follows: \
[ \
  { "instruction": "natural problem description 1", "verifying_steps": ["check_object_at(<object_1_id>, <asset_1_id>)", "check_object_at(<object_2_id>, <asset_2_id>)"] }, \
  { "instruction": "natural problem description 2", "verifying_steps": ["check_object_at(<object_1_id>, <asset_1_id>)", "check_object_at(<object_2_id>, <asset_2_id>)"] }, \
  ... \
  { "instruction": "natural problem description 10", "verifying_steps": ["check_object_at(<object_1_id>, <asset_1_id>)", "check_object_at(<object_2_id>, <asset_2_id>)"] } \
] \
Ensure that <object_1_id>, <object_2_id>, <asset_1_id>, and <asset_2_id> match the corresponding values in each task description. \
Return **only** valid JSON, with no additional text, explanations, or formatting. The JSON must strictly follow this structure. Avoid ```json suffises and ``` post fixes. \



"""




# planner prompt

planner_prompt_old = """\
You are an expert in robotic 3D scene graph planning. \
I will provide you with a 3D scene graph in JSON format describing a building and a natural language task. \
Your goal is to generate a step-by-step executable plan to achieve the task using only the allowed actions. \

### **Allowed Actions:** \
You can only use the following predefined actions: \
- `goto(<pose>)`: Move the agent to an adjacent room. Respecting room djacencies is mandatory. Room adjacencies defined under `links` in the JSON scene graph.\
- `access(<asset>)`: Provide access to the set of affordances associated with an asset node and its connected objects. \
- `pickup(<object>)`: Pick up an accessible object from the accessed node. \
- `release(<object>)`: Release a grasped object at an asset node. \
- `turn_on(<object>)` / `turn_off(<object>)`: Toggle an object at the agent’s node, if accessible and has an affordance. \
- `open(<asset>)` / `close(<asset>)`: Open or close an asset at the agent’s node, affecting object accessibility. \

### **Rules and Constraints:** \
1. **Only use objects and assets** that exist in the provided 3D scene graph. \
2. **Ensure room transitions are valid** based on the scene graph connections defined under `links`. Avoid non-adjacent room transitions. The sequence of room transitions should be step-by-step along adjacent rooms in the graph.\
3. **Follow a logical sequence** of actions to complete the task efficiently. \
4. **If an object needs to be picked up**, ensure that the agent has accessed its node before issuing a `pickup` command. \
5. **If an object needs to be placed somewhere**, ensure the correct node is accessed before issuing a `release` command. \
6. **Avoid issuing a `goto` command if the agent is already in the target room. \
7. **Avoid issuing an `access` command to an object before picking up. The `access` command is i=only relevat for assets.\

### **Output Format:** \
Your response must contain **only the list of plan actions** in valid JSON array format, with no additional text, explanation, or formatting. \
Each action must follow this structure: \
["goto(r_5)", "pickup(o_9)", "goto(r_9)", "access(a_36)", "release(o_9)", "close(a_36)"] \
where r_5, o_9, a_36 are valid nodes from the scene graph. Do not include any extra text, markdown, or explanations. Avoid ```json suffixes and ``` post fixes.
"""


planner_prompt = """\
You are an expert in robotic 3D scene graph planning. \
I will provide you with a 3D scene graph in JSON format describing a building and a natural language task. \
Your goal is to generate a step-by-step executable plan to achieve the task using only the allowed actions. \

### **Allowed Actions:** \
You can only use the following predefined actions: \
- `goto(<pose>)`: Move the agent **to an adjacent room**. The adjacency is strictly defined under the `links` section of the scene graph. **You must never move between non-adjacent rooms in a single step.** \
- `access(<asset>)`: Provide access to the set of affordances associated with an **asset node** and its connected objects. **Use `access()` only for assets, not objects.** \
- `pickup(<object>)`: Pick up an accessible object **only after the associated asset has been accessed**. \
- `release(<object>)`: Release a grasped object at an asset node. **Ensure the target asset is accessed before releasing.** \
- `turn_on(<object>)` / `turn_off(<object>)`: Toggle an object at the agent’s current node, if accessible and has an affordance. \
- `open(<asset>)` / `close(<asset>)`: Open or close an asset at the agent’s current node, affecting object accessibility. \

### **Rules and Constraints:** \
1. **Room transitions must be strictly adjacent.** The agent can only move along the adjacency links defined in the scene graph. Multi-step movements are not allowed in a single `goto()` command. Instead, find the shortest valid path using adjacency constraints. \
2. **Plan actions in the correct sequence:** Move → Access → Pickup → Move → Release. Avoid skipping or reordering steps. \
3. **Ensure objects are accessible before interacting:** \
   - Use `access(<asset>)` **before** `pickup(<object>)`. \
   - Use `access(<asset>)` **before** `release(<object>)`. \
   - Never attempt `pickup()` or `release()` without ensuring accessibility first. \
4. **Avoid redundant actions:** \
   - Do not issue `goto(<pose>)` if the agent is already in the target location. \
   - Do not issue `access(<asset>)` if the asset is already accessed. \
5. **If an invalid action is detected, replan accordingly:** \
   - If an invalid `goto(<pose>)` action occurs, verify the adjacency and correct the path. \
   - If an invalid `pickup(<object>)` occurs, check if `access(<asset>)` is missing and fix it. \
   - If an invalid `release(<object>)` occurs, ensure the correct asset is accessed before releasing. \

### **Output Format:** \
Your response must contain **only the list of plan actions** in valid JSON array format, with no additional text, explanation, or formatting. \
Each action must follow this structure: \
["goto(r_5)", "pickup(o_9)", "goto(r_9)", "access(a_36)", "release(o_9)", "close(a_36)"] \
where `r_5`, `o_9`, and `a_36` are valid nodes from the scene graph. **Do not include any extra text, markdown, or explanations.** Avoid `json` suffixes and ` ``` ` post fixes. \
"""

planner_prompt_alt = """\
You are an expert in robotic 3D scene graph planning. \
I will provide you with a 3D scene graph in JSON format describing a building and a natural language task. \
Your goal is to generate a step-by-step executable plan to achieve the task using only the allowed actions. \

### **Allowed Actions:** \
You can only use the following predefined actions: \
- `goto(<pose>)`: Move the agent **to an adjacent room**. The adjacency is strictly defined under the `links` section of the scene graph. **You must never move between non-adjacent rooms in a single step.** \
- `access(<asset>)`: Provide access to the set of affordances associated with an **asset node** and its connected objects. **Use `access()` only for assets, not objects.** \
- `pickup(<object>)`: Pick up an accessible object **only after the associated asset has been accessed**. \
- `release(<object>)`: Release a grasped object at an asset node. **Ensure the target asset is accessed before releasing.** \
- `turn_on(<object>)` / `turn_off(<object>)`: Toggle an object at the agent’s current node, if accessible and has an affordance. \
- `open(<asset>)` / `close(<asset>)`: Open or close an asset at the agent’s current node, affecting object accessibility. \

### **Rules and Constraints:** \
1. **Room transitions must be strictly adjacent.** The agent can only move along the adjacency links defined in the scene graph. Multi-step movements are not allowed in a single `goto()` command. Instead, find the shortest valid path using adjacency constraints. \
2. **Plan actions in the correct sequence:** Move → Access → Pickup → Move → Release. Avoid skipping or reordering steps. \
3. **Ensure objects are accessible before interacting:** \
   - Use `access(<asset>)` **before** `pickup(<object>)`. \
   - Use `access(<asset>)` **before** `release(<object>)`. \
   - Never attempt `pickup()` or `release()` without ensuring accessibility first. \
4. **Avoid redundant actions:** \
   - Do not issue `goto(<pose>)` if the agent is already in the target location. \
   - Do not issue `access(<asset>)` if the asset is already accessed. \
5. **If an invalid action is detected, replan accordingly:** \
   - If an invalid `goto(<pose>)` action occurs, verify the adjacency and correct the path. \
   - If an invalid `pickup(<object>)` occurs, check if `access(<asset>)` is missing and fix it. \
   - If an invalid `release(<object>)` occurs, ensure the correct asset is accessed before releasing. \

### **Output Format:** \
Your response must contain **only the list of plan actions** in valid JSON array format, with no additional text, explanation, or formatting. \
Each action must follow a structure like this: \
["action_1(<parameter>)", "action_2(<parameter>)", "action_3(<parameter>)", "action_4(<parameter>)", "action_5(<parameter>)", "action_6(<parameter>)"] \
Wher (<parameter>) are valid nodes from the scene graph, like rooms, objects or assets.
An example of a plan with a series of actions using a passive asset:
["goto(bedroom_5)", "pickup(book_9)", "goto(corridor_7)", "goto(kitchen_9)", "access(dining table_36)", "release(book_9)"] \
An example of a plan with a series of actions using a openable asset: \
["goto(bedroom_5)", "pickup(bottle_2)", "goto(corridor_7)", "goto(kitchen_9)", "access(refridgerator_10)", "open(refridgerator_10)", "release(bottle_2)", "close(refridgerator_10)"] \
**Do not include any extra text, markdown, or explanations.** Avoid `json` suffixes and ` ``` ` post fixes. \
"""


# planner prompt

graph_pruning_prompt = """\
You are an expert in robotic 3D scene graph planning. \
I will provide you with a 3D scene graph in JSON format describing a building and a natural language task. \
Your goal is to identify the **relevant rooms** involved in the task, specifically: \
- The room where an object needs to be picked up. \
- The room where an object needs to be placed. \
- Any additional room required to access an asset. \

### **Output Format:** \
Return the list of relevant room IDs **only** in JSON array format. \
Each room should appear **only once**, even if multiple actions occur in the same room. \
Do not include any extra text, explanations, or formatting. Avoid `json` prefixes or suffixes. \

### **Example:** \
#### **Input Task:**  
*"Please go to r_5 (bedroom), pick up o_9 (book), and place it on a_36 (dining table) in r_9 (dining room)."*  

#### **Expected Output:**  
["r_5", "r_9"]
Avoid ```json suffixes and ``` post fixes.
"""


graph_pruning_prompt_alt = """\
You are an expert in robotic 3D scene graph planning. \
I will provide you with a 3D scene graph in JSON format describing a building and a natural language task. \
Your goal is to identify the **relevant rooms** involved in the task, specifically: \
- The room where an object needs to be picked up. \
- The room where an object needs to be placed. \
- Any additional room required to access an asset. \

### **Output Format:** \
Return the list of relevant room IDs **only** in JSON array format. \
Each room should appear **only once**, even if multiple actions occur in the same room. \
Do not include any extra text, explanations, or formatting. Avoid `json` prefixes or suffixes. \

### **Example:** \
#### **Input Task:**  
*"Please go to bedroom_5, pick up book_9, and place it on dining table_36 in dining room_9."*  

#### **Expected Output:**  
["bedroom_5", "dining room_9"]
Avoid ```json suffixes and ``` post fixes.
"""