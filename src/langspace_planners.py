import json
import multiprocessing
import time

def invoke_llm_with_timeout(llm, plan_gen_message, timeout=10):
    """
    Runs LLM invocation in a separate process to enforce a strict timeout.
    If it doesn't complete within `timeout` seconds, it gets killed.
    """
    def target(output):
        """Worker function to call LLM invoke and store the result."""
        try:
            response = llm.invoke(plan_gen_message)
            output.put(response.content.strip())  # Store result in queue
        except Exception as e:
            output.put(f"ERROR: {str(e)}")  # Store error in queue

    output = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(output,))
    process.start()
    process.join(timeout)  # Wait for the process to finish or timeout

    if process.is_alive():
        print("⏳ LLM invocation timed out. Terminating...")
        process.terminate()  # Kill the process forcefully
        process.join()
        return None  # Return None to indicate timeout

    result = output.get()  # Retrieve the response
    if result.startswith("ERROR:"):
        print(f"❌ LLM Invocation Error: {result}")
        return None  # Handle errors gracefully
    return result

def generate_plan_feedback_timeout(task_description, scenegraph, llm, planner_prompt, max_retries=3, timeout=10):
    """
    Generates a valid plan using an iterative feedback loop.
    Retries up to `max_retries` times if the plan is invalid or times out.
    """
    
    json_graph = json.dumps(scenegraph.get_scene_graph(), indent=4)

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} to generate a valid plan...")

        # Create the initial plan request
        plan_gen_message = [
            ("system", planner_prompt),
            ("human", "The 3D scene graph in JSON format:"),
            ("human", json_graph),
            ("human", "The task description:"),
            ("human", task_description),
        ]

        # Invoke LLM with strict timeout
        generated_plan = invoke_llm_with_timeout(llm, plan_gen_message, timeout=timeout)
        
        if generated_plan is None:
            print(f"⚠️ Skipping attempt {attempt} due to timeout or LLM error.")
            continue  # Skip to the next retry

        try:
            plan = json.loads(generated_plan)  # Parse JSON response
        except json.JSONDecodeError:
            print("❌ JSON decoding failed. Retrying...")
            continue  # Retry with a new generation

        # Execute the plan and get validation feedback
        result = scenegraph.execute_plan(plan)

        if result == True:
            print(f"✅ Plan is valid on attempt {attempt}: {plan}")
            return plan  # Return valid plan
        
        # If invalid, prepare feedback for re-planning
        print(f"❌ Plan failed on attempt {attempt}: {result}")

        # Generate feedback message
        feedback_message = f"The generated plan was invalid due to: {result}. Please revise the plan accordingly."

        # Update the prompt to include feedback
        plan_gen_message.append(("human", feedback_message))

    print("❌ Failed to generate a valid plan after multiple attempts.")
    return None  # Return None if no valid plan was found


import json
import copy
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_fixed


# graph pruner

def prune_scenegraph(scenegraph, task_description, llm, graph_pruning_prompt):


    json_graph = json.dumps(scenegraph.get_scene_graph(), indent=4)

    scenegraph_pruner_gen_message = [
        (
            "system",
            graph_pruning_prompt,
        ),
        
        ("human", "The 3D scene graph in JSON format:"),
        ("human", json_graph),
        ("human", "The task description:"),
        ("human", task_description),
    ]

    scenegraph_pruner_reply = llm.invoke(scenegraph_pruner_gen_message).content
    #pruned_scenegraph = scenegraph_pruner_reply.content
    scenegraph_pruner_steps = json.loads(scenegraph_pruner_reply)
    print(scenegraph_pruner_steps)

    for room in scenegraph_pruner_steps:
        scenegraph.subset_expand_node(room)

    pruned_graph = scenegraph.get_scene_graph_subset()
    json_pruned_graph = json.dumps(pruned_graph, indent=4)

    return json_pruned_graph




import json
import copy
from tenacity import retry, stop_after_attempt, wait_fixed

def generate_plan_feedback(instruction, scenegraph, json_graph, room_graph, llm, planner_prompt, max_retries=3):
    """
    Generates a valid plan using an iterative feedback loop.
    Retries up to `max_retries` times if the plan is invalid.
    """
    #print(instruction["instruction"])

    # Initialize message history outside the loop
    plan_gen_message = [
        ("system", planner_prompt),
        ("system", "The room adjacencies to strictly follow:"),
        ("system", room_graph),
        #("system", "In case of a failure, please revise the plan according to the error messages to aid replanning."),
        ("human", "The 3D scene graph in JSON format:"),
        ("human", json_graph),
        ("human", "The task description:"),
        ("human", instruction["instruction"]),
    ]

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} to generate a valid plan...")

        # Invoke LLM to generate plan
        plan_gen_llm_reply = llm.invoke(plan_gen_message)
        generated_plan = plan_gen_llm_reply.content.strip()

        try:
            plan = json.loads(generated_plan)  # Parse JSON
        except json.JSONDecodeError:
            print("❌ JSON decoding failed. Retrying...")
            plan_gen_message.append(("human", "The previous response was not valid JSON. Ensure the response is a JSON array of actions."))
            continue  # Retry with a new generation

        # Execute plan and get validation feedback
        exec_result = scenegraph.execute_plan(plan)
        validate_result = scenegraph.check_objects_classes(instruction["verifying_step"])

        print("Results:")
        print("  >", exec_result)
        print("  >", validate_result)

        if exec_result == "Valid plan." and validate_result == "Valid plan.":
            print(f"✅ Plan is valid on attempt {attempt}: {plan}")
            return plan  # Return valid plan
        
        # If invalid, append both the plan and feedback to history
        print(f"❌ Plan failed on attempt {attempt}: {exec_result}")

        if exec_result == "Valid plan.":
            feedback_message = f"The actions has not be executed accirding to the instructions: {validate_result}. Please revise the plan accordingly."
        else:
            feedback_message = f"The previous plan was invalid due to: {exec_result}. Here is the generated plan:\n{generated_plan}\nRevise the plan to fix the issues and try again."

        plan_gen_message.append(("human", feedback_message))  # Append feedback

        print("plan_gen_message:", plan_gen_message)

    print("❌ Failed to generate a valid plan after multiple attempts.")
    return None  # Return None if no valid plan was found








import tiktoken

def count_tokens(messages, model="gpt-4"):
    """Counts the number of tokens in a LangChain-style messages list."""
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    
    for role, content in messages:
        total_tokens += len(encoding.encode(content))  # Count tokens in message content
    
    return total_tokens


import re
import json

def clean_json_response(response_text):
    """Removes ```json and ``` formatting from the response."""
    return re.sub(r"```json\s*|\s*```", "", response_text).strip()


import asyncio
import nest_asyncio

nest_asyncio.apply() 

# Allow asyncio to run inside Jupyter
nest_asyncio.apply()

async def invoke_with_timeout(llm, message, timeout=10):
    """Runs LLM invocation asynchronously with a strict timeout."""
    try:
        return await asyncio.wait_for(llm.agenerate(messages=[message]), timeout=timeout)
    except asyncio.TimeoutError:
        print("⏳ LLM invocation timed out. Skipping...")
        return None

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

async def generate_plan_feedback_gemini_timeout(instruction, scenegraph, json_graph, room_graph, llm, planner_prompt, max_retries=3):
    """
    Generates a valid plan using an iterative feedback loop.
    Retries up to `max_retries` times if the plan is invalid.
    """
    #print(instruction["instruction"])

    task_description = instruction["instruction"]


    # Initialize message history outside the loop
    plan_gen_message = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content="The room adjacencies to strictly follow:"),
        HumanMessage(content=room_graph),
        HumanMessage(content="The 3D scene graph in JSON format:"),
        HumanMessage(content=json_graph),
        HumanMessage(content="The task description:"),
        HumanMessage(content=task_description)
    ]

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt} to generate a valid plan...")

        # Invoke LLM to generate plan
        #plan_gen_llm_reply = llm.invoke(plan_gen_message)
        #generated_plan = plan_gen_llm_reply.content.strip()

        plan_gen_llm_reply = await invoke_with_timeout(llm, plan_gen_message, timeout=15)

        if plan_gen_llm_reply is not None:

            generated_plan = plan_gen_llm_reply.generations[0][0].text
            generated_plan = clean_json_response(generated_plan)
            plan = json.loads(generated_plan)

            try:
                plan = json.loads(generated_plan)  # Parse JSON
            except json.JSONDecodeError:
                print("❌ JSON decoding failed. Retrying...")
                plan_gen_message.append(HumanMessage(content="The previous response was not valid JSON. Ensure the response is a JSON array of actions."))
                continue  # Retry with a new generation

            # Execute plan and get validation feedback
            exec_result = scenegraph.execute_plan(plan)
            validate_result = scenegraph.check_objects_classes(instruction["verifying_step"])

            print("Results:")
            print("  >", exec_result)
            print("  >", validate_result)

            if exec_result == "Valid plan." and validate_result == "Valid plan.":
                print(f"✅ Plan is valid on attempt {attempt}: {plan}")
                return plan  # Return valid plan
            
            # If invalid, append both the plan and feedback to history
            print(f"❌ Plan failed on attempt {attempt}: {exec_result}")

            if exec_result == "Valid plan.":
                feedback_message = f"The actions has not be executed accirding to the instructions: {validate_result}. Please revise the plan accordingly."
            else:
                feedback_message = f"The previous plan was invalid due to: {exec_result}. Here is the generated plan:\n{generated_plan}\nRevise the plan to fix the issues and try again."

            plan_gen_message.append(HumanMessage(content=feedback_message))  # Append feedback

            print("plan_gen_message:", plan_gen_message)



    print("❌ Failed to generate a valid plan after multiple attempts.")
    return None  # Return None if no valid plan was found













