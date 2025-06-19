# LangSpace_reasoning

## Abstract

This project explores spatial reasoning with large language models (LLMs) over
3D scene graphs to enable high-level task planning from natural language instruc-
tions. Previous works, such as SayPlan [1] and SayCan [2], have demonstrated the
potential of LLMs in robotic planning, but rely on frontier models like ChatGPT-4,
which are computationally expensive and require internet access. In this work, I
investigate fine-tuning open-source LLMs (Llama3 1/3/8B [3]) for task planning
over structured scene representations. I build my own 3D scene graph task planning
system to assess alternative inference strategies, such as single-shot and iterative
multi-prompting and to generate training data for SFT and PPO fine tuning. The
goal is to determine whether a fine-tuned, locally deployable model can achieve
planning performance comparable to or exceeding that of general-purpose frontier
LLMs while improving efficiency. I will present experimental results evaluat-
ing the impact of representation choices, prompting techniques, and fine-tuning
configurations on plan quality and execution feasibility.

Check LangSpace_reasoning.pdf for more details.
