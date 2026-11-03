# Source feasibility

Checked 2026-10-17 against the papers, repos, model cards, and FCC documents named in the comparison sheet. The sheet was not edited. A row is "runnable" only when a small team can get code, a sim or dataset, a research-usable license, and either one 24GB GPU or an API.

## What you can run

**Cooperative driving, lightest path.** [CoDrivingLLM](https://github.com/FanGShiYuu/CoDrivingLLM) is highway-env plus a GPT-4o-mini key. The paper includes unequipped human drivers that do not share intent. The repo has no LICENSE file. GPT-4o-mini’s API is scheduled to shut down on 14 January 2027.

**Cooperative driving, only Apache-licensed closed-loop repo.** [CoLMDriver](https://github.com/cxliu0314/CoLMDriver) is Apache-2.0, CARLA 0.9.10.1, InternVL2-4B plus Qwen2.5-3B. The documented launch puts the renderer, the vision model, and the negotiator on separate CUDA devices. Qwen2.5-3B itself is under the Qwen Research License (non-commercial), not Apache-2.0.

**One 24GB GPU, self-hosted brain.** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) is Apache-2.0, 7.61B, and has a vision sibling if you add images later. [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) is MIT, 3.8B, 128K context. Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, and Mistral-7B-Instruct-v0.3 also fit.

**Technique donors with code.** [DiLu](https://github.com/PJLab-ADG/DiLu) (Apache-2.0, highway-env, memory and reflection, 5–10 s per decision). [LeapAD](https://github.com/PJLab-ADG/LeapAD) (Apache-2.0, slow teacher / fast 1.8B student, CARLA 0.9.15). [LMDrive](https://github.com/opendilab/LMDrive) (Apache-2.0, the actual CARLA LangAuto benchmark, released weights; training from scratch is about 2–3 days per stage on 8 A100 80GB GPUs).

## What not to build on

- **AgentsCoDriver.** The paper is real. No public repo.
- **Talk2Drive.** Field study on a 2019 Lexus with Autoware. The repo is speech-to-GPT glue, not a simulator. GPT-4 response time in the paper is about 1.6 s. Takeover reductions of 75.9% and 65.2% are in the paper.
- **GPT-Driver.** Archived repo, no license, open-loop nuScenes, GPT-3.5 fine-tune scheduled to shut down 23 October 2026. Workshop paper, not a main-conference paper.
- **DriveGPT4.** RA-L 2024, LLaMA 2, BDD-X video. Training code was still unreleased on 28 January 2026.
- **DriveVLM.** CoRL 2024, Qwen-VL, nuScenes plus a private SUP-AD set. The GitHub repo is a project page. It is not the LangAuto benchmark.
- **EMMA.** TMLR 2025, Gemini, no official code. OpenEMMA is a different project.

LangCoop and TalkingVehiclesGym have public CARLA code and no LICENSE file. Cite Talking Vehicles as CoopReflect, AAMAS 2026, not CoRL 2025. V2V-LLM and V2V-GoT answer questions on recorded logs. They do not drive a simulator closed-loop. V2V-LLM was trained on 8 NVIDIA A100-80GB GPUs.

## Models and APIs to treat carefully

Qwen1.5-1.8B is the Tongyi Qianwen Research License, non-commercial only. The sheet calls it Apache-2.0. The same non-commercial limit is on Qwen2.5-3B-Instruct. Qwen2.5-7B-Instruct and the other Qwen2.5 sizes except 3B and 72B are Apache-2.0.

Qwen2.5-VL-7B is not LeapAD’s scene model. LeapAD uses Qwen-VL-7B. DeepSeek-R1-Distill-Qwen-7B fits the GPU and is MIT on that repo; the card tells you to force a long `<think>` trace, which fights a driving deadline. DistilGPT2’s 0.14–0.45 s times are NVIDIA A40 measurements from TalkingVehiclesGym, and its context is 1,024 tokens.

Published standard prices match the sheet: GPT-4o-mini $0.15 / $0.60, GPT-4o $2.50 / $10, Gemini 2.5 Flash-Lite $0.10 / $0.40, Gemini 2.5 Flash $0.30 / $2.50, Claude Haiku 4.5 $1 / $5, Claude Sonnet 4.6 $3 / $15. GPT-4o and GPT-4o-mini shut down on 14 January 2027. The May 2024 GPT-4o snapshot (`gpt-4o-2024-05-13`) shuts down on 23 October 2026 and is priced at $5 / $15. Gemini 2.5 Flash and Flash-Lite stay up for accounts that have already used them; Google points new projects at newer models. CoDrivingLLM’s base model is GPT-4o mini, not an older GPT-4.

## V2X citations

The 11 February 2025 effective date is right. The order is **FCC 24-123**, not FCC 24-106. FCC 24-106 is a notice about broadband data caps. FCC 20-164 is the right 2020 order for the 45 / 30 MHz split. DSRC operation in the upper 30 MHz runs through 14 December 2026.

Do not cite Boualouache and Engel (arXiv 2201.10500, IEEE COMST 2023) for the claim that V2X needs a full operator 5G core. The paper says PC5 sidelink V2V bypasses the cellular core.

The PMC row is a real paper: Miao, Virtusio, and Hua, Sensors, 27 January 2021. It reports a Taiwan trial, not an OBU price survey. 5GAA says direct 5G NR-V2X is not the North American industry profile. The FCC order does not ban it.
