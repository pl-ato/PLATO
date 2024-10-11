# Mastering Scene Rearrangement with Expert-assisted Curriculum Learning and Adaptive Trade-Off Tree-Search

[Zan Wang*](https://silvester.wang/),
[Hanqing Wan*](https://hanqingwangai.github.io/),
[Wei Liang](https://liangwei-bit.github.io/web/)

> Code for the paper "Mastering Scene Rearrangement with Expert-assisted Curriculum Learning and Adaptive Trade-Off Tree-Search (IROS 2024)".

## Abstarct

Scene Rearrangement Planning (SRP) has recently emerged as a crucial interior scene task; however, current approaches still face two primary issues. First, prior works define the action space of SRP using handcrafted coarse-grained actions, which are inflexible for scene arrangement transition and impractical for real-world deployment. Secondly, the scarcity of realistic indoor scene rearrangement data hinders popular data-hungry learning approaches and quantitative evaluation. To tackle these issues, we propose a fine-grained action space definition and curate a large-scale scene rearrangement dataset to facilitate the training of learning approaches and comprehensive benchmarking. Building upon this dataset, we introduce a novel framework, PLATO, designed for efficient agent training and inference. Our approach features an exPert-assisted curriculum Learning (PL) paradigm that possesses a Behavior Cloning (BC) and an offline Reinforcement Learning (RL) curriculum for agent training, along with an advanced tree-search-based planner enhanced by an Adaptive Trade-Off (ATO) strategy to improve expert agent performance further. We demonstrate the superior performance of our method over baseline agents through extensive experiments and provide a detailed analysis to elucidate its rationale.

## Instructions

> We currently only release the code of SRP environment. Please check the code.

- Download data from [OneDrive](https://1drv.ms/f/c/a3c8b9329182f3c6/EgwV7Q3qvDdEjMUe_T_D7bYBefoO_vpB0M5C3L6MZCNoHw?e=ZGSxjo)

- Create a python environment and install `pybind11`, `pillow`, `pytorch`, and `numpy`.

- Enter `env/build` and execute `make` to build the SRP environment.

- Run `python greedy.py` to test the environment.

## Citation

If you find our project useful, please consider citing us:

```bibtex
@inproceedings{wang2024mastering,
  title={Mastering Scene Rearrangement with Expert-assisted Curriculum Learning and Adaptive Trade-Off Tree-Search},
  author={Wang, Zan and Wang, Hanqing and Liang, Wei},
  booktitle={International Conference on Intelligent Robots and Systems (IROS)},
  year={2024}
}
```
