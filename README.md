# Inference-time Policy Adapter (IPA)

This is the official repo for the paper ["Inference-Time Policy Adapters (IPA): Tailoring Extreme-Scale LMs without Fine-tuning"](https://arxiv.org/abs/2305.15065) (EMNLP 2023)

## Requirement
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `quark` with:
```
conda env create -f environment.yml
```

## Instruction
The ``main`` branch contains policy adapter for **constrained generation** task. We put the other three tasks, detoxification, open-ended generation and dialogue generation, in the ``toxicity``, ``open_ended`` and ``dialogue`` branch separately. 

## Citation
If you use this codebase in your work, please consider citing our paper:
```
@article{Lu2023InferenceTimePA,
  title={Inference-Time Policy Adapters (IPA): Tailoring Extreme-Scale LMs without Fine-tuning},
  author={Ximing Lu and Faeze Brahman and Peter West and Jaehun Jang and Khyathi Raghavi Chandu and Abhilasha Ravichander and Lianhui Qin and Prithviraj Ammanabrolu and Liwei Jiang and Sahana Ramnath and Nouha Dziri and Jillian R. Fisher and Bill Yuchen Lin and Skyler Hallinan and Xiang Ren and Sean Welleck and Yejin Choi},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.15065},
  url={https://api.semanticscholar.org/CorpusID:258865629}
}
```