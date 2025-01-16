# Aya Dataset

### Paper

Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning
https://aclanthology.org/2024.acl-long.620/

The `Aya Dataset` is a multilingual instruction fine-tuning dataset curated by an open-science community via [Aya Annotation Platform](https://aya.for.ai/) from Cohere For AI. The dataset contains a total of 204k human-annotated prompt-completion pairs along with the demographics data of the annotators.

This dataset can be used to train, finetune, and evaluate multilingual LLMs.

Homepage: https://huggingface.co/datasets/CohereForAI/aya_dataset


### Citation

```
@misc{singh2024aya,
      title={Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning}, 
      author={Shivalika Singh and Freddie Vargus and Daniel Dsouza and Börje F. Karlsson and Abinaya Mahendiran and Wei-Yin Ko and Herumb Shandilya and Jay Patel and Deividas Mataciunas and Laura OMahony and Mike Zhang and Ramith Hettiarachchi and Joseph Wilson and Marina Machado and Luisa Souza Moura and Dominik Krzemiński and Hakimeh Fadaei and Irem Ergün and Ifeoma Okoh and Aisha Alaagib and Oshan Mudannayake and Zaid Alyafeai and Vu Minh Chien and Sebastian Ruder and Surya Guthikonda and Emad A. Alghamdi and Sebastian Gehrmann and Niklas Muennighoff and Max Bartolo and Julia Kreutzer and Ahmet Üstün and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2402.06619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `aya_dataset`

#### Tasks

* `aya_dataset`: measure perplexity on the Aya dataset, via rolling loglikelihoods.

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
