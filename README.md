# Walking on Unknown Paths: Probing the Trade-Off between In- and Cross-Topic Generalization
This repository includes the code for running the experiments to probe in the In- and Cross-Topic experimental setting.

Further details can be found in our publication _Walking on Unknown Paths: Probing the Trade-Off between In- and Cross-Topic Generalization_


> **Abstract:** Pre-trained Language Models (PLMs) show impressive success when learning downstream tasks.
However, large training data is often unavailable or does not exist, for example, when evaluating PLMs for unseen topics.
Using out-of-distribution experiments like the Cross-Topic setup, we can emulate such future scenarios by holding out instances belonging to distinct topics.
Often we see apparent performance gaps between such setups and commonly used ones where we randomly choose instances to train and evaluate (In-Topic).
Further, results show that we can not transfer the superior of specific PLMs from one setup to another.
Therefore, we propose a probing-based evaluation framework to analyze this trade-off between In- and Cross-Topic.
By analyzing different PLMs, we gain crucial insights into why this trade-off arises, how PLMs differ, and how they evolve during fine-tuning in these setups.
With these findings in mind, we can better understand PLMs and build ground to select appropriate ones for a given situation.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
**(change this as needed!)**

* `data/` -- directory for the data
* `probes/` -- directory for the data
* `src` -- contains all necessary python files 

## Requirements

This repository requires Python3.6 or higher, further requirements can be found in the requirements.txt. 
Further, it need the spacy model `en_core_web_sm`.
For now, this repository requires a running MLFLOW instance for reporting and dropbox as storage for computed models. Please define the corresponding URL and dropbox auth-token in the file `src/defs/default_config.py`

## Data

We make use of the _UKP ArgMin_ dataset [Stab et al. 2018](https://aclanthology.org/D18-1402) and the _WTWT_ dataset [Conforti et al. 2020](conforti-etal-2020-will).
Once you have obtained both datasets, put them in the `folder` folder.

## Generate probing tasks
To generate the Cross-Topic probing task run the following commands:

```
$ parse_task_cross_topic.py --task ukp-argmin 
$ parse_task_cross_topic.py --task wtwt
```

To generate the Cross-Topic topic information classification tasks run the following commands:

```
$ parse_task_cross_topic_tokens.py --task ukp-argmin 
$ parse_task_cross_topic_tokens.py --task wtwt
```

To generate the In-Topic probing tasks run the following commands:

```
$ convert_cross_topic_tasks.py
```

To generate the control tasks run the follwing commands:

```
$ convert_control_task.py
```

## Experiments I

To run the first experiments one a model - like `bert-base-uncased` - run the following command

```
$ run-probes.py --task ukp-argmin --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task ukp-argmin-in --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task wtwt --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
$ run-probes.py --task wtwt-in --model bert-base-uncased --seeds 0,1,2 --out_filter token-types --amnesic False
```

## Experiments 2

To run the second experiments one a model - like `bert-base-uncased` - run the following command to get fit the amnesic probe

```
$ run-probes.py --task ukp-argmin --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task ukp-argmin-in --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task wtwt --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
$ run-probes.py --task wtwt-in --model bert-base-uncased --seeds 0,1,2 --in_filter token-types --amnesic True
```

Afterwards we can run a probes again without topic information, for example NER for Cross-Topic on _ukp-argmin_

```
$ run-probes-without-property.py --amnesic_experiment probes-amnesic-ukp-argmin-token-types-40-cls-topic-maj --target_experiment probes-ukp-argmin-ner
```

## Experiments 3

To run the third experiments a fine-tuned model (fold=0, seed=0) - like `bert-base-uncased-ft-ukp-argmin-0-0` - run the following command. Note you need to provide the fine-tuned model as a folder in running directory

```
$ evolution-dropbox.py --task ukp-argmin --model_name bert-base-uncased-ft-ukp-argmin-0-0 --seed 0 --fold 0 
```
