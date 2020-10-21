# Veiled Toxicity Detection
This repo contains code for our paper Fortifying Toxic Speech Detectors Against Veiled Toxicity (https://arxiv.org/abs/2010.03154).

### Setup
The main environment requirements for this project are `python 3.6`, `pytorch 1.3.1`, and `pytorch-pretrained-bert 0.6.1`.

### Data processing
We mainly use three sources of data (as describe in our paper): the Social Bias Inference Corpus [1], a random crawl of Reddit, and queries to the Perspective API [2].

- `resources/process_SBF.ipynb`: filters the SBIC dataset by their offensiveness and Perspective API toxicity scores (described in the paper).

- `resources/processed_dataset/*.pkl`: pickle files containing the processed datasets (format: `List` of `(original_text, (target_group_annotations))`)

### Training the initial student model

- `model/*-aws-run-bert.sh`: trains a base classifier; this classifier approximates the behavior of Perspective API on our processed datasets.

### Applying different influence metrics

- `bert_embin.py`: implements the embedding similarity metric.

- `bert_influence.py`: implements the influence functions metric.

- `bert_trackin.py`: implements the gradient product (TrackIn) metric.

- `*-aws-run-inf.sh`: calculates the influence of each training example on each probing example using different influence metrics.

### Training the fortified model

- `analysis.ipynb`: visualizes influence as presented in the paper, and marks data to relabel.

- `model/*-aws-correct-bert.sh`: trains an improved classifier with re-labeled influential training examples.



If you have any questions, please email Xiaochuang Han at xiaochuang.han@gmail.com. Thank you!

[1] Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A Smith & Yejin Choi. Social Bias Frames: Reasoning about Social and Power Implications of Language. ACL (2020)

[2] https://www.perspectiveapi.com/#/home
