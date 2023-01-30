# Emotion Detection

For a more in depth look at the project you can look at the [report](https://github.com/xaviergenelin/Emotion-Classification/blob/main/FinalProject_Report.pdf) that we wrote.

## Purpose

This project examines human emotion behind a conversation. Most models simplify the classification into a binary (positive, negative) or ternary (postiive, neutral, negative) set. We look to go a step further to classify the conversations into one of five emotions: mad, sad, joyful, powerful, and scared. We examine three different models for this task: a support vector machine, a bidirectional long short-term memory neural network, and a partially retrained Bidirectional Respresentations from Transformers (BERT) model.

## Dependencies
* torch
* pandas
* matplotlib
* sklearn
* transformers 
* numpy

## Dataset Description

We used an Empathetic Dialogue dataset to train and test our model. The data contains 25,000 conversations between a speaker and listener, labeled with one of 32 human emotions. Each label is assigned between 425 and 1200 conversations, with 3 to 8 utterances per conversation. We use the first utterance from each conversation. From the 32 emotions in the dataset, we consolidated them down to our 5 of interest:

| New Label | Original Label                                                       |
|-----------|----------------------------------------------------------------------|
| Mad       | Angry, Jealous, Annoyed, Furious, Disgusted                          |
| Sad       | Sad, Devastated, Ashamed, Guilty,  Disappointed, Embarrassed, Lonely |
| Joyful    | Joyful, Excited, Trusting, Caring,  Grateful, Hopeful, Content       |
| Powerful  | Confident, Prepared, Proud, Faithful                                 |
| Scared    | Anxious, Apprehensive, Terrified, Afraid                             |

Five emotion labels - sentimental, surprised, nostalgic, anticipating, and impressed - didn't translate into our new categories and were dropped from the dataset.

## Machine Learning Methods

We examined three models: a support vector machine, a bidirectional long short-term memory neural network, and a partially retrained Bidirectional Respresentations from Transformers (BERT) model.

### Support Vector Machine

This traditional model we used as our baseline model. We used Stanford's GloVe word embeddings to create a list of 200-dimensional embeddings for each word in the dataset. To address class imbalance, we over-sampled in our training dataset from minority classes. We tuned the model using cross_validation and 12 hyperparemeter combinations:
| Inv. Regularization Strength (C) | Kernel | Gamma | F1    |
|----------------------------------|--------|-------|-------|
| 1                                | Linear | NA    | 0.597 |
| 10                               | Linear | NA    | 0.605 |
| 100                              | Linear | NA    | 0.608 |
| 1000                             | Linear | NA    | 0.610 |
| 1                                | RBF    | .001  | 0.192 |
| 1                                | RBF    | .0001 | 0.081 |
| 10                               | RBF    | .001  | 0.431 |
| 10                               | RBF    | .0001 | 0.192 |
| 100                              | RBF    | .001  | 0.578 |
| 100                              | RBF    | .0001 | 0.432 |
| 1000                             | RBF    | .001  | 0.601 |
| 1000                             | RBF    | .0001 | 0.578 |

The highest performing model achieving a macro-averaged F1 score of 0.610.

### Bi-LSTM Network

We used the same GloVe embeddings as before, but instead of using oversampling to address the class imbalance, we used a weighted cross-entropy to prioritize under-represented classes. The model consisted of an embedding layer, a bi-directional bi-LSTM with 2 layers each with 512 hidden units per direction, and a 2-layer fully-connected network:

![](https://github.com/xaviergenelin/Emotion-Classification/blob/main/images/lstm-architecture.png)

We used our validation set to optimize 8 sets of hyperparameters:

| Epochs | Learning Rate | LSTM Layers | Dropout | F1   |
|--------|---------------|-------------|---------|------|
| 5      | .001          | 2           | .1      | .717 |
| 5      | .001          | 2           | .2      | .721 |
| 14     | .0001         | 2           | .3      | .708 |
| 7      | .0001         | 3           | .3      | .705 |
| 10     | .01           | 3           | .2      | .695 |
| 11     | .001          | 3           | .3      | .689 |
| 9      | .001          | 3           | .4      | .700 |
| 8      | .001          | 3           | .5      | .699 |

All models were trained for 25 epochs but the reported score is for the best-performing epoch.

### BERT Transformers

The model consists of a 768 layer word embedding layer, 11 encoder layers, and a pooler layer. We used both the "bert-base-uncased" model and "twitter-roberta-base-sentiment" models. We trained 5 different models compounding the follinwg additional layers for retraining:

| Model   | Layers  (Re-)Trained | Validation  Macro F1 |
|---------|----------------------|----------------------|
| BERT    | 1                    | 0.433                |
| BERT    | 2                    | 0.680                |
| BERT    | 3                    | 0.778                |
| BERT    | 4                    | 0.787                |
| BERT    | 5                    | 0.798                |
| RoBERTa | 1                    | 0.250                |
| RoBERTa | 2                    | 0.342                |
| RoBERTa | 3                    | 0.484                |
| RoBERTa | 4                    | 0.547                |
| RoBERTa | 5                    | 0.553                |

Each model was trained for 25 epochs and validated using our held-out validation set.

## Results

We retrained each of the combined trainining and validation sets. Each model was trained in the same way as its best performing precendent during tuning yieliding the following results:
| Model    | Test Macro-Averaged F1 Score |
|----------|------------------------------|
| BERT     | 0.813                        |
| LSTM-CNN | 0.707                        |
| SVC      | 0.613                        |
