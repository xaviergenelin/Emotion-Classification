# Emotion-Classification

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

This traditional model we used as our baseline model. We used Stanford's GloVe word embeddings to create a list of 200-dimensional embeddings for each word in the dataset. We tuned the model using cross_validation and 

### Bi-LSTM Network

### BERT Transformers

## Results
