# Evaluating the Impact of Positional Encoding in Transformer Decoder and Encoder Architecture


![GitHub](https://img.shields.io/github/license/Sinanmz/TransformerPosEnc)
![Python](https://img.shields.io/badge/Python-3.9-blue)

## Table of Contents

- [Introduction](#introduction)
- [Method](#method)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Expected Results](#expected-results)
- [Results](#results)
- [Future Project Ideas](#future-project-ideas)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)

## Introduction

Transformer architecture has taken the world of NLP and Deep Learning by storm since it was introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), in 2017. Before Transformers, best performing NLP models were RNNs and their variants such as LSTM, GRU, etc. The success of RNNs in NLP tasks is due to their structure of not being Bag of Words (BoW) models and receiving the tokens one by one, thus having some sort of positional awareness. Even though RNNs performed better than BoW models at the time, they have pretty serious flaws. For example, their short-term memory isn't that great (Even though LSTMs have longer short-term memory, it's just not enough), training RNNs is not that computationally efficient since those models have to receive the tokens one by one, thus preventing parallelizability of computations which GPUs take advantage of, they're not able to utilize transfer learning, the problem of vanishing gradients and exploding gradients is pretty serious in RNNs at training time, etc. 


Transformer Decoder and Encoders are Bag of Words models, which rely on self-attention. Self-attention is a mechanism that allows the input tokens to interact with each other. By applying self-attention, we get representations for each of our input tokens that capture not only that specific token's embedding but also the context in which it was given to the model. The amount that each input token pays attention to any other token is variable and can be learned throughout the training phase. Transformer Encoder and Decoders, utilize slightly different kinds of self-attention. Encoders use unmasked self-attention, meaning each token is free to pay attention to any other token, while Decoders use masked attention, meaning we force each token to pay attention to itself and the tokens that come before it.


Since in self-attention, each token can freely pay attention to other tokens regardless of how far it is from that token, these models don't have the problem of poor short-term memory. Because Transformers are BoW models we don't face RNN training issues of not utilizing parallelizibility. Also, it's pretty common to use transfer learning when training a Transformer model. 


Now, what about the positional awareness problem of Bag of Words models? In the paper "Attention Is All You Need" it was noted that before feeding our input tokens to the model, we add positional encoding vectors to token embeddings to get positional embedding vectors, then feed those to the model instead. In the paper, positional encoding vectors didn't have learnable parameters and were merely sine and cosine waves with different frequencies. Even though those vectors in fact gave the model some positional awareness, it's pretty common to use positional encoding vectors with learnable parameters.


In this small project, I attempt to figure out how much positional encoding actually helps Transformer models for the task of text classification.


<br>


## Method
For this project, I've used the PyTorch framework for building and training my models from scratch. For my models, I wrote 6 models in total, 3 Transformer Decoders, and 3 Transformer Encoders. In each Decoder and Encoder group, I wrote 1 model without positional encoding, 1 model with positional encoding with learnable parameters that are learned throughout the training phase, and 1 model with positional encoding using sine and cosine waves without learnable parameters (like the method that was suggested in the paper "Attention Is All You Need"). My Encoder and Decoder models each consist of 8 stacked self-attention blocks, each of which consisting of multi-headed self-attention layers, layer-normalization layers, MLP layers, and residual connections. Number of heads in each self-attention layer 8.

For the tokenizer, I've used the "bert-base-uncased" tokenizer through the Hugging Face library. Each token has an embedding vector of size 512 and it is learned throughout the training phase. Positional encoding vectors are of the same size as token embedding vectors. 

The task that I chose for this project is sentiment analysis. Classification is done by feeding the final representation of the last non-padding token of each input sequence through a feed-forward layer and performing softmax on the result. The last non-padding token in each input sequence is [CLS] special token (token id = 101) and in all the models this token is not going to receive positional encoding. Each model is going to be trained on the training data for 10 epochs. Before each epoch, in the middle of it, and at the end of the last epoch, the training loss, test loss, training accuracy, and test accuracy is going to be measured.


<br>

## Dataset
For the dataset, I chose the "glue-sst2" dataset. I wrote a Python script named `data_preprocessor.py` which downloads the dataset through the datasets library, tokenizes it, puts the [CLS] token at the end of each sample sequence of tokens, replaces the tokens in the "bert-base-uncased" vocabulary that aren't in the training data with [UNK] token(by doing this our vocabulary gets down to 11573 tokens), and saves the sequences of token ids, attention masks, labels, and vocabulary size in the preprocessed_data directory.


<br>

## Model Architecture
If you want to learn about the model architectures, you can find the information in the 'models.py' script which contains the code for all the 6 models used for this project.

<br>

## Training
The training was done using [Google Colab](https://colab.research.google.com/) and the GPUs provided by this service. You can find the notebooks which contain the code for loading the training data, batching them, and training the models in the training_notebooks directory. Also, the models are saved at each checkpoint in the checkpoints directory.

<br>

## Evaluation
After training was done, each model at each checkpoint was loaded and the training losses, test losses, training accuracies, and test accuracies were measured and saved in the Evaluation directory. You can find the notebooks containing the note for this part of the project in the eval_notebooks directory.

Once all the data is gathered, the final evaluation is done by plotting the losses and accuracies of each model at each checkpoint. You can find the code for this part in Eval.ipynb in the Evaluation directory.

<br>

## Expected Results
Since for this specific task of text classification bidirectional relationship between tokens is pretty valuable, I expect the encoders to perform better than decoders in general, that's actually the reason why bidirectional LSTMs perform better than unidirectional LSMTs for this task. As for the goal of this project, I expect positional encoding to have a noticeable impact on the performance of the models utilizing it.

<br>

## Results

### Losses
Here is the plot of the losses of our 6 models at ever checkpoint:
<img title="Losses" src="plots\losses.png">

Here is the plot of the test losses of our 6 models that are divided into encoder and decoder groups:
<img title="Test Losses Based on Group" alt="Alt text" src="plots\group_losses.png">

### Accuracies
Here is the plot of the accuracies of our 6 models at ever checkpoint:
<img title="Accuracies"  src="plots\accuracies.png">

Here is the plot of the test accuracies of our 6 models that are divided into encoder and decoder groups:
<img title="Test Accuracies Based on Group" alt="Alt text" src="plots\group_accuracies.png">

### Overal Results:
<div align="center">

| Model                                   	| Best Test Loss Value 	| Best Test Accuracy    	|
|-----------------------------------------	|----------------------	|-----------------------	|
| Encoder W/O Pos Enc                     	| 0.579 (at epoch 2.5) 	| 0.811 (at epoch 5.5)  	|
| Encoder W/ Pos Enc W/ Learnable Params  	| 0.576 (at epoch 2.0) 	| 0.804 (at epoch 10.0) 	|
| Encoder W/ Pos Enc W/O Learnable Params 	| 0.454 (at epoch 5.5) 	| 0.808 (at epoch 10.0) 	|
| Decoder W/O Pos Enc                     	| 0.546 (at epoch 2.0) 	| 0.805 (at epoch 6.0)  	|
| Decoder W/ Pos Enc W/ Learnable Params  	| 0.574 (at epoch 2.5) 	| 0.802 (at epoch 7.5)  	|
| Decoder W/ Pos Enc W/O Learnable Params 	| 0.462 (at epoch 8.0) 	| 0.813 (at epoch 8.0)  	|

</div>

### My Take on the Results
As we can see in the table above, the encoders didn't outperform the decoders. I don't exactly know the reason why that's the case here and should do more research on that.
If we pay attention to positional encoding, it's obvious that the models with positional encoding same as the one mentioned in the paper "Attention Is All You Need" clearly performed better than the other models and have loss values that are significantly lower than the other models. Although their accuracy isn't that much different. This means that these models are more confident when they're predicting the right label. And another thing that's interesting about these models is that they tend to overfit the training data after more epochs than the other models.   
As for the models with positional encoding with learnable parameters, these models did not perform as expected and weren't any better than the ones without positional encoding. I don't know the exact reason, but I guess for them to really perform better, we should train them on a larger amount of data.

<br>

## Future Project Ideas
Now, that we observed how much positional encoding helps Transformer Encoders and Decoders, it's interesting to find out how much it helps the other Bag of Words models. Specifically, I'm interested to see how much it helps the 1D-Convolution method for NLP tasks.
Also, it would be fun to find out exactly how much pretraining a Transformer model helps the performance.

<br>

## Acknowledgments
[This repository](https://github.com/karpathy/ng-video-lecture) by Andrej Karpathy helped a lot in constructing my models from scratch.

<br>

## Contributing
If you want to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request, explaining your changes in detail.
