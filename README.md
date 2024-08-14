# Transformer keras (Implemented by me)
## This is just an untrained implementation for the transformer from "Attention is all you need paper" as it needs high resources to train the model
![image](https://github.com/user-attachments/assets/8192be0b-0f85-4efa-a49f-2ef8362cdace)
![image](https://github.com/user-attachments/assets/7f5080cb-be32-401b-a630-470aadad08f2)


# Transformer-based Text Summarization Model

This repository contains the implementation of a Transformer-based model for text summarization. The model leverages multi-head attention and positional encoding to generate concise summaries from input text sequences.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Upcoming](#Upcoming)


## Introduction

This project implements a Transformer model designed for the task of text summarization. The model is based on the architecture described in the paper "Attention Is All You Need" and is built using TensorFlow and Keras. It includes an encoder and decoder, both utilizing multi-head attention and feed-forward layers.

## Model Architecture

The model consists of the following components:
- **Encoder**: Embedding layer, multi-head self-attention, and feed-forward layers.
- **Decoder**: Embedding layer, masked multi-head self-attention, multi-head attention with encoder outputs, and feed-forward layers.
- **Positional Encoding**: Adds positional information to the input embeddings.

## Dataset

The dataset used in this project is a combination of the CNN/DailyMail dataset. It contains articles and their corresponding summaries. The data is preprocessed by removing special characters and tokenizing the text.

# Upcoming
- finding a solution for the big parameter number (300M) for word embeddings
  - Tying weights would reduce 60M parameters
  - Low rank Embedding of k = 100 would result in reduction to 100M parameters but with a risk of underfitting
  - reducing the embedding dimension
  - Using petrained Word2Vec embeddings
  - Weight Pruning with tuning

