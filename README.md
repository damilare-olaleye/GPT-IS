# README

This code implements a conversation model based on the nanoGPT3 architecture using PyTorch. The model is trained on a dataset of conversations and can generate responses to user input.

## Requirements
- Python 3.x
- PyTorch
- Transformers library
- torch.cuda.amp
- json

## Usage

1. Install the required dependencies.

2. Prepare the dataset:
   - The input conversations should be stored in a JSON file named `index.json`.
   - The conversation dataset should have the following format:
     ```
     [
         {
             "content": "User input 1",
             "output": "Assistant response 1"
         },
         {
             "content": "User input 2",
             "output": "Assistant response 2"
         },
         ...
     ]
     ```

3. Set the hyperparameters in the code according to your requirements:
   - `batch_size`: The batch size for training and evaluation.
   - `block_size`: The length of the input sequence.
   - `max_iters`: The maximum number of training iterations.
   - `eval_interval`: The interval at which to evaluate the model during training.
   - `learning_rate`: The learning rate for training.
   - `device`: The device to run the model on ('cuda' for GPU or 'cpu' for CPU).
   - `eval_iters`: The number of iterations to evaluate the model.
   - `n_embd`: The embedding dimension for the model.
   - `n_head`: The number of attention heads in the model.
   - `n_layer`: The number of layers in the model.
   - `dropout`: The dropout rate for the model.

4. Run the code:
   - The code trains the model on the conversation dataset and saves the trained model in the current directory.
   - After training, the code starts an interactive session where you can have a conversation with the model.
   - Enter your input as a user and the model will generate a response as an assistant.
   - You can exit the conversation by typing 'exit', 'quit', ':wq', 'q', or 'clear'.

## Additional Notes

- The code uses sentiment analysis and entity recognition pipelines from the Hugging Face `transformers` library.
- The sentiment analysis pipeline is used to analyze the sentiment of user input.
- The entity recognition pipeline is used to extract entities from user input.
- The sentiment analysis and entity recognition pipelines require an internet connection to download the pre-trained models.

Please note that this README provides a high-level overview of the code. For detailed explanations, please refer to the code comments and consult the relevant documentation for the libraries used.



# Bigram Language Model

This repository contains code for a Bigram Language Model implemented using PyTorch. The model is trained on text data and can generate text based on user input. It utilizes the Transformers library for sentiment analysis and named entity recognition tasks.

## Prerequisites
- Python 3.x
- PyTorch
- Transformers

## Installation
1. Clone the repository:

   ```shell
   git clone https://github.com/username/repository.git
   ```

2. Install the required dependencies:

   ```shell
   pip install torch transformers
   ```

## Usage
1. Ensure that you have the necessary text data file named `input.txt`. The file should contain the text data used for training the language model.

2. Modify the hyperparameters in the code as per your requirements:

   - `batch_size`: The batch size used for training and evaluation.
   - `block_size`: The sequence length of input/output blocks.
   - `max_iters`: The maximum number of training iterations.
   - `eval_interval`: The interval at which to evaluate the model during training.
   - `learning_rate`: The learning rate for the optimizer.
   - `device`: The device to be used for training ('cuda' or 'cpu').
   - `eval_iters`: The number of iterations for evaluating the model during estimation of loss.
   - `n_embd`: The embedding dimension.
   - `n_head`: The number of self-attention heads.
   - `n_layer`: The number of transformer layers.
   - `dropout`: The dropout probability.

3. Train the model:

   ```shell
   python train.py
   ```

4. After training, the model will be saved as `trained_model.pt` and the conversation data will be saved as `conversation_data.pt`.

5. Interact with the model:

   ```shell
   python interact.py
   ```

   Enter your input and the model will generate a response based on the trained language model. To exit the interaction, type 'exit', 'quit', ':wq', or 'q'.

## Acknowledgments
This code is based on the Bigram Language Model implemented using PyTorch. The model architecture and training process are inspired by the Transformer model. The sentiment analysis and named entity recognition tasks are performed using the Transformers library.
