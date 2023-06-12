import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import pipeline
import torch.cuda.amp as amp

from bimodalLanguageModel import generate_response


# Function to interact with the model and get responses from user input
def interact_with_model():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', ':wq', 'q']:
            print("exiting...")
            break

        response = generate_response(user_input)
        print(f'output: {response}')  
        
  
        
# Interact with the model
interact_with_model()