# Import the necessary libraries
import torch
import torch.nn as nn

# Define a dictionary of words and their corresponding indices
word_idx = {'hello': 0, 'world': 1, 'foo': 2, 'bar': 3, 'baz': 4, 'qux': 5, 'quux': 6, 'corge': 7, 'grault': 8,
            'garply': 9, 'waldo': 10, 'fred': 11, 'plugh': 12, }

# Create a tensor with the index of the word 'hello'
lookup_tensor = torch.tensor([word_idx['hello']], dtype=torch.long)

# Create an embedding layer with a vocabulary size of 13 and an embedding dimension of 5
embeds = nn.Embedding(len(word_idx), 5)

# Use the embedding layer to get the embedding of the word 'hello'
hello_embed = embeds(lookup_tensor)

# Print the embedding of the word 'hello'
print(hello_embed)