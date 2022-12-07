"""
Script to encode each caption from Laion dataset images 
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration

MAX_LENGTH = 15
tokenizer = T5Tokenizer.from_pretrained("t5-large")

captions = np.load("captions.npy")

tmp = captions[0]

test = tokenizer(tmp, padding = "longest", max_length=MAX_LENGTH, truncation=True)


