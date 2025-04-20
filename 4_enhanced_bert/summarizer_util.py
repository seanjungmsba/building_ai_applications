from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding
from transformers.modeling_utils import PreTrainedModel

'''
📄 Summarization Utilities

This module loads a pretrained encoder-decoder transformer model 
(roberta2roberta_L-24_bbc) from Hugging Face and defines a function
for generating abstractive summaries from raw text.
'''

# Load model from Hugging Face model hub
model_name = "google/roberta2roberta_L-24_bbc"

# Load tokenizer for tokenizing input text
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

# Load sequence-to-sequence transformer model (RoBERTa encoder-decoder)
model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(article: str) -> str:
    """
    Generates an abstractive summary for a given article.

    Args:
        article (str): Full body of text to summarize

    Returns:
        str: One-sentence summary of the article
    """

    # Convert the input article into token IDs and attention masks
    batch_encoding: BatchEncoding = tokenizer(
        text=article,
        return_tensors='pt'  # PyTorch-compatible format
    )

    # Extract tokenized input from batch
    input_ids: Tensor = batch_encoding.input_ids

    # Generate output token IDs using greedy decoding (can be replaced with beam search)
    output_ids = model.generate(input_ids)[0]

    # Decode output tokens to a clean text string
    result = tokenizer.decode(token_ids=output_ids, skip_special_tokens=True)

    return result
