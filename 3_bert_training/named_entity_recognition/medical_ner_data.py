"""
NER Tagging with BIO Scheme for Medical Entities

This script prepares data for a Named Entity Recognition (NER) task in the medical domain using
the BIO (Begin, Inside, Outside) tagging scheme. Entities include 'condition', 'symptom', and 'procedure'.
The annotations and mappings defined here will be used to train and validate an NER model using tools 
like Hugging Face Transformers.
"""

# --------------------------------------------
# 1. Entity to Index Mapping
# --------------------------------------------
# The following dictionary maps BIO-tagged entity labels to integer indices.
# These are useful for converting human-readable labels into numeric form for modeling.

label_to_id = {
    "O": 0,                # Outside any entity
    "B-condition": 1,      # Beginning of a 'condition' entity
    "I-condition": 2,      # Inside a 'condition' entity
    "B-symptom": 3,        # Beginning of a 'symptom' entity
    "I-symptom": 4,        # Inside a 'symptom' entity
    "B-procedure": 5,      # Beginning of a 'procedure' entity
    "I-procedure": 6       # Inside a 'procedure' entity
}

# --------------------------------------------
# 2. BIO Tagging Scheme Documentation
# --------------------------------------------
def bio_tag():
    """
    BIO Tagging Scheme:
    
    - B-: Beginning of a multi-word entity
    - I-: Inside of a multi-word entity
    - O : Outside of any entity

    This scheme is used for labeling sequences where entities may span multiple tokens.
    Although this function serves as a docstring only, it helps developers understand the format.
    """
    pass

# --------------------------------------------
# 3. Training Texts and Labels
# --------------------------------------------
# These are sample training sentences and their corresponding token-level BIO labels.

train_texts = [
    "Patient has diabetes.",                      # Condition
    "MRI scan showed a tumor.",                   # Procedure & Condition
    "He has hypertension.",                       # Condition
    "The patient is experiencing a headache.",    # Symptom
    "She underwent a CT scan yesterday."          # Procedure
]

train_labels = [
    ["O", "O", "B-condition", "O"],                               # 'diabetes' → condition
    ["B-procedure", "I-procedure", "O", "O", "B-condition", "O"], # 'MRI scan' → procedure, 'tumor' → condition
    ["O", "O", "B-condition", "O"],                               # 'hypertension' → condition
    ["O", "O", "O", "O", "B-symptom", "O"],                       # 'headache' → symptom
    ["O", "O", "O", "B-procedure", "I-procedure", "O", "O"]       # 'CT scan' → procedure
]

# --------------------------------------------
# 4. Validation Texts and Labels
# --------------------------------------------
# These are sample validation sentences and their corresponding token-level BIO labels.

val_texts = [
    "The doctor diagnosed asthma.",                   # Condition
    "CT scan found pneumonia.",                       # Procedure & Condition
    "The patient complained of severe cough.",        # Symptom
    "He was given a flu shot as treatment."           # Procedure
]

val_labels = [
    ["O", "O", "O", "B-condition", "O"],                              # 'asthma' → condition
    ["B-procedure", "I-procedure", "O", "B-condition", "O"],         # 'CT scan' → procedure, 'pneumonia' → condition
    ["O", "O", "O", "O", "B-symptom", "O", "O"],                      # 'cough' → symptom
    ["O", "O", "O", "O", "B-procedure", "I-procedure", "O", "O"]      # 'flu shot' → procedure
]

# --------------------------------------------
# 5. Format for Compatibility
# --------------------------------------------
# Wrapping each sentence string into its own list.
# Some NLP tokenizers (e.g., Hugging Face's pipeline or dataset format) expect each document
# to be in the form of a list of strings (even if each list has only one sentence).

train_texts = [[text] for text in train_texts]
val_texts = [[text] for text in val_texts]
