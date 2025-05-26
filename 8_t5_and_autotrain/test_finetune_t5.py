import os
from dotenv import load_dotenv
from transformers import pipeline

# Load variables from .env into environment
load_dotenv()

# Access the model name
model_name = os.getenv("HF_MODEL_NAME")

# Use in Hugging Face pipeline
pipe = pipeline("text2text-generation", model=model_name)

# Run inference
input_topics = ["politics", "sports", "technology"]
results = pipe(input_topics, max_new_tokens=100)

# Print results
for topic, story in zip(input_topics, results):
    print(f"🧠 Topic: {topic}")
    print(f"📖 Generated Story: {story['generated_text']}\n")

'''
🧠 Topic: politics
📖 Generated Story: Politics is a fascinating subject that captures the imagination People have long studied Politics to understand its mysteries In the future we may uncover even more secrets about Politics

🧠 Topic: sports
📖 Generated Story: Sport is a fascinating subject that captures the imagination People have long studied Sports to understand its mysteries In the future we may uncover even more secrets about Sports

🧠 Topic: technology
📖 Generated Story: Technology is a fascinating subject that captures the imagination People have long studied Technology to understand its mysteries In the future we may uncover even more secrets about Technology
'''