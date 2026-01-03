import os
from transformers import pipeline
from transformers import Tool

class CooltoolTool(Tool):
    name = "CoolTool"
    description = (
        "CoolToolv"
    )
    inputs = ["text"]
    outputs = ["text"]
    def __call__(self, prompt: str):
        token = os.environ['hf']
        text_generator = pipeline(model="microsoft/Orca-2-13b", token=token)
        generated_text = text_generator(prompt, max_length=500, num_return_sequences=1, temperature=0.7)
        print(generated_text)
        return generated_text
