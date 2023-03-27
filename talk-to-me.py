from transformers import pipeline
from transformers import TextDataset, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import warnings
import logging
import torch

def convert_ids_to_entities(input_ids, predictions):
    entities = []
    current_entity = None

    for token_id, pred in zip(input_ids, predictions):
        if pred != 0:
            if current_entity is None:
                current_entity = {
                    "start": token_id,
                    "end": token_id,
                    "label": ner_model.config.id2label[pred]
                }
            else:
                current_entity["end"] = token_id
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    return entities



warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

cache_dir = "./cache"

# Load the BERT NER model
# ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("models/spells-ner")
ner_tokenizer = AutoTokenizer.from_pretrained("models/spells-ner")


# Load the conversation and spell GPT-2 models and tokenizers
dm_model = GPT2LMHeadModel.from_pretrained('/Users/robertnasuti/Desktop/Dev/GPT2Training/models/dm-gpt2-llm', cache_dir=cache_dir)
dm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

spells_model = GPT2LMHeadModel.from_pretrained('/Users/robertnasuti/Desktop/Dev/GPT2Training/models/spells-gpt2-llm', cache_dir=cache_dir)
spells_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create text-generation pipelines for the GPT-2 models
response_generator = pipeline('text-generation', model=dm_model, tokenizer=dm_tokenizer)
spells_generator = pipeline('text-generation', model=spells_model, tokenizer=spells_tokenizer)

# Function to generate a response using the GPT-2 models
def generate_response(prompt, max_length=100):
    # Detect named entities using the BERT NER model
    ner_input = ner_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = ner_model(**ner_input)

    # Process the output tensor to obtain the entity information
    predictions = torch.argmax(output.logits, dim=2)
    input_ids = ner_input["input_ids"].numpy()[0]
    predictions = predictions[0].numpy()

    entities = convert_ids_to_entities(input_ids, predictions)

    # Identify if there's any entity related to a spell
    spell_entity = None
    for entity in entities:
        if entity["label"] == "SPELL":  # Adjust this to the appropriate label for a spell
            spell_entity = entity["word"]
            break

    # Generate a response using the conversation GPT-2 model
    output_text = response_generator(prompt, max_length=max_length)
    response_text = output_text[0]["generated_text"]

    # If a spell entity was found, generate spell information using the spell GPT-2 model
    if spell_entity:
        spell_info = spells_generator(spell_entity, max_length=max_length)
        response_text += "\n\n" + spell_info[0]["generated_text"]

    return response_text

while True:
    prompt = input("User: ")
    generated_text = generate_response(prompt)
    print("GPT-2 DM: " + generated_text)

