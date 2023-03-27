from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class SpellsNERDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.tokenizer = tokenizer
        self.sentences, self.labels = self.read_data(data_file)

    def read_data(self, data_file):
        sentences = []
        labels = []
        with open(data_file, "r") as f:
            for line in f:
                sentence = line.strip()
                tokens = []
                token_labels = []

                i = 0
                inside_spell = False
                first_spell_token = True
                while i < len(sentence):
                    if sentence[i:i + 9] == "[B-SPELL]":
                        inside_spell = True
                        first_spell_token = True
                        i += 9
                    elif sentence[i:i + 10] == "[/B-SPELL]":
                        inside_spell = False
                        i += 10
                    elif sentence[i] == " ":
                        i += 1
                        continue
                    else:
                        token_start = i
                        token_end = sentence[token_start:].find(" ") + token_start
                        if token_end == token_start - 1:
                            token_end = len(sentence)
                        tokens.append(sentence[token_start:token_end])
                        if inside_spell:
                            token_labels.append("B-SPELL" if first_spell_token else "I-SPELL")
                            first_spell_token = False
                        else:
                            token_labels.append("O")
                        i = token_end
                sentences.append(tokens)
                labels.append(token_labels)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        label_seq = self.labels[idx]

        # Tokenize and encode the tokens
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        # Convert labels to indices
        label_map = {"O": 0, "B-SPELL": 1, "I-SPELL": 2}  # Add more label types if needed
        labels = [label_map[l] for l in label_seq]

        # Pad labels to the same length as the input text
        pad_length = 128 - len(labels)
        labels = labels + [label_map["O"]] * pad_length

        # Convert the labels list to a tensor
        labels = torch.tensor(labels)

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}





# Load the pre-trained model, tokenizer, and configuration
model_name = "bert-base-cased"
config = AutoConfig.from_pretrained(model_name, num_labels=3)  # Adjust num_labels if you have more labels
model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
unsplit_dataset = SpellsNERDataset("data/spells-ner/train.txt", tokenizer)
train_dataset, val_dataset = train_test_split(unsplit_dataset, test_size=0.2, random_state=42)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./models/spells-ner/output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model to the 'models/spells-ner' directory
model.save_pretrained("models/spells-ner")
tokenizer.save_pretrained("models/spells-ner")