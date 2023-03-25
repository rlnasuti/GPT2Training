from transformers import TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

cache_dir = "./cache"

# Step 1: Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="./data/spells-gpt2-llm/spells-data.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Step 2: Set up the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)

# Step 3: Fine-tune the model
training_args = TrainingArguments(
    output_dir="./models/spells-gpt2-llm/output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Step 4: Save the fine-tuned model
model.save_pretrained("./models/spells-gpt2-llm")
