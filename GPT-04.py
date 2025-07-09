# 📦 Install required libraries
!pip install -q transformers datasets

# 📚 Import
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
os.environ["WANDB_DISABLED"] = "true"
# ✅ Load dataset
df = pd.read_csv('/kaggle/input/goodbooks-10k/books.csv')

# 🧹 Preprocess: combine description and reviews
# 🔎 Check column names
print("Columns:", df.columns)

# 🛠 Fix based on available columns
# This dataset uses 'book_id', 'title', 'authors', 'average_rating', etc.
# We'll combine 'title', 'authors', and 'average_rating' as a proxy
texts = (df['title'] + " by " + df['authors'] + ". Rated " + df['average_rating'].astype(str)).tolist()


# 💾 Save to .txt for training
os.makedirs("dataset", exist_ok=True)
with open("dataset/train.txt", "w", encoding="utf-8") as f:
    for line in texts:
        f.write(line.strip().replace('\n', ' ') + '\n')

# 🧠 Load GPT-2 model and tokenizer
model_name = "gpt2"  # or "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 🪄 Create TextDataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset("dataset/train.txt", tokenizer)

# 🧪 Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ⚙️ Training setup
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,                # Increase for better results
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# 🎯 Train the model
trainer.train()

# 💾 Save model and tokenizer
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

# 🎯 Generate text from a prompt using a locally fine-tuned GPT-2 model.
from transformers import pipeline

generator = pipeline('text-generation', model='./gpt2-finetuned', tokenizer='./gpt2-finetuned')

prompt = "Once upon a time"
outputs = generator(prompt, max_length=80, num_return_sequences=1)

print("Generated Text:\n", outputs[0]['generated_text'])

