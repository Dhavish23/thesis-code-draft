import pandas as pd

print("Loading datasets...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake rows:", len(fake))
print("True rows:", len(true))


# Add labels
fake["label"] = "fake"
true["label"] = "real"


# Use ONLY the headline (title)
# This helps the model learn how headlines look,
# which matches how users will input news in the website
fake["text"] = fake["title"]
true["text"] = true["title"]


# Keep only needed columns
fake = fake[["text", "label"]]
true = true[["text", "label"]]


# Merge datasets
df = pd.concat([fake, true])


# Shuffle dataset
df = df.sample(frac=1, random_state=42)


print("Final dataset size:", df.shape)


# Save combined dataset
df.to_csv("data.csv", index=False)

print("Dataset saved as data.csv")