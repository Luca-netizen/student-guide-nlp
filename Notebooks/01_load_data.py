from datasets import load_dataset
import pandas as pd

ds = load_dataset("allenai/sciq")
df = pd.DataFrame(ds["train"])
print(df.shape)
print(df.columns)
print(df.head())

