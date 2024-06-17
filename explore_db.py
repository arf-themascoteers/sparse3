import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/indian_pines.csv")
unique_classes = df.groupby('class').size()
print(unique_classes)

train, test = (train_test_split(df, test_size=0.1,stratify=df['class']))

print(train.shape, test.shape)

unique_classes = train.groupby('class').size()
print(unique_classes)

unique_classes = test.groupby('class').size()
print(unique_classes)