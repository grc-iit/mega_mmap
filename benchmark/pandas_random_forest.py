#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier

train_path = sys.argv[1]
test_path = sys.argv[2]
print(f'RF classifier on {train_path} and {test_path}')
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)
print(train_df)
rf = RandomForestClassifier(n_estimators=4, max_depth=5)
rf.fit(train_df[['x', 'y']], train_df['class'])
preds = rf.predict(test_df[['x', 'y']])
print(pd.DataFrame({'preds': preds, 'actual': test_df['class']}))
accuracy = (preds == test_df['class']).sum()
accuracy /= len(preds)
print(f'Accuracy: {accuracy}')
