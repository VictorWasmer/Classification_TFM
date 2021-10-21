import pandas as pd

crosswalk_df = pd.read_csv('annotations.csv')
sidewalk_df = pd.read_csv('annotations2.csv')

print(crosswalk_df.head())
print(sidewalk_df.head())

vertical_stack = pd.concat([crosswalk_df, sidewalk_df], axis=0)

vertical_stack.to_csv('annotations_xwalk_swalk.csv', index = False)
