'''
@author jasperan

This code will translate the old_predictions.json file into the correct submission format

example:

{
    "target": {
        "0": 1,
        "1": 2
    }
}

where key=idx_test and value=prediction from the model.
'''

import json
import pandas as pd

# Opening JSON file
f = open('../old_predictions.json')
df_test = pd.read_csv('H:/WORK/reto_nuwe/reto/test.csv')  
# returns JSON object as 
# a dictionary
data = json.load(f)

new_object = {}

print(len(data['target']))
print(len(df_test))
assert len(data['target']) == len(df_test)



for x in range(len(df_test)):
    #for key in data['target']:
    current_img = df_test.iloc[x]['path_img'].replace('all_imgs/', '')
    current_idx = df_test.iloc[x]['idx_test']
    new_object[str(current_idx)] = data['target'][current_img]
    print(current_img, current_idx, data['target'][current_img])

print(len(new_object))

final_json = {'target': new_object}

with open('../predictions.json', 'w') as f:
        json.dump(final_json, f)