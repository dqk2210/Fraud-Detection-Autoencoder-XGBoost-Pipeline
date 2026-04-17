import json

with open('D:/Khanh/hoc/DO_AN/Credit-card-fraud-detection-using-a-deep-learning-multistage-model/1ST_JOURNAL_PAPER_RESULTS/SIMPLE_AUTOENCODER/SMOTE_AE_CNN/SMOTE_AE_CNN_5_FOLDS.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

with open('notebook_code.py', 'w', encoding='utf-8') as out:
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            out.write(''.join(cell['source']) + '\n\n')
