# Machine-Learning-Project
Team 29's Machine Learning Project

## Classification data preparation

```bash
pip install -r requirements.txt
python -m src.data.cleaning   # print the rows removed by each cleaning rule
python -m src.data.eda        # write tables and plots to outputs/classification/
```

```python
from src.data.loader import load_all_models, get_features
from src.data.cleaning import clean_classification

df, report = clean_classification(load_all_models())  # run before the train/test split
numeric, categorical = get_features("strict")         # or "relaxed"
```
