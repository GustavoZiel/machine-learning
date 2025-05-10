# Projeto 1 - Aprendizado de Máquina
<!-- 
> Kaggle Dataset Page: https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success

Possible Datasets:

https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success/data
https://www.kaggle.com/datasets/adilshamim8/math-students/data
https://www.kaggle.com/datasets/melissamonfared/qs-world-university-rankings-2025
https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024
https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset
 -->

## Pipeline

- Presenting the Competition
- Presenting the Datasets
- EDA for each Dataset
  - train
    - family
    - Plot de sales
      - Enxergar sazonalidade em alguns e outros não
  - stores
  - oil
    - no weekend values
    - Ecuador Economy
  - holidays
    - Work Days
    - Transferred
  - transactions
- Preprocessing
  - NaN Values
  - Downcast
  - Encode
    - Label Encoding
    - One Hot Encoding
    - Target Encoding
  - Normalization
    - MinMaxScaler
    - StandardScaler
- Feature Engineering
  - Data Features
  - Earthquake
  - Payday
  - Nixtla Framework
  - Lags
  - Rolling
  - Train Test Split
- Train
  - Multiple Regressions Models in Default Config Performance for each processed dataset
    - Time Series Split
    - Várias Méttricas
      - RMSE
      - MAE
      - RMSLE
      - MAPE
    - Encontrar melhores, foco em:
      - LightGBM (A priori esse será o melhor e mais rápido)
      - CatBoos
      - XGBoost (Ruim)
  - Apenas LightGBM
    - HyperParameter Tuning
      - Optuna
    - Ensemble per Family
    - Tune Global X Tune Ensemble
    - Performance Final
    - Score X    
  - Conclusion
    - Vários conceitos praticados
    - Várias bibliotecas e frameworks utilziados
      - Nixtla
      - Optuna
      - LightGBM
    - Modelos averiguados
      - Performance
      - Rapidez