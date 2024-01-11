import pandas as pd

# Libraries for data wrangling
import pandas as pd
import numpy as np 
# Libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Library to see the progress
from tqdm import tqdm

# Libraries with functions used in modelling
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from category_encoders import MEstimateEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer

# Libraries with the models
from xgboost import XGBClassifier


# Library to ignore warnings
import warnings
warnings.filterwarnings("ignore")




# Função de validação adversarial
def adversarial_validation(train, test, target_variable, seed, label):
    
    # Utiliza o dataset de treino para descobrir quais são as features categóricas
    numerical_features = list(test._get_numeric_data())
    categorical_features = list(test.drop(numerical_features, axis = 1))

    # Remove a coluna 'Exited' do conjunto de treinamento
    adv_train = train.drop(target_variable, axis=1)

    # Cria uma cópia do conjunto de teste
    adv_test = test.copy()

    # Adiciona uma coluna 'is_test' indicando se a amostra pertence ao conjunto de teste (1) ou de treinamento (0)
    adv_train['is_test'] = 0
    adv_test['is_test'] = 1

    # Concatena os conjuntos de treinamento e teste
    adv = pd.concat([adv_train, adv_test], ignore_index=True)

    # Embaralha as amostras
    adv_shuffled = adv.sample(frac=1)

    # Divide as features (X) e os rótulos (y)
    adv_X = adv_shuffled.drop('is_test', axis=1)
    adv_y = adv_shuffled.is_test

    # Configura a validação cruzada estratificada com 5 folds
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    # Inicializa listas para armazenar pontuações de validação e previsões
    val_scores = []
    predictions = np.zeros(len(adv))

    # Loop sobre os folds da validação cruzada
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(adv_X, adv_y), total=skf.get_n_splits(), desc="Cross-validation")):
        
        # Cria um modelo usando M-estimate encoding e um classificador XGBoost
        adv_lr = make_pipeline(MEstimateEncoder(cols=categorical_features), XGBClassifier(random_state=seed))
        
        # Treina o modelo no conjunto de treinamento atual
        adv_lr.fit(adv_X.iloc[train_idx], adv_y.iloc[train_idx])

        # Faz previsões no conjunto de validação e calcula a pontuação ROC AUC
        val_preds = adv_lr.predict_proba(adv_X.iloc[val_idx])[:, 1]
        predictions[val_idx] = val_preds
        val_score = roc_auc_score(adv_y.iloc[val_idx], val_preds)
        val_scores.append(val_score)

    # Calcula a curva ROC e plota
    fpr, tpr, _ = roc_curve(adv['is_test'], predictions)
    plt.figure(figsize=(10, 10))
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Datasets indistinguíveis")
    sns.lineplot(x=fpr, y=tpr, label="Validação adversarial com classificador")
    plt.title(f'{label} Validation = {np.mean(val_scores):.5f}', weight='bold', size=17)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.show()



def cross_val_score(dataset , estimator, cv, seed, label, show_importance = False):
    
    X = dataset.copy()
    y = X.pop('Exited')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    train_scores, val_scores= [], []
    feature_importances_table = pd.DataFrame({'value' : 0}, index = list(X.columns))
    
    #training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
        model = clone(estimator)
        
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]    
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
                  
        val_predictions[val_idx] += val_preds
        feature_importances_table['value'] += permutation_importance(model, X_val, y_val, random_state = seed, scoring = make_scorer(roc_auc_score, needs_proba = True), n_repeats = 5)\
            .importances_mean / cv.get_n_splits()
        
        #evaluate model for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
        
    if show_importance:
        plt.figure(figsize = (20, 15))
        plt.title(f'Features with Biggest Importance of {np.mean(val_scores):.5f} Model', size = 18, weight = 'bold')
        sns.barplot(feature_importances_table.sort_values('value', ascending = False).T, orient = 'h', palette = 'viridis')
        plt.show()
    else:
        print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    return val_scores, val_predictions