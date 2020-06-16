
# standard imports
import pandas as pd
import numpy as np
import shutil, os
from random import shuffle
import pickle
import matplotlib.pyplot as plt

#image processing imports
from PIL import Image
import cv2
from mtcnn import MTCNN
from fer import FER

# ML imports
import tensorflow
import keras
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve 
from sklearn.metrics import confusion_matrix, precision_recall_curve,f1_score, fbeta_score, log_loss
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Image processing functions
def get_size_statistics(DIR):
    '''Returns basic information about images in a specific directory in order to determine appropriate uniform image size'''
    
    heights = []
    widths = []
    img_count = 0
    
    for img in os.listdir(DIR):
        path = os.path.join(DIR, img)
        if "DS_Store" not in path:
            data = np.array(Image.open(path))
            heights.append(data.shape[0])
            widths.append(data.shape[1])
            img_count += 1
            
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))
    print("Min Height: " + str(min(heights)))
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))
    print("Min Width: " + str(min(widths)))


def load_photos(DIR, IMG_SIZE):
    '''Loads images from a directory and stores them in a dataframe as an array of pixels with the
    corresponding file name for a potential merge later in the analysis.'''
    data = []
    img_files = []
    
    for img_file in os.listdir(DIR):
    
        path = os.path.join(DIR, img_file)
        
        if ".DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            data.append(np.array(img))
            img_files.append(img_file)

    df = pd.DataFrame({'images': data, 'img_file':np.array(img_files)})
    return df

# filtering functions
def filter_df(df, no_filters, yes_filters):
    '''Takes in a dataframe, a list to filter on 'Nos' and a list to filter on 'Yes.' Returns a the filtered df.'''

    for col in no_filters:
        df = df[df[col] == -1]
    
    df = df[df.is_female != 0]
    df.drop(no_filters, axis=1, inplace=True)
    df['is_popular'] = np.where(df.popularity_score < 4.4, 0, 1)
    return df

def copy_images(df, dir_name):
    '''Takes in a dataframe of photos to filer on and moves them into their own folder.'''
    files_to_keep = np.array(df.img_file) + '.jpg'
    
    for fn in files_to_keep:
        shutil.copy('selfie_data/images/' + fn, 'selfie_data/' + dir_name)

# Non-image data analysis
def cv_scores(X, y, models, scores, folds):
    '''Takes in features and target from a dataframe, a list of models and a list of scores, then runs
    cv on each of the models and returns the scores.
    input:
        X: features
        y: target
        models: list of models
        scores: desires model scores
        folds: number of k folds
    output: desired scores of each model'''
    
    for key, model in models.items():
        print(str(key))
        for score in scores:
            print(score, cross_val_score(model, X, y, cv=folds, scoring=score).mean())
        print('\n')


def grid_search(X, y, model, param_grid, score, cv=10, refit=False):
    
    '''Takes in data and a pipeline and returns the optimal paremters and best score.'''
    
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=score, cv=cv, refit=refit)
    grid_search.fit(X, y)
    
    print('Best paremters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)


def cv_sampling(X, y, models, folds, sampler):
    '''Takes in features, target, list of models and oversampling/undersamping object and runs cv.
    
    input:
        X: features
        y: target
        models: list of models
        folds: number of k folds
        sampler: oversample, smote or undersample
    output:
        F1 score and AUC for each model'''
    
    kf = KFold(n_splits=folds, shuffle=True)
    
    for name, model in models.items():
    
        f1s = []
        roc_aucs = []
        X, y = np.array(X), np.array(y)

        for train_ind, val_ind in kf.split(X,y):
            # training and validation sets
            X_train_cv, y_train_cv = X[train_ind], y[train_ind]
            X_val, y_val = X[val_ind], y[val_ind] 

            # resampling the training data
            X_resampled_cv, y_resampled_cv = sampler.fit_sample(X_train_cv,y_train_cv)

            # fitting the model on the resampled data
            model.fit(X_resampled_cv, y_resampled_cv)

            # prediction on the validation set
            pred = model.predict(X_val)

            f1s.append(f1_score(y_val, pred))
            roc_aucs.append(roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

        print(str(name)) 
        print('f1', np.mean(f1s))
        print('ROC AUC', np.mean(roc_aucs))
    
def set_target(df, target):
    
    df_target = df[['img_file', target]]
    df_target.img_file = df_target.img_file + '.jpg'
    df_target[target] = df_target[target].replace(-1, 0)
    
    
    return df_target

def merge_target(df_photos, df_target):
    
    return df_photos.merge(df_target, left_on='img_file', right_on='img_file')


def check_dist(df, target):
    neg = df[target].value_counts()[0]
    pos = df[target].value_counts()[1]
    
    return neg / (pos + neg)


def train_test(df, target):
    # sets the features and the target
    X, y = np.array(df.images), np.array(df[target])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # proper shape of X_train/X_test
    X_train = np.array([x.flatten() for x in X_train])
    
    X_train = X_train.reshape((-1, 128, 128, 3))
    
    X_test = np.array([x.flatten() for x in X_test])
    X_test = X_test.reshape((-1, 128, 128, 3))
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # one hot encodes the target for modeling
    y_train_cat = to_categorical(y_train, num_classes=2)
    
    return X_train/255, X_test/255, y_train_cat, y_train, y_test


def predict(model, y_test, test_data):
    preds = model.predict_classes(test_data)
    
    print('Accuracy:', accuracy_score(y_test, preds))
    print('Dist of Preds:')
    print(pd.Series(preds).value_counts())