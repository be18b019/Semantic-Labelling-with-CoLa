import numpy as np
import pandas  as pd
import math
import time
import enchant
import nltk
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from helpers import unicodeToAscii,representsDate, representsFloat, representsInt, shape_for_scale, shape_for_training, add_scaled_single_value_feat_rows, get_single_value_feat_rows, printables, dict_char_classes, single_value_feats, global_stats
from feature_extraction import load, extract_char_dist_feat, extract_word_shape_feat
from column_feature_extraction import load as col_load
from classification import classify



def list_single_features(n_chars):
    #for the exact feature importance
    features=['char dist. \'{}\' '.format(char) for char in printables]
    for i in range(n_chars):
        features.append('word-shape \'{}\''.format(i+1))
    #for i in range(n_chars):
        #features.append('dict. lookup \'{}\''.format(i+1))
    
    for feat in single_value_feats:
        features.append('global stats \'{}\''.format(feat))
    for i in range(len(single_value_feats),n_chars):
        features.append('padding')
    return features

def list_single_features_summed(n_chars):
    #for summed up feature importance for the row features
    features_summed=[]
    for i in range(n_chars):
        features_summed.append('char dist.')
    for i in range(n_chars):
        features_summed.append('word-shape')
    #for i in range(n_chars):
    #    features_summed.append('dict. lookup')

    for feat in single_value_feats:
        features_summed.append('global stats')
    for i in range(len(single_value_feats),n_chars):
        features_summed.append('padding')
    return features_summed

def list_col_features(n_chars):
    features=['global stats. \'{}\' '.format(feat) for feat in global_stats]
    for i in range(len(global_stats), n_chars):
        features.append('padding')
    for i in range(n_chars):
        features.append('char dist. \'{}\''.format(printables[i]))
    for i in range(n_chars):
        features.append('word-shape \'{}\''.format(i+1))
    for i in range(50):
        features.append('word embedding \'{}\''.format(i+1))
    for i in range(45):
        features.append('padding')
    for i in range(n_chars):
        features.append('paragraph embedding \'{}\''.format(i+1))
    for i in range(20):
        features.append('cotent histogram \'{}\''.format(i+1))
    for i in range(75):
        features.append('padding')
    return features

def list_col_features_summed(n_chars):
    features=['global stats.' for feat in global_stats]
    for i in range(len(global_stats), n_chars):
        features.append('padding')
    for i in range(n_chars):
        features.append('char dist.')
    for i in range(n_chars):
        features.append('word-shape')
    for i in range(50):
        features.append('word emb.')
    for i in range(45):
        features.append('padding')
    for i in range(n_chars):
        features.append('par. emb.')
    for i in range(20):
        features.append('cotent hist.')
    for i in range(75):
        features.append('padding')
    return features


def calc_feature_importances(RF, X_train, features, n_chars):
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ser_feature_importances=pd.Series(dtype=object)
    for f in range(X_train.shape[1]):
        feat = pd.Series([importances[indices[f]]], index=[features[indices[f]]])
        ser_feature_importances=ser_feature_importances.append(feat)
        
    return ser_feature_importances

def calc_feature_importances_summed(ser_feature_importances, feats):
    feats=list(set(feats))
    imp=[ser_feature_importances.loc[feats[i]].sum() for i in range(len(feats))]
    ser_feat_importances_summed=pd.Series(imp, index=feats).sort_values(ascending=False)
    return ser_feat_importances_summed

#adapt font used in latex
def make_plot(ser_feature_importances_summed, file_name):
   
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') #Arial
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(len(ser_feature_importances_summed)), 
            ser_feature_importances_summed.to_numpy(), 
            tick_label=list(ser_feature_importances_summed.keys()), 
            align='center')
    #file_name='feat_imp'
    plt.savefig('data/graphs/{}.png'.format(file_name), dpi=1000)
    plt.savefig('data/graphs/{}.svg'.format(file_name))
    plt.savefig('data/graphs/{}.pdf'.format(file_name))
    plt.show()

def sci_plot(list_importances):
    plt.style.use('science')
    plt.figure(figsize=(15, 15)) #1515 for col, 1512 cell
    plt.subplots_adjust(hspace=1)
    
    dataset_names=['Uniprot', 'LLD', 'OLD', 'NBDC', 'DBPedia', 'WikiData', 'T2D VizNet', 'City']
    #dataset_names=['Uniprot', 'LLD', 'OLD', 'NBDC', 'DBPedia', 'WikiData']
    for  i in range(1,len(list_importances)+1):
        plt.subplot(4, 2, i) #4 for col, 3 cell
        h=plt.bar(range(len(list_importances[i-1])), 
                list_importances[i-1].to_numpy(), 
                tick_label=list(list_importances[i-1].keys()), 
                align='center')
        
        plt.title('{}'.format(dataset_names[i-1]), fontsize=25)
        #plt.xlabel('feature')
        plt.ylabel('importance [\%]', fontsize=20)  
        plt.yticks(fontsize=20)
        xticks_pos = [0.65*patch.get_width() + patch.get_xy()[0] for patch in h]
        plt.xticks(xticks_pos, list(list_importances[i-1].keys()),  ha='right', rotation=30, fontsize=20)
    
    file_name='col_feat_imp' #col
    plt.savefig('data/graphs/{}.png'.format(file_name), dpi=1000)
    plt.savefig('data/graphs/{}.svg'.format(file_name))
    plt.savefig('data/graphs/{}.pdf'.format(file_name))
    plt.show()
    


if __name__=='__main__': 
    key='uniprot'
    samples=100 #1000
    n_chars=95 #includes 98% of whole cells for uniprot
    n_features=3
    classifier='RF' 
    keys=['uniprot','lld', 'old','nbdc', 'dbpedia', 'wikidata', 'sherlock', 'rammandan']
    #keys=['uniprot','lld', 'old','nbdc', 'dbpedia', 'wikidata']

    '''
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    #feature importances cell approach
    
    
    '''
    
    #feature importance try with LinSVm
    '''
    classifier='LinSVM'
    X_train, X_test, y_train, y_test, dict_labels=col_load(key)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
    coef = clf.coef_.todense().ravel()
    
    names=
    imp = cclf.coef_
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    '''
    
    '''
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ser_feature_importances=pd.Series(dtype=object)
    for f in range(X_train.shape[1]):
        feat = pd.Series([importances[indices[f]]], index=[features[indices[f]]])
        ser_feature_importances=ser_feature_importances.append(feat)
        
    return ser_feature_importances
    '''
    
    '''
    X_train, X_test, y_train, y_test, dict_labels=load(key, samples)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
    ser_feature_importances=calc_feature_importances(clf, X_train, list_single_features(n_chars), n_chars)
    ser_feature_importances_summed=calc_feature_importances(clf, X_train, list_single_features_summed(n_chars), n_chars)
    ser_feature_importances_summed=calc_feature_importances_summed(ser_feature_importances_summed, list_single_features_summed(n_chars))
    ser_feature_importances_summed=ser_feature_importances_summed.drop(labels=['padding'])
    make_plot(ser_feature_importances_summed, 'cell_dbpedia_feat_imp')
    '''
    
    '''
    X_train, X_test, y_train, y_test, dict_labels=col_load(key)
    ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
    ser_feature_importances=calc_feature_importances(clf, X_train, list_col_features(n_chars), n_chars)
    ser_feature_importances_summed=calc_feature_importances(clf, X_train, list_col_features_summed(n_chars), n_chars)
    ser_feature_importances_summed=calc_feature_importances_summed(ser_feature_importances_summed, list_col_features_summed(n_chars))
    ser_feature_importances_summed=ser_feature_importances_summed.drop(labels=['padding'])
    '''
    '''
    #feature importances column approach
    list_importances=[]
    for key in keys:
        X_train, X_test, y_train, y_test, dict_labels=col_load(key)
        ser_scores_params, df_cm, report, clf=classify(classifier, X_train, y_train, X_test, y_test, dict_labels, key)
        ser_feature_importances=calc_feature_importances(clf, X_train, list_col_features(n_chars), n_chars)
        ser_feature_importances_summed=calc_feature_importances(clf, X_train, list_col_features_summed(n_chars), n_chars)
        ser_feature_importances_summed=calc_feature_importances_summed(ser_feature_importances_summed, list_col_features_summed(n_chars))
        ser_feature_importances_summed=ser_feature_importances_summed.drop(labels=['padding'])
        list_importances.append(ser_feature_importances_summed) 
        
    sci_plot(list_importances)
    '''
    '''
    ser_imp=pd.Series(list_importances)
    
    with pd.ExcelWriter('feat.xlsx') as writer:
        ser_imp.to_excel(writer)    
    ser=pd.read_excel(r'feat.xlsx',engine='openpyxl')
    '''
    '''
    list_importances=[pd.Series(data=[0.401870,0.345528,0.252602], index=['char dist.','global stats','word-shape']),
    pd.Series(data=[0.467817,0.305249,0.226933], index=['char dist.','global stats','word-shape']),
    pd.Series(data=[0.509940,0.278484,0.211576], index=['char dist.','global stats','word-shape']),
    pd.Series(data=[0.558082,0.288783,0.153135], index=['char dist.','global stats','word-shape']),
    pd.Series(data=[0.550565,0.290952,0.158483], index=['char dist.','global stats','word-shape']),
    pd.Series(data=[0.431782,0.340832,0.227386], index=['char dist.','global stats','word-shape'])]
    
    sci_plot(list_importances)
    '''
    
    
    
