import pandas as pd
import numpy as np
import math
import unicodedata
from datetime import datetime

#§ or ° are not included but is a standard so its fine, maybe include 255 ascii signs
#malformed so copied
#chars=string.printable 
#len is 95 as DEL is excluded
printables='''0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
dict_char_classes={
        'lower':'abcdefghijklmnopqrstuvwxyz',
        'upper':'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'number':'0123456789',
        'special':'''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
        }

single_value_feats=['n_chars', 'n_numbers', 'n_letters', 'n_spaces',
                    'n_uppers', 'n_specials', 'is_date', 'perc_numbers',
                    'perc_letters', 'perc_specials', 'first_letter', 'digits_before_comma',
                    'digits_after_comma', 'negativ_number', 'first_number', 
                    'unit_place', 'tenth_place']

cell_feats=['char_dist', 'word_shape', 'dict_lookup']

global_stats=['n_none','frac_none', 'entropy','frac_unique','frac_num',
     'frac_alpha','frac_sym','mean_num_count', 'std_num_count',
     'mean_alpha_count', 'std_alpha_count', 'mean_sym_count', 
     'std_sym_count', 'sum_chars', 'min_chars','max_chars', 
     'median_chars', 'mode_chars','kurt_chars', 'skew_chars',
     'any_chars', 'all_chars','dates','min_num', 'max_num', 'median_num',
     'mean_num', 'mode_num', 'std_num', 'kurt_num', 'skew_num']

endpoint_dict =	{
  'uniprot':'http://sparql.uniprot.org/sparql', #works great
  'ebi':'https://www.ebi.ac.uk/rdf/services/sparql', #certification error
  'disgenet':'http://rdf.disgenet.org/sparql/', #no results for found datatype properties
  'monarch':'http://rdf.monarchinitiative.org/sparql', #doesnt work --> returns some html stuff
  'lld':'http://linkedlifedata.com/sparql', #includes many databases
  'old':'http://sparql.openlifedata.org/', #includes many databases
  'wikipathways':'http://sparql.wikipathways.org/', #doesnt work --> returns some html stuff
  'tcga':'http://tcga.deri.ie/', #doesnt work --> returns some html stuff
  'pubchem':'https://pubchemdocs.ncbi.nlm.nih.gov/rdf', #doesnt work --> returns some html stuff
  'nbdc':'http://integbio.jp/rdf/sparql', #very much dt_properties
  'dbpedia':'http://dbpedia.org/sparql',
  'wikidata': 'https://query.wikidata.org/sparql'
}
#http://drugbank.bio2rdf.org/sparql old endpoint?
#'https://opensparql.sbgenomics.com/blazegraph/namespace/tcga_metadata_kb/sparql' doesnt work --> urlerror
#drugbank=lld

# from https://stackoverflow.com/a/518232/2809427
#normalizing the data to only contain ascii printables
def unicodeToAscii(s):
    chars=printables
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in chars)

#https://stackoverflow.com/questions/1265665/how-can-i-check-if-a-string-represents-an-int-without-using-try-except
def representsInt(s):
    try: 
        int(s)
        return True
    except:
        return False
    
def representsFloat(s):
    try: 
        float(s)
        return True
    except:
        return False
    
 #from https://stackoverflow.com/questions/33204500/pandas-automatically-detect-date-columns-at-run-time
def representsDate(s):
    try:
        pd.to_datetime(s, utc=True)
        return True
    except:
        return False

def drop_nan_inf_none(col):
    col=pd.Series(list(filter(None, col)))
    for i in range(len(col)):
        try:
            cell=float(col[i])
            if np.isnan(cell):
                col=col.drop(i)
                continue
            if np.isinf(cell):
                col=col.drop(i)
                continue
            if cell>np.finfo(np.float32).max:
                col=col.drop(i)
                continue
        except:
            continue
    return col.reset_index(drop=True)

def check_float_min_max(X):
    for i in range(len(X)):
        mask=X[i]>np.finfo(np.float64).max
        X[i][mask]=np.finfo(np.float64).max
        mask=X[i]<np.finfo(np.float64).min
        X[i][mask]=np.finfo(np.float64).min
    return X

def shape_for_scale(X, X_shape, n_features):
    X_helper=np.zeros(shape=(X_shape[0]*X_shape[1], n_features))
    idx=0
    for i in range(X_shape[0]):
        for j in range(n_features):
            X_helper[idx:idx+X_shape[1],j]=X[i][j,:]
        idx=idx+X_shape[1]
    return X_helper

def shape_for_training(X, X_shape, n_features):
    shaped=np.zeros(shape=(X_shape[0], n_features*X_shape[1]))
    idxrow=0
    for i in range(X_shape[0]):
        idxcolumn=0
        for j in range(n_features):
            shaped[i,idxcolumn:idxcolumn+X_shape[1]]=X[idxrow:idxrow+X_shape[1],j]
            idxcolumn=idxcolumn+X_shape[1]
        idxrow=idxrow+X_shape[1]
    return shaped

def get_single_value_feat_rows(X, n_features):
    X_single_feat=np.zeros(shape=(X.shape[0], X[0].shape[1]))
    for i in range(len(X)):
        X_single_feat[i,:]=X[i][n_features-1,:]
    return X_single_feat

def add_scaled_single_value_feat_rows(X_single_feat_scaled, X_scaled, n_chars, n_features):
    idx_row=0
    for i in range(X_single_feat_scaled.shape[0]):
        X_scaled[idx_row:idx_row+n_chars, n_features-1]=X_single_feat_scaled[i]
        idx_row+=n_chars
    return X_scaled

def make_feat_importance_file(key, classifier, samples, n_chars, n_features, duration, ser_metrics, ser_feature_importances, ser_feature_importances_summed):
    ser_metrics['duration']=duration
    ser_metrics['samples']=samples
    ser_metrics['n_chars']=n_chars
    ser_metrics['n_features']=n_features
    ser_metrics['database']=key
    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M_%S')
    #path='C:/Users/jgtha/Dropbox/Default Python Directory/data/measurements/'
    file_name='data/feature_importances/Scores_{}_{}_classifier_{}_samples_{}_nchars_{}_features_{}.xlsx'.format(key, classifier, samples, n_chars, n_features, dt_string)
    metrics_sheet='params_and_scores'
    feature_importance_sheet='feature_importances'
    feature_importance_summed_sheet='feature_importances_summed'
    with pd.ExcelWriter(file_name) as writer:
        ser_metrics.to_excel(writer, sheet_name=metrics_sheet, header=['params and scores'])
        ser_feature_importances.to_excel(writer, sheet_name=feature_importance_sheet, header=['feature importances'])
        ser_feature_importances_summed.to_excel(writer, sheet_name=feature_importance_summed_sheet, header=['summed feature importances'])



