'''
Functions for computing Variation of Information & Bhattacharyya distance between two matrices assumed to come from multivariate gaussian distributions

Created on 21 Feb 2018

@author: bobaseb (Sebastian Bobadilla-Suarez)

*requires numpy & scikit-learn
'''

import numpy as np
from sklearn.covariance import ledoit_wolf

def var_info(mat1, mat2,reg=0):
    n = mat1.shape[1]
    mat0 = np.hstack([mat1,mat2])
    if(reg==1):
        cov_mat0 = ledoit_wolf(mat0)[0]
        cov_mat1 = ledoit_wolf(mat1)[0]
        cov_mat2 = ledoit_wolf(mat2)[0]
    else:
        cov_mat0 = np.cov(mat0)
        cov_mat1 = np.cov(mat1)
        cov_mat2 = np.cov(mat2)        
    (sign0, logdet0) = np.linalg.slogdet(cov_mat0)
    (sign1, logdet1) = np.linalg.slogdet(cov_mat1)
    (sign2, logdet2) = np.linalg.slogdet(cov_mat2)
    ln_det_mat0 = logdet0
    ln_det_mat1 = logdet1
    ln_det_mat2 = logdet2
    H_mat1 = 0.5*np.log(np.power((2*np.exp(1)*np.pi), n)) + 0.5*ln_det_mat1
    H_mat2 = 0.5*np.log(np.power((2*np.exp(1)*np.pi), n)) + 0.5*ln_det_mat2
    MI = 0.5*(ln_det_mat1 + ln_det_mat2 - ln_det_mat0)
    return 2*MI - H_mat1 - H_mat2;

def bhdist (mu1, mu2, mat1, mat2,reg=0):
    #Bhattacharyya_distance assuming normal distros
    diff_mn_mat = np.matrix(mu1-mu2).T
    if(reg==1):
        cov_mat1 = ledoit_wolf(mat1)[0]
        cov_mat2 = ledoit_wolf(mat2)[0]
    else:
        cov_mat1 = np.cov(mat1)
        cov_mat2 = np.cov(mat2) 
    cov_mat_mn = (cov_mat1 + cov_mat2)/2
    icov_mat_mn = invcov_mah(cov_mat_mn,0)
    term1 = np.dot(np.dot(diff_mn_mat.T, icov_mat_mn), diff_mn_mat)/8
    (sign1, logdet1) = np.linalg.slogdet(cov_mat1)
    (sign2, logdet2) = np.linalg.slogdet(cov_mat2)
    (sign_mn, logdet_mn) = np.linalg.slogdet(cov_mat_mn)
    ln_det_mat1 = logdet1
    ln_det_mat2 = logdet2
    ln_det_mat_mn = logdet_mn
    term2 = (ln_det_mat_mn/2) - (ln_det_mat1+ln_det_mat2)/4
    result = term1+term2;
    return result[0,0];
