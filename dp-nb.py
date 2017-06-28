#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:31:44 2017

@author: ubu
"""

import numpy as np
import pandas as pd
import pymc3 as pm
from theano import shared, tensor as tt



df = pd.read_csv("/.../data/Kohorte1J.csv")

N, _ = df.shape
K = 20   # max number of mixture components


y_ANZ_KH = df.ANZ_KH[:, np.newaxis]

x_age = df.alter[:, np.newaxis]
x_sex = df.sex[:, np.newaxis] -1
x_met = df.met[:, np.newaxis]
x_cancer = df.cancer[:, np.newaxis]
x_Charlson = df.Charlson_index[:, np.newaxis]
x_chemo = df.chemo[:, np.newaxis]
x_radio = df.radio[:, np.newaxis]
x_surgery = df.surgery[:, np.newaxis]
x_chemrad = df.chemrad[:, np.newaxis]
x_chemsurg = df.chemsurg[:, np.newaxis]
x_radiosurg = df.radiosurg[:, np.newaxis]
x_chemradsurg = df.chemradsurg[:, np.newaxis]
x_urban = df.urban[:, np.newaxis]
x_rural = df.rural[:, np.newaxis]
x_thinly = df.thinly[:, np.newaxis]

x_age_shared = shared(x_age, broadcastable=(False, True))
x_sex_shared = shared(x_sex, broadcastable=(False, True))
x_met_shared = shared(x_met, broadcastable=(False, True))
x_cancer_shared = shared(x_cancer, broadcastable=(False, True))
x_Charlson_shared = shared(x_Charlson, broadcastable=(False, True))
x_chemo_shared = shared(x_chemo, broadcastable=(False, True))
x_radio_shared = shared(x_radio, broadcastable=(False, True))
x_surgery_shared = shared(x_surgery, broadcastable=(False, True))
x_chemrad_shared = shared(x_chemrad, broadcastable=(False, True))
x_chemsurg_shared = shared(x_chemsurg, broadcastable=(False, True))
x_radiosurg_shared = shared(x_radiosurg, broadcastable=(False, True))
x_chemradsurg_shared = shared(x_chemradsurg, broadcastable=(False, True))
x_urban_shared = shared(x_urban, broadcastable=(False, True))
x_rural_shared = shared(x_rural, broadcastable=(False, True))
x_thinly_shared = shared(x_thinly, broadcastable=(False, True))

def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

def stick_breaking(v):
    return v * tt.concatenate([tt.ones_like(v[:, :1]),
                               tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                              axis=1)

with pm.Model() as model:
    lambda0 = pm.Normal('lambda0', 0., 10., shape=K)
    lambda1 = pm.Normal('lambda1', 0., 10., shape=K)
    lambda2 = pm.Normal('lambda2', 0., 10., shape=K)
    lambda3 = pm.Normal('lambda3', 0., 10., shape=K)
    lambda4 = pm.Normal('lambda4', 0., 10., shape=K)
    lambda5 = pm.Normal('lambda5', 0., 10., shape=K)
    lambda6 = pm.Normal('lambda6', 0., 10., shape=K)
    lambda7 = pm.Normal('lambda7', 0., 10., shape=K)
    lambda8 = pm.Normal('lambda8', 0., 10., shape=K)
    lambda9 = pm.Normal('lambda9', 0., 10., shape=K)
    lambda10 = pm.Normal('lambda10', 0., 10., shape=K)
    lambda11 = pm.Normal('lambda11', 0., 10., shape=K)
    lambda12 = pm.Normal('lambda12', 0., 10., shape=K)
    lambda13 = pm.Normal('lambda13', 0., 10., shape=K)
    lambda14 = pm.Normal('lambda14', 0., 10., shape=K)
    lambda15 = pm.Normal('lambda15', 0., 10., shape=K)
    v = norm_cdf(lambda0 + lambda1 * x_age_shared + lambda2 * x_sex_shared + 
                 lambda3 * x_met_shared + lambda4 * x_cancer_shared + lambda5 * x_Charlson_shared +
                 lambda6 * x_chemo_shared + lambda7 * x_radio_shared + lambda8 * x_surgery_shared +
                 lambda9 * x_chemrad_shared + lambda10 * x_chemsurg_shared + lambda11 * x_radiosurg_shared +
                 lambda12 * x_chemradsurg_shared + lambda13 * x_urban_shared + lambda14 * x_rural_shared +
                 lambda15 * x_thinly_shared)
    w = pm.Deterministic('w', stick_breaking(v))
    
with model:
    beta0 = pm.Normal('beta0', 0., 10., shape=K)
    beta1 = pm.Normal('beta1', 0., 10., shape=K)
    beta2 = pm.Normal('beta2', 0., 10., shape=K)
    beta3 = pm.Normal('beta3', 0., 10., shape=K)
    beta4 = pm.Normal('beta4', 0., 10., shape=K)
    beta5 = pm.Normal('beta5', 0., 10., shape=K)
    beta6 = pm.Normal('beta6', 0., 10., shape=K)
    beta7 = pm.Normal('beta7', 0., 10., shape=K)
    beta8 = pm.Normal('beta8', 0., 10., shape=K)
    beta9 = pm.Normal('beta9', 0., 10., shape=K)
    beta10 = pm.Normal('beta10', 0., 10., shape=K)
    beta11 = pm.Normal('beta11', 0., 10., shape=K)
    beta12 = pm.Normal('beta12', 0., 10., shape=K)
    beta13 = pm.Normal('beta13', 0., 10., shape=K)
    beta14 = pm.Normal('beta14', 0., 10., shape=K)
    beta15 = pm.Normal('beta15', 0., 10., shape=K)
    mu = pm.Deterministic('mu', beta0 + beta1 * x_age_shared + beta2 * x_sex_shared + 
                 beta3 * x_met_shared + beta4 * x_cancer_shared + beta5 * x_Charlson_shared +
                 beta6 * x_chemo_shared + beta7 * x_radio_shared + beta8 * x_surgery_shared +
                 beta9 * x_chemrad_shared + beta10 * x_chemsurg_shared + beta11 * x_radiosurg_shared +
                 beta12 * x_chemradsurg_shared + beta13 * x_urban_shared + beta14 * x_rural_shared +
                 beta15 * x_thinly_shared)
    
with model:
    psi = pm.Lognormal('psi', 0., 2., shape=K)
    obs = pm.Mixture('obs', w, pm.NegativeBinomial.dist(mu, psi), observed=y_ANZ_KH)
    # zero-inflation:
    #obs = pm.Mixture('obs', w, pm.ZeroInflatedNegativeBinomial.dist(mu, tau), observed=y_ANZ_KH)


SAMPLES = 20000
BURN = 10000
THIN = 1

with model:
    step = pm.Metropolis()
    trace_ = pm.sample(SAMPLES, step, random_seed=41)
    
trace = trace_[BURN::THIN]

