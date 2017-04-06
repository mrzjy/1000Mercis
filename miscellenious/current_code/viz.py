# -*- coding: utf-8 -*-
"""
@author: plkovac
"""

"""
Viz = understanding data and vizualize repartition
"""

#==============================================================================
# Libraries
#==============================================================================

import numpy as np
import pandas as pd
from math import ceil
import dateutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#==============================================================================
# Data retrieval
#==============================================================================
def load_data(shuffle=False):
    # wdir=''
    path_neg = '/home/pl_vernhet/cookies_data/v1/negative_tuid_centrale_finale_with_header.csv'
    path_pos = '/home/pl_vernhet/cookies_data/v1/positive_tuid_centrale_finale_with_header.csv'

    data_neg = pd.read_csv(path_neg, encoding = 'utf-8', sep = '|')
    data_pos = pd.read_csv(path_pos, encoding = 'utf-8', sep = '|')
    
    data_neg["label"]=0
    data_pos["label"]=1
    data = pd.concat( [data_neg, data_pos] )
    data_neg['visits']=1
    data_pos['visits']=1
    data['visits']=1
    if shuffle:
        data.reindex( np.random.permutation( data.index ) )
    print 'Data imported and formated \n'
    print 'data_pos : %d lines' %( data_pos.shape[0] ) 
    print 'data_neg : %d lines \n' %( data_neg.shape[0] )
    return data_neg, data_pos, data

#==============================================================================
# I - First statistics & visualization
#==============================================================================

def first_stats(d1, d2, ds, viz=False, names = ['csp-', 'csp+','all']):
    inputs = [d1,d2,ds]
    nb = []

    # rajouter .groupby('domaine')['tuid'].nunique()  -->
    # rajouter .groupby('tuid')['domaine'].nunique()  -->
    
    for e, idx in zip( inputs, names):
        
        print '************************************************'
        print '      COOKIES %s   ARE BROWSING ' % (idx)
        print '************************************************'
        print '\n'
        stat_cookies = e.groupby('tuid')['visits'].sum()
        print '%s are browsing:' %(idx)
        print  'Mean = %.2f' % ( stat_cookies.mean() ) 
        print  'Std = %.2f' %( stat_cookies.std() ) 
        nb += [stat_cookies.shape[0]]

        if viz:
            xmax = stat_cookies.max()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.hist(stat_cookies, bins=300, range=(1,xmax))
            plt.title('Histogram for %s cookies browsing websites' %(idx))
            plt.xlabel('browsed websites')
            ax.set_xlim(1, ceil(1.05*xmax))
            plt.xscale('log', nonposy='clip')
            plt.yscale('log', nonposy='clip')
            name = 'Histo_cookies_'+idx
            plt.savefig('../img/'+name)

    for e, idx in zip( inputs, names):
        print '\n'
        print '************************************************'
        print '      DOMAINES ARE VISITED BY %s' % (idx) 
        print '************************************************'
        print '\n'
        stat_domaines = e.groupby('domaine')['visits'].sum()
        print 'Domains are being visited by %s'%(idx)
        print  'Mean = %.2f'%( stat_domaines.mean()) 
        print  'Std = %.2f' %( stat_domaines.std()) 
        
        if viz:
            xmax = stat_domaines.max()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.hist(stat_domaines, bins=300, range=(1,xmax))
            plt.title('Histogram for domains visited by %s' %(idx))
            plt.xlabel('Domains visits')
            ax.set_xlim(1, ceil(1.05*xmax))
            plt.xscale('log', nonposy='clip')
            plt.yscale('log', nonposy='clip')
            name = 'Histo_domaines_'+idx
            plt.savefig('../img/'+name)

    for idx, count in zip(names,nb):
        print '   %s : %d' %(idx,count)
        print '\n'
    print '~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n'
    
def metrics(data, names = ['csp-', 'csp+']):
    N = {}

    for i,idx in zip(range(2),names):    
        N[idx] = data[data['label'] == i].groupby('domaine')['tuid'].nunique()
        print "debug 1:"
        N[idx] = N[idx].to_frame()
        print N[idx].columns
    contrast = N[names[1]].join(N[names[0]], how='outer', lsuffix=names[1], rsuffix=names[0]).fillna(0)
    print contrast.describe()
    score = contrast.apply( lambda row: ( row['0'+names[1] ]-row['0'+names[0] ])/( row['0'+names[1] ]+row['0'+names[0] ]), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(score, bins=10, range=(-1,1))
    plt.title('Histogram for domaines-score repartition')
    plt.xlabel('Domaines scores')
    ax.set_xlim(-1, 1)
    plt.yscale('log', nonposy='clip')
    name = 'Histo_scores_domaines'
    plt.savefig('../img/'+name)  
    
#==============================================================================
# Script launch
#==============================================================================
 
if __name__ == '__main__':
    d_neg, d_pos, data = load_data()
    first_stats(d_neg, d_pos, data, viz=True)
    #metrics(data)
    
    
    
    
