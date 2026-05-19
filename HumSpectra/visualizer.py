import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from itertools import combinations

import HumSpectra.utilits as ut

feature_names_translate_dict = {
    'org_carbon' : 'Орг. углерод, мг/л'
}

def box_plot_default(MainData, path,target_column='Class',dpi=150, save=False,median_label_fontsize=13,
                     ylabel_fontsize=19,yticks_fontsize=19,legend_fontsize=16):
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    
    for feature in numeric_cols:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=dpi)
        box = sns.boxplot(data = MainData.reset_index(),
                            hue=target_column, y=feature, ax=axes)
        ut.add_median_labels(box,fontsize=median_label_fontsize)
        plt.legend(loc='upper right', bbox_to_anchor = (1, 0.5, 0.3, 0.5),fontsize=legend_fontsize)
        plt.ylabel(feature,fontsize=ylabel_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
            
        plt.tight_layout()
                
        if save:
            plt.savefig(path + fr'\{feature}.png')
            plt.close()

def hist_plot_default(MainData, path, target_column='Class',dpi=150, save=False,
                     xlabel_fontsize=19,xticks_fontsize=19,legend_fontsize=16):
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    
    for feature in numeric_cols:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=dpi)
        sns.histplot(data = MainData, hue=target_column, x=feature, ax=axes,
                        multiple="stack",palette="rainbow",edgecolor=".3",linewidth=.5,log_scale=False,)
        plt.xlabel(feature,fontsize = xlabel_fontsize)
        plt.xticks(fontsize = xticks_fontsize)
        plt.legend(fontsize = legend_fontsize)
            
        if feature == 'org_carbon':
            plt.xlabel(feature_names_translate_dict['org_carbon'])
            plt.title(f'Распределение {feature_names_translate_dict['org_carbon']}')
                
        if save:
            plt.savefig(path + fr'\{feature}.png')
            plt.close()
             
def swarm_plot_default(MainData, path, target_column='Class',dpi=150, save=False,size=16,
                       ylabel_fontsize=19,xticks_fontsize=19):
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    
    for feature in numeric_cols:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=dpi)
        sns.swarmplot(data = MainData, x=target_column, y=feature,hue=target_column,legend=False, ax=axes,
                        palette="rainbow",edgecolor=".3",linewidth=.5,log_scale=False,size=size)
        plt.xticks(rotation=45,fontsize=xticks_fontsize)
        plt.ylabel(feature,fontsize=ylabel_fontsize)
        
        if feature == 'org_carbon':
            plt.ylabel(feature_names_translate_dict['org_carbon'],fontsize=ylabel_fontsize)
            plt.title(f'Распределение {feature_names_translate_dict['org_carbon']}')
            
        plt.tight_layout()
            
        if save:
            plt.savefig(path + fr'\{feature}.png')
            plt.close()
            
def strip_plot_default(MainData, path, target_column='Class',dpi=150, save=False,size=12,
                       ylabel_fontsize=19,xticks_fontsize=19,legend_fontsize=16):
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    
    for feature in numeric_cols:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=dpi)
        sns.stripplot(data=MainData, x=target_column, y=feature, hue=target_column,
                    legend=False, ax=axes, palette="rainbow",
                    edgecolor=".3", linewidth=.5, 
                    jitter=0.3,  # Добавляет случайный шум по горизонтали
                    size=size)
        plt.xticks(rotation=45,fontsize=xticks_fontsize)
        plt.ylabel(feature,fontsize=ylabel_fontsize)
        
        if feature == 'org_carbon':
            plt.ylabel(feature_names_translate_dict['org_carbon'],fontsize=ylabel_fontsize)
            plt.title(f'Распределение {feature_names_translate_dict['org_carbon']}')
                
        if save:
            plt.savefig(path + fr'\{feature}.png')
            plt.close()            
                
def scatter_plot_default(MainData, path, target_column='Class',dpi=150, save=False, label_class=None,
                         ylabel_fontsize=19,xlabel_fontsize=19,xticks_fontsize=19,yticks_fontsize=19,legend_fontsize=16):
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    combinations_list = list(combinations(numeric_cols, 2))
    
    for feature_1, feature_2 in combinations_list:
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
        
        sns.scatterplot(data=MainData, hue=target_column, y=feature_2, x=feature_1, 
                       ax=axes, alpha=0.6, legend=True)
        
        plt.ylabel(feature_2,fontsize = ylabel_fontsize)
        plt.xlabel(feature_1,fontsize = xlabel_fontsize)
        plt.yticks(fontsize=yticks_fontsize)
        plt.xticks(fontsize=xticks_fontsize)
        plt.legend(fontsize=legend_fontsize)
        
        plt.tight_layout()
        if save:
            plt.savefig(path + fr'\{feature_1}_{feature_2}.png')
            plt.close()
    
    return 
  