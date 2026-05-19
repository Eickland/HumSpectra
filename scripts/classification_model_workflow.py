import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_classification, random_forest_classification
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection

CACHE_DATA_PATH = Path('')  #Путь к папке, куда лежит база данных
CLASSIFICATION_PATH = Path('') #Путь к папке с результатами

GrindBestCombEnabled = False
SingleModelEnabled = True

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0.25)
data.dropna(inplace=True)

MainData = ut.filter_by_iqr_per_group(data,iqr_param=2.5,target_column='Subclass', filter_columns=["asm_350",'Suva'])

# Создаем маски для разных условий
mask_adom_2025 = (MainData['sample_class'] == 'ADOM') & (MainData['Sample'].str[-4:] == '2025')

# Отделяем строки с Baikal для случайного разделения
baikal_df = MainData[MainData['sample_class'] == 'Baikal'].copy()

# Случайно делим Baikal строки на две части
np.random.seed(42)  # для воспроизводимости
baikal_indices = baikal_df.index.tolist()
np.random.shuffle(baikal_indices)
half_baikal_count = len(baikal_indices) // 2

# Индексы Baikal для первой части
baikal_part1_indices = baikal_indices[:half_baikal_count]

# Создаем финальную маску для первой части
mask_part1 = mask_adom_2025 | MainData.index.isin(baikal_part1_indices)

# Разделяем DataFrame
test_data = MainData[mask_part1].copy()
train_data = MainData[~mask_part1].copy()

MainData.set_index(["Class","Subclass","Sample"],inplace=True)
train_data.set_index(["Class","Subclass","Sample"],inplace=True)
test_data.set_index(["Class","Subclass","Sample"],inplace=True)

if GrindBestCombEnabled:

    feature_columns = MainData.select_dtypes(include=['float']).columns

    accuracy_lda = []
    weight_accuracy_lda=[]
    ADOM_accuracy_lda = []
    ADOM_accuracy_lda_cross_val = []
    
    accuracy_rf = []
    weight_accuracy_rf=[]
    ADOM_accuracy_rf = []
    ADOM_accuracy_rf_cross_val = []    
    
    combinations_2Dlist = []
    count_feature_list = []

    num_fearure_list = [1,2,3]
    target = 1
    
    
    num_iterations = 0
    k = 1
    
    for num_fearure_max in num_fearure_list:
        num_iterations += len(list(combinations(feature_columns, num_fearure_max)))

    for num_fearure_max in num_fearure_list:
    
        combinations_list = list(combinations(feature_columns, num_fearure_max))

        for feature_comb in combinations_list:
            
            print(f'{k}/{num_iterations}')
            print(feature_comb)
            feature_comb = list(feature_comb)

            res_lda = lda_classification(MainData[feature_comb],index_level=target,cv_dataset=test_data[feature_comb],external_validation=False)
            weight_accuracy_lda.append(res_lda[-4])
            accuracy_lda.append(res_lda[-5])
            #ADOM_accuracy_lda.append(res_lda[-2]['precision'].loc['ADOM'])
            #ADOM_accuracy_lda_cross_val.append(res_lda[-1]['classification_report']['ADOM']['precision']) # type: ignore

            res_rf = random_forest_classification(MainData[feature_comb],index_level=target,cv_dataset=test_data[feature_comb],external_validation=False)
            weight_accuracy_rf.append(res_rf[-4])
            accuracy_rf.append(res_rf[-5])
            #ADOM_accuracy_rf.append(res_rf[-2]['precision'].loc['ADOM'])
            #ADOM_accuracy_rf_cross_val.append(res_rf[-1]['classification_report']['ADOM']['precision']) # type: ignore      
                  
            combinations_2Dlist.append(feature_comb)
            count_feature_list.append(num_fearure_max)
            
            k+=1
            
    accuracy_data = pd.DataFrame()
    
    accuracy_data['LDA_weight_accuracy'] = weight_accuracy_lda
    accuracy_data['LDA_accuracy'] = accuracy_lda    
    accuracy_data['RF_weight_accuracy'] = weight_accuracy_rf
    accuracy_data['RF_accuracy'] = accuracy_rf
    
    #accuracy_data['ADOM_accuracy_lda'] = ADOM_accuracy_lda
    #accuracy_data['ADOM_accuracy_lda_CV'] = ADOM_accuracy_lda_cross_val
    #accuracy_data['ADOM_accuracy_rf'] = ADOM_accuracy_rf
    #accuracy_data['ADOM_accuracy_rf_CV'] = ADOM_accuracy_rf_cross_val
    
    accuracy_data['feature'] = combinations_2Dlist
    accuracy_data['num_feature'] = count_feature_list

    main_model_output_save_path = Path.joinpath(Path.cwd(),"Server","Received_data","Model",f"Model_accuracy_ADOM_subclass_2025_cross_val.xlsx")
    
    accuracy_data.sort_values('LDA_weight_accuracy',ascending=False).to_excel(main_model_output_save_path)

if SingleModelEnabled:
    
    feature_comb = ['Component_3', 'Suva', 'r412_547']

    res_lda = lda_classification(MainData[feature_comb],index_level=1,
                                output_html_path=str(Path.joinpath(CLASSIFICATION_PATH,"LDA",f"LDA_{feature_comb}.html"),
                                                    ),
                                external_validation=False,
                                cv_dataset=test_data[feature_comb]
                                                    )

    res_rf = random_forest_classification(MainData[feature_comb],index_level=1,
                                output_html_path=str(Path.joinpath(CLASSIFICATION_PATH,"RF",f"RF_{feature_comb}.html"),
                                                    ),
                                external_validation=False,
                                cv_dataset=test_data[feature_comb]
                                                    )

#model_data = pd.read_excel(Path.joinpath(Path.cwd(),"Server","Received_data","Model",f"Model_accuracy_cross-val.xlsx"))



