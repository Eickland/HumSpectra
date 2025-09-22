import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def MolClassesSpectrum(specList,draw=True,ax=None):

    density_class_list=[]
    sample_list = []
    for i in range(len(specList)):
        spec = specList[i]
        data_class = spec.get_mol_class(how="perminova")
        density_class_list.append(data_class["density"])
        sample_list.append(spec.metadata["name"])
        mol_class_list=data_class["class"].to_list()
    mol_class_data = pd.DataFrame(np.array(density_class_list), index=sample_list, columns=mol_class_list)
    df_reversed = mol_class_data.sort_index(ascending=False).copy()
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8, 6))
    df_reversed.plot(kind='barh', stacked=True, ax=ax)
    # Настройки графика
    ax.set_title('Распределение классов по образцам')
    ax.set_xlabel('Доля')
    ax.set_ylabel('Образец')
    #plt.xticks(rotation=30)  # Чтобы подписи по оси X не наклонялись
    ax.legend(title='Классы', bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда справа
    # Показать график
    plt.tight_layout()

    return df_reversed


def CalcMetricSpectrum(specList,func="weight",draw=True):

    list_data = []
    for i in range(len(specList)):
        spec = specList[i]
        data = spec.get_mol_metrics(func=func)
        data.rename(columns={"value": f"{spec.metadata['name']}"},inplace=True)
        if i == 0:
            data_prev = data
            continue
        else:
            data_prev=data.merge(data_prev, on="metric")

    return data_prev


def FormulaSpecData(specList, draw=True):
    all_formula_list = []
    sample_list = []
    CHON_formula_list = []
    CHOS_formula_list = []
    CHONS_formula_list = []
    dict_list = []
    data = pd.DataFrame()
    for i in range(len(specList)):
        spec = specList[i]

        sample_list.append(spec.metadata["name"])

        all_formula_list.append(spec.table.dropna().shape[0])

        dict = spec.table[["N","S"]].value_counts().to_dict()

        result_CHON = [key for key in dict.keys() if key[0] > 0 and key[1] < 1]
        result_CHOS = [key for key in dict.keys() if key[0] < 1 and key[1] > 0]
        result_CHONS = [key for key in dict.keys() if key[0] > 0 and key[1] > 0]
        sum_CHON = []
        sum_CHOS = []
        sum_CHONS = []
        for res in result_CHON:
            sum_CHON.append(dict[res])
        for res in result_CHOS:
            sum_CHOS.append(dict[res])
        for res in result_CHONS:
            sum_CHONS.append(dict[res])
        CHON_formula_list.append(sum(sum_CHON))
        CHOS_formula_list.append(sum(sum_CHOS))
        CHONS_formula_list.append(sum(sum_CHONS))
        dict_list.append(dict)
    data["All formulas"] = all_formula_list
    data["CHON, formulas"] = CHON_formula_list
    data["CHOS, formulas"] = CHOS_formula_list
    data["CHONS, formulas"] = CHONS_formula_list
    data["Dict, {N, S}: count"] = dict_list
    data["Sample name"] = sample_list
    data.set_index("Sample name",inplace=True)

    if draw:
        data.plot(kind='bar', stacked=False, figsize=(10, 8))
        plt.xticks(rotation=30)
        plt.tight_layout()
        
    return data