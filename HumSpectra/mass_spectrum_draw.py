import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def ChooseColor(row):
    """
    Придает формуле цвет на диаграмме Ван-Кревелина.
    """
    if "N" in row:
        if "S" in row:
            if row["S"] != 0:
                return "red" if row["N"] != 0 else "green"
            else:
                return "orange" if row["N"] != 0 else "blue"
        else:
            return "orange" if row["N"] != 0 else "blue"
    else:
        return "blue"
    

def vk(spec: pd.DataFrame,
               ax = None,
               sizes=(7, 30)):
    """
    Возвращает диаграмму Ван-Кревелина для подставленного спектра.
    """

    s = spec["intensity"]
    size = s

    spec = spec.copy(deep=True)
    spec["Color value"] = spec.apply(lambda x: ChooseColor(x) ,axis=1)

    if "O/C" not in list(spec.columns):

        spec["O/C"] = spec["O"]/spec["C"]
        spec["H/C"] = spec["H"]/spec["C"]

    if ax is None:

        fig, ax = plt.subplots(1,1,figsize=(6,6))

        ax.set_xlim((0.0,1.0))
        ax.set_ylim((0.0,2.2))
        ax.set_title(f"{spec.attrs['name']}, {spec.dropna().shape[0]} formulas")

        sns.scatterplot(data = spec, x = "O/C", y = "H/C",hue="Color value",hue_order=["blue","orange","green","red"], size = size, alpha = 0.7,legend = False,sizes=sizes)
    
    else:

        sns.scatterplot(data = spec, x = "O/C", y = "H/C",hue="Color value",hue_order=["blue","orange","green","red"], size = size, alpha = 0.7,legend = False,sizes=sizes,ax=ax)

        ax.set_xlim([0,1])
        ax.set_ylim([0,2.2])
        ax.set_title(f"{spec.attrs['name']}, {spec.dropna().shape[0]} formulas")