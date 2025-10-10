import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def ChooseColor(row:pd.Series):
    """
    Придает формуле цвет на диаграмме Ван-Кревелина.
    """
    if "N" in row.columns:
        if "S" in row.columns:
            if row["S"] != 0:
                return "red" if row["N"] != 0 else "green"
            else:
                return "orange" if row["N"] != 0 else "blue"
        else:
            return "orange" if row["N"] != 0 else "blue"
    else:
        return "blue"
    

def draw_vk(spec: pd.DataFrame,
               ax = None):
    """
    Возвращает диаграмму Ван-Кревелина для подставленного спектра.
    """

    s = spec["intensity"]/spec["intensity"].median()
    size = s

    spec = spec.copy()
    spec["Color value"] = spec.apply(lambda x: ChooseColor(x) ,axis=1)

    if "O/C" not in list(spec.columns):

        spec["O/C"] = spec["O"]/spec["C"]
        spec["H/C"] = spec["H"]/spec["C"]

    if ax is None:

        fig, ax = plt.subplots(1,1,figsize=(6,6))

        ax.set_xlim((0.0,1.0))
        ax.set_ylim((0.0,2.2))
        ax.set_title(f"{spec.attrs['name']}, {spec.dropna().shape[0]} formulas")

        sns.scatterplot(data = spec, x = "O/C", y = "H/C",hue="Color value",hue_order=["blue","orange","green","red"], size = size, alpha = 0.7,legend = False,sizes=(4, 40))
    
    else:

        sns.scatterplot(data = spec, x = "O/C", y = "H/C",hue="Color value",hue_order=["blue","orange","green","red"], size = size, alpha = 0.7,legend = False,sizes=(4, 40),ax=ax)

        ax.set_xlim([0,1])
        ax.set_ylim([0,2.2])
        ax.set_title(f"{spec.attrs['name']}, {spec.dropna().shape[0]} formulas")