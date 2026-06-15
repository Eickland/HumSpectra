import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, Tuple, Optional

from ..utilits import utilits as ms_util
from ..assign import assign as ms_assign
from ..decorators import decorators as ms_dec


def gen_from_brutto(table: pd.DataFrame) -> pd.DataFrame:
    """
    Generate mass from brutto table

    Parameters
    ----------
    table: pandas Dataframe
        table with elemnt contnent

    Return
    ------
    pandas DataFrame
        Dataframe with elements and masses
    """
    masses = ms_util.get_elements_masses(table.columns.to_list())

    table["calc_mass"] = table.multiply(masses).sum(axis=1)
    table["calc_mass"] = np.round(table["calc_mass"], 6)
    table.loc[table["calc_mass"] == 0, "calc_mass"] = np.nan

    return table

@ms_dec._freeze
@lru_cache(maxsize=None)
def brutto_gen(elems: Optional[Dict[str, Tuple[int, int]]] = None, rules: bool = True) -> pd.DataFrame:
    """
    Generete brutto formula dataframe

    Parameters
    ----------
    elems: dict
        Dictonary with elements and their range for generate brutto table 
        Example: {'C':(1,60),'O_18':(0,3)} - content of carbon (main isotope) from 1 to 59,
        conent of isotope 18 oxygen from 0 to 2. 
        By default it is {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
    rules: bool
        Rules: 0.25<H/C<2.2, O/C < 1, nitogen parity, DBE-O <= 10. 
        By default it is on, but for tmds should be off

    Returns
    -------
    pandas Dataframe
        Dataframe with masses for elements content
    """

    working_elems = ms_dec._process_elems(elems)

    #load elements table. Generatete in mass folder
    elems_mass_table = ms_util.elements_table()
    elems_arr = []
    elems_dict = {}
    for el in working_elems:
        elems_arr.append(np.array(range(working_elems[el][0],working_elems[el][1])))
        if '_' not in el:
            temp = elems_mass_table.loc[elems_mass_table['element']==el].sort_values(by='abundance',ascending=False).reset_index(drop=True)
            elems_dict[el] = temp.loc[0,'mass']
        else:
            temp = elems_mass_table.loc[elems_mass_table['element_isotop']==el].reset_index(drop=True)
            elems_dict[el] = temp.loc[0,'mass']

    #generate grid with all possible combination of elements in their ranges
    t = np.array(np.meshgrid(*elems_arr)).T.reshape(-1,len(elems_arr))
    gdf = pd.DataFrame(t,columns=list(elems_dict.keys()))
    #do rules H/C, O/C, and parity
    if rules:
       gdf = ms_assign.filter_by_rules(gdf)

    #calculate mass
    masses = np.array(list(elems_dict.values()))
    gdf['mass'] = gdf.multiply(masses).sum(axis=1)
    gdf['mass'] = np.round(gdf['mass'], 6)

    gdf = gdf.sort_values("mass").reset_index(drop=True)

    return gdf
 