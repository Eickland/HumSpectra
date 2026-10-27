import copy
import pandas as pd
import numpy as np
from typing import Optional, Any

from ..utilits import utilits as ms_util
from ..brutto import brutto as ms_brutto

def filter_by_rules(gdf: pd.DataFrame) -> pd.DataFrame:
    
    data = copy.deepcopy(gdf)
    temp = copy.deepcopy(data)
    temp = ms_util._merge_isotopes(temp)

    if 'C' not in temp or 'H' not in temp or 'O' not in temp:
        raise Exception('For applying rules in brutto must be CHO elements or their isotopes')
        
    temp['H/C'] = temp['H']/temp['C']
    temp['O/C'] = temp['O']/temp['C']
    data = data.loc[(temp['H/C'] < 2.2) & (temp['H/C'] > 0.25) & (temp['O/C'] < 1)]

    if 'N' not in temp:
        temp['N'] = 0
        
    temp['DBE-O'] = 1.0 + temp["C"] - 0.5 * temp["H"] + 0.5 * temp['N'] - temp['O'] # type: ignore
    data = data.loc[temp['DBE-O'] <= 10]
        
    if 'N' in temp:
        temp['parity'] = (temp['H'] + temp['N'])%2
        data = data.loc[temp['parity']==0]
    
    return data

def filter_bruttos(table: pd.DataFrame,
                   rules: Optional[dict] = None) -> pd.DataFrame:
    """
    Фильтрация таблицы сгенерированных брутто-формул.

    rules (все ключи необязательные):
      'hc': (min, max)        диапазон H/C
      'oc': (min, max)        диапазон O/C
      'nc': (min, max)        диапазон N/C
      'sc': (min, max)        диапазон S/C
      'dbe': (min, max)       двойные связи/циклы: C - H/2 + N/2 + 1
      'elements': {'N': (0, 2), 'S': (0, 1), ...}  диапазоны чисел атомов
      'mass_dependent': [ {'mass': (550, None), 'hc': (1.0, 2.2), ...}, ... ]
                          правила, действующие только для формул в диапазоне масс
      'custom': callable(table) -> bool-маска той же длины
    Границу можно задать None (без ограничения).
    """
    if not rules:
        return table

    keep = _rules_mask(table, rules)

    for sub in rules.get('mass_dependent', []):
        lo, hi = sub.get('mass', (None, None))
        in_range = np.ones(len(table), dtype=bool)
        if lo is not None:
            in_range &= table['mass'].to_numpy() >= lo
        if hi is not None:
            in_range &= table['mass'].to_numpy() <= hi
        # вне диапазона масс правило не действует, внутри - обязательно
        keep &= ~in_range | _rules_mask(table, sub)

    if 'custom' in rules:
        keep &= np.asarray(rules['custom'](table), dtype=bool)

    return table.loc[keep]          # порядок строк (сортировка по mass) сохраняется


def _between(values: np.ndarray, bounds) -> np.ndarray:
    lo, hi = bounds
    ok = np.ones(len(values), dtype=bool)
    if lo is not None:
        ok &= values >= lo
    if hi is not None:
        ok &= values <= hi
    return ok


def _rules_mask(table: pd.DataFrame, rules: dict) -> np.ndarray:
    n = len(table)
    keep = np.ones(n, dtype=bool)

    def col(name):
        return table[name].to_numpy(dtype=float) if name in table.columns else np.zeros(n)

    C = col('C') + col('C_13')       # полный углерод с учетом 13C
    H, O, N, S = col('H'), col('O'), col('N'), col('S')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = {'hc': H / C, 'oc': O / C, 'nc': N / C, 'sc': S / C}

    for key, vals in ratios.items():
        if key in rules:
            keep &= _between(np.nan_to_num(vals, nan=np.inf), rules[key])   # C=0 -> отбрасывается

    if 'dbe' in rules:
        keep &= _between(C - H / 2 + N / 2 + 1, rules['dbe'])

    for el, bounds in rules.get('elements', {}).items():
        if el in table.columns:
            keep &= _between(col(el), bounds)

    return keep

def assign_optimized(data: pd.DataFrame,
            brutto_dict: Any|None = None,
            generated_bruttos_table: Optional[pd.DataFrame] = None,
            rel_error: float|None = None,
            abs_error: float|None = None,
            sign: str ='-',
            mass_min: Optional[float] =  None,
            mass_max: Optional[float] = None,
            intensity_min: Optional[float] =  None,
            intensity_max: Optional[float] = None,
            charge_max: int = 1,
            nitrogen_precision_factor: float = 2.0,
            sulfur_precision_factor: float = 2.0,
            unique_formulas: bool = True,
            unique_priority: str = 'error',
            formula_filter={
                'hc': (0.3, 2.2),
                'oc': (0.0, 0.8),
                'elements': {'N': (0, 1), 'S': (0, 1)},
                'mass_dependent': [
                    {'mass': (550, None), 'hc': (1.2, 2.2), 'oc': (0.2, 0.7)},
                ],
            },
    ) -> pd.DataFrame:
        """
        Assigning brutto formulas to signal by mass
        
        Parameters
        -----------
        brutto_dict: dict
            Optional. Deafault None.
            Custom Dictonary for generate brutto table.
            Example: {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'C_13':(0,3)}
        generated_bruttos_table: pandas DataFrame 
            Optional. Contain column 'mass' and elements, 
            should be sorted by 'mass'.
            Can be generated by function brutto_generator.brutto_gen(). 
            if 'None' generate table with default elemnets and ranges
            C: 4-50, H 4-100, O 0-25, N 0-3, S 0-2.
        rel_error: float
            Optional. default 0.5, permissible error in ppm for assign mass to brutto formulas
        abs_error: float
            Optional. default None, permissible absolute error for assign mass to brutto formulas
        sign: str
            Optional. Deafult '-'.
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            '0' for neutral
        mass_min: float
            Optional. Default None. Minimall mass for assigment
        mass_max: float
            Optional. Default None. Maximum mass for assigment
        intensity_min: float
            Optional. Default None. Minimall intensity for assigment
        intensity_max: float
            Optional. Default None. Maximum intensity for assigment
        charge_max: int
            Maximum charge in m/z. Default 1.
        nitrogen_precision_factor: float
            Optional. Default 2.0. Factor by which to increase precision for formulas containing nitrogen
        sulfur_precision_factor: float
            Optional. Default 2.0. Factor by which to increase precision for formulas containing sulfur
        unique_formulas: bool
            Optional. Default True. Each brutto formula can be assigned to only one peak.
            If the best formula of a peak is already taken, the next suitable formula
            within the error window is used, otherwise the peak stays unassigned.
        unique_priority: str
            Optional. Default 'error'. Order in which (peak, formula) pairs are distributed:
            'error' - pairs with smaller mass error first,
            'intensity' - more intense peaks choose first (then by error).

        Return
        ------
        pd.DataFrame 
        """

        name = data.attrs['name']

        if generated_bruttos_table is None:
            generated_bruttos_table = ms_brutto.brutto_gen(brutto_dict)
        generated_bruttos_table = filter_bruttos(generated_bruttos_table, formula_filter)
        if generated_bruttos_table.empty:
            raise ValueError('formula_filter отбросил все формулы')

        if mass_min is None:
            mass_min = data['mass'].min()
        if mass_max is None:
            mass_max = data['mass'].max()
        if intensity_min is None:
            intensity_min = data['intensity'].min()
        if intensity_max is None:
            intensity_max = data['intensity'].max()
        
        if sign == '-':
            mass_shift = - 0.00054858 + 1.007825  # electron and hydrogen mass
        elif sign == '+':
            mass_shift = 0.00054858  # electron mass
        elif sign == '0':
            mass_shift = 0
        else:
            raise Exception('Sended sign to assign method is not correct. May be "+","-","0"')

        data.attrs['sign'] = sign

        if rel_error is not None and abs_error is not None:
            raise Exception('one of rel_error or abs_error must be None in assign method')
        
        if rel_error is not None:
            rel = True
            
        elif abs_error is not None:
            rel = False
        else:
            rel = True
            rel_error = 0.5  

        data = data.loc[:,['mass', 'intensity']].reset_index(drop=True)


        elems = list(generated_bruttos_table.drop(columns=["mass"]))

        # 1. Подготовка данных в NumPy
        target_masses = data['mass'].to_numpy()
        target_intensities = data['intensity'].to_numpy()
        ref_masses = generated_bruttos_table["mass"].to_numpy()
        
        # Элементы и их маски
        elems = [e for e in generated_bruttos_table.columns if e != 'mass']
        ref_elements = generated_bruttos_table[elems].values
        
        n_idx = elems.index('N') if 'N' in elems else -1
        s_idx = elems.index('S') if 'S' in elems else -1
        
        # Предрассчитаем маски для азота и серы в справочнике
        has_n = ref_elements[:, n_idx] > 0 if n_idx != -1 else np.zeros(len(ref_masses), dtype=bool)
        has_s = ref_elements[:, s_idx] > 0 if s_idx != -1 else np.zeros(len(ref_masses), dtype=bool)
        has_special = has_n | has_s

        # Итоговые массивы для хранения результатов
        final_assign = np.zeros(len(data), dtype=bool)
        final_charge = np.ones(len(data), dtype=int)
        final_elements = np.zeros((len(data), len(elems)))

        # Фильтр по диапазонам (делаем один раз маской)
        valid_mask = (target_masses >= mass_min) & (target_masses <= mass_max) & \
            (target_intensities >= intensity_min) & (target_intensities <= intensity_max)

        # Формулы, уже приписанные какому-либо пику (для unique_formulas)
        used_formulas = np.zeros(len(ref_masses), dtype=bool)

        # 2. Основной цикл только по зарядам (их мало!)
        for charge in range(1, charge_max + 1):
            if final_assign.all(): break # Если всё уже приписали, выходим

            # Еще не приписанные и входящие в диапазоны
            todo_mask = valid_mask & (~final_assign)
            if not todo_mask.any(): continue

            m = (target_masses[todo_mask] + mass_shift) * charge
            base_limit = (rel_error if rel else abs_error) / charge # type: ignore

            # Все формулы-кандидаты в окне допуска (факторы N/S окно только сужают)
            tol = m * base_limit * 1e-6 if rel else np.full(len(m), base_limit)
            lo = np.searchsorted(ref_masses, m - tol, side='left')
            hi = np.searchsorted(ref_masses, m + tol, side='right')
            counts = hi - lo
            if counts.sum() == 0: continue

            # Развертываем пары (пик, формула)
            peak_ids = np.repeat(np.arange(len(m)), counts)
            offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            ref_ids = np.repeat(lo, counts) + offsets

            # Расчет ошибок
            diffs = np.abs(ref_masses[ref_ids] - m[peak_ids])
            errors = diffs / m[peak_ids] * 1e6 if rel else diffs

            # Ужесточаем лимит для N и S
            precision_limit = np.full(len(ref_ids), base_limit, dtype=float)
            if n_idx != -1:
                precision_limit = np.where(has_n[ref_ids], precision_limit / nitrogen_precision_factor, precision_limit)
            if s_idx != -1:
                precision_limit = np.where(has_s[ref_ids], precision_limit / sulfur_precision_factor, precision_limit)

            ok = (errors <= precision_limit) & ~used_formulas[ref_ids]
            peak_ids, ref_ids, errors = peak_ids[ok], ref_ids[ok], errors[ok]
            if len(peak_ids) == 0: continue

            # Порядок рассмотрения пар: лучшие первыми
            if unique_priority == 'error':
                order = np.argsort(errors, kind='stable')
            elif unique_priority == 'intensity':
                peak_int = target_intensities[todo_mask][peak_ids]
                order = np.lexsort((errors, -peak_int))
            else:
                raise ValueError('unique_priority must be "error" or "intensity"')
            peak_ids, ref_ids = peak_ids[order], ref_ids[order]

            if unique_formulas:
                # Жадное назначение: каждый пик и каждая формула используются не более одного раза,
                # пик, чья лучшая формула уже занята, берет следующую подходящую
                peak_taken = np.zeros(len(m), dtype=bool)
                sel_peaks, sel_refs = [], []
                for p, r in zip(peak_ids.tolist(), ref_ids.tolist()):
                    if peak_taken[p] or used_formulas[r]:
                        continue
                    peak_taken[p] = True
                    used_formulas[r] = True
                    sel_peaks.append(p)
                    sel_refs.append(r)
                sel_peaks = np.asarray(sel_peaks, dtype=int)
                best_ref_indices = np.asarray(sel_refs, dtype=int)
            else:
                # Для каждого пика лучшая подходящая формула
                sel_peaks, first = np.unique(peak_ids, return_index=True)
                best_ref_indices = ref_ids[first]

            # Записываем результат
            global_indices = np.where(todo_mask)[0][sel_peaks]
            final_assign[global_indices] = True
            final_charge[global_indices] = charge
            final_elements[global_indices] = ref_elements[best_ref_indices]

        # 3. Сборка результата
        res_df = pd.DataFrame(final_elements, columns=elems)
        res_df['assign'] = final_assign
        res_df['charge'] = final_charge
        
        # Объединяем с исходными массами
        out = pd.concat([data[['mass', 'intensity']], res_df], axis=1)
        out.attrs['name'] = name
        return out
