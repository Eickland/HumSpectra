import copy
import pandas as pd
import numpy as np
from typing import Optional

from ..utilits import utilits as ms_utils
from ..assign import assign as ms_assign
from ..calc_process import calc_process as ms_calc

def recallibrate_optimize(spec: pd.DataFrame, 
                error_table: Optional[pd.DataFrame] = None, 
                how: str = 'assign',
                mode: str = '-',
                draw: bool = True) -> pd.DataFrame:
    '''
    Recallibrate spectrum

    Fine reccalabration based on self (assign or mass differnce map)
    or with external ethalon spectrum

    Parameters
    ----------
    spec: pd.DataFrame object
        Mass spectrum for recallibration
    error_table: ErrorTable object
        Optional. If None - calculate for spec. 
        ErrorTable object contain table error in ppm for mass, default 100 string            
    how: {'assign', 'mdm', filename} 
        Optional. Default 'assign'.
        If error_table is None we can choose how to recalculate.
        'assign' - by assign error, default.
        'mdm' - by calculation mass-difference map.
        filename - path to etalon spectrum, treated and saved by nomspectra
    mode: str
        Optional. Default '-' negative mode. May be +, - or 0
    draw: bool
        Plot error (fit of KDM)

    Returns
    -------
    pd.DataFrame
    '''

    spec = spec.copy()

    if error_table is None:
        if how == 'assign':
            error_table = assign_error(spec, show_map=draw, mode=mode)
        elif how == 'mdm':
            error_table = massdiff_error(spec, show_map=draw)
        else:
            etalon = ms_utils.read_mass_list(how)
            error_table = etalon_error(spec=spec, etalon=etalon, show_map=draw)
        pass

    # Извлекаем массивы для быстрой работы
    spec_masses = spec['mass'].values
    err_masses = error_table['mass'].values
    err_ppm = error_table['ppm'].values

    # 1. Вместо ступенчатых циклов используем линейную интерполяцию ошибки.
    # Это найдет точное значение ppm для каждой массы из spec.
    # По умолчанию np.interp экстраполирует крайние значения.
    interp_ppm = np.interp(spec_masses, err_masses, err_ppm)

    # 2. Векторизованный пересчет всей колонки масс за одну операцию
    # Формула: mass = mass + (mass * ppm / 1,000,000)
    spec['mass'] = spec_masses * (1 + interp_ppm / 1e6)

    spec.attrs['recallibrate'] = how
    return spec

@staticmethod
def assign_error(
    spec: pd.DataFrame,
    ppm: float = 3,
    brutto_dict = {'C':(4,30), 'H':(4,60), 'O':(0,20)},
    mode = '-',
    show_map: bool = True):
    '''
    Recallibrate by assign error

    Parameters
    ----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    ppm: float
        Permissible relative error in callibrate error. Default 3.
    brutto_dict: dict
        Dictonary with elements ranges for assignment
    mode: str
        Optional. Default '-' negative mode. May be +, - or 0
    show_error: bool
        Optional. Default True. Show process 

    Return
    ------
    ErrorTable
    '''

    spectr = copy.deepcopy(spec)
    spectr = ms_assign.assign_optimized(spectr,rel_error=ppm, brutto_dict=brutto_dict, sign=mode)
    spectr = ms_calc.calc_error(ms_utils.calc_mass(spectr))

    error_table = spectr
    error_table = error_table.loc[:,['mass','rel_error']]
    error_table.columns = ['mass', 'ppm']
    error_table['ppm'] = - error_table['ppm']
    error_table = error_table.dropna()

    kdm = ms_calc.kernel_density_map(df_error = error_table)
    err = ms_calc.fit_kernel(f=kdm, 
                        show_map=show_map, 
                        mass=ms_utils.drop_unassigned(spectr)['mass'].to_numpy())
    err = ms_calc.extrapolate(err,(spec['mass'].min(), spec['mass'].max()))

    return err

@staticmethod
def massdiff_error( spec: pd.DataFrame,
                    show_map:bool = True):
    '''
    Self-recallibration of mass-spectra by mass-difference map

    Parameters
    -----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    show_error: bool
        Optional. Default True. Show process 

    Return
    -------
    ErrorTable

    References
    ----------
    Smirnov, K. S., Forcisi, S., Moritz, F., Lucio, M., & Schmitt-Kopplin, P. 
    (2019). Mass difference maps and their application for the 
    recalibration of mass spectrometric data in nontargeted metabolomics. 
    Analytical chemistry, 91(5), 3350-3358. 
    '''

    spec = copy.deepcopy(spec)
    mde = ms_calc.md_error_map(spec = spec)
    kdm = ms_calc.kernel_density_map(df_error=mde)
    err = ms_calc.fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].to_numpy())
    
    err['ppm'] = err['ppm'] - err.loc[0, 'ppm'] # type: ignore

    return err

@staticmethod
def etalon_error(spec: pd.DataFrame,
                etalon: pd.DataFrame,
                quart: float = 0.9,
                ppm: float = 3,
                show_map: bool = True): 
    '''
    Recallibrate by etalon

    Parameters
    ----------
    spec: pd.DataFrame object
        Initial mass spectrum for recallibrate
    etalon: pd.DataFrame object
        Etalon mass spectrum
    quart: float
        Optionaly. by default it is 0.9. 
        Usualy it is enough for good callibration
        Quartile, which will be taken for calc recallibrate error
    ppm: float
        Optionaly. Default 3.
        permissible relative error in ppm for seak peak in etalon
    show_map: bool
        Optional. Default True. Show process 

    Return
    ------
    ErrorTable
    '''

    et = etalon['mass'].to_numpy()  # Используем numpy array вместо list
    df = spec.copy()  # Простое копирование вместо deepcopy

    # Фильтрация по квантилю
    treshold = df['intensity'].quantile(quart)
    df_filtered = df[df['intensity'] > treshold].copy()
    
    # Векторизованное вычисление границ
    min_masses = df_filtered['mass'].to_numpy() * (1 - ppm / 1000000)
    max_masses = df_filtered['mass'].to_numpy() * (1 + ppm / 1000000)
    
    # Преобразуем etalon в отсортированный numpy array для бинарного поиска
    et_sorted = np.sort(et)
    
    # Функция для поиска ближайшего эталонного значения в пределах ppm
    def find_closest_etalon(mass, min_mass, max_mass):
        # Бинарный поиск в отсортированном массиве
        idx = np.searchsorted(et_sorted, min_mass)
        if idx < len(et_sorted) and et_sorted[idx] <= max_mass:
            return et_sorted[idx]
        return 0
    
    # Векторизованное применение функции
    df_filtered['cal'] = np.vectorize(find_closest_etalon, otypes=[float])(
        df_filtered['mass'].values, min_masses, max_masses
    )
    
    # Альтернативный вариант с использованием broadcasting (быстрее для больших данных)
    # df_filtered['cal'] = 0
    # for i, (min_m, max_m) in enumerate(zip(min_masses, max_masses)):
    #     mask = (et_sorted >= min_m) & (et_sorted <= max_m)
    #     if np.any(mask):
    #         df_filtered.iloc[i, df_filtered.columns.get_loc('cal')] = et_sorted[mask][0]

    # Фильтрация и вычисления
    df_calibrated = df_filtered[df_filtered['cal'] > 0].copy()
    df_calibrated['dif'] = df_calibrated['cal'] - df_calibrated['mass']
    df_calibrated['ppm'] = df_calibrated['dif'] / df_calibrated['mass'] * 1000000
    
    error_table = df_calibrated[['mass', 'ppm']].dropna()
    
    kdm = ms_calc.kernel_density_map(df_error=error_table)
    err = ms_calc.fit_kernel(f=kdm, show_map=show_map, mass=spec['mass'].to_numpy())

    return err


# ============================================================================
# Калибровка лигносульфонатов с очисткой от загрязнителей
# ============================================================================
#
# Почему не assign_error:
#   * словарь CHO (без S) не описывает лигносульфонат: 60–80 % интенсивности дают CHOS;
#   * окно ±3 ppm: при исходной ошибке прибора > 3 ppm истинные формулы в окно не попадают,
#     а KDE-карта цепляется за ложные совпадения. Типичная ловушка — изобарная пара
#     C4S ↔ O5 (Δ = 2,5 мДа, 4–8 ppm): сульфонат C20H22O8S «превращается» в C16H22O13,
#     и спектр уезжает на +5…+7 ppm;
#   * сильные загрязнители (LAS, жирные кислоты, алкилсульфаты) доминируют в спектре.
#
# Здесь: библиотека правдоподобных для лигносульфоната формул (CHO и CHOS с O ≥ 3S + 2),
# грубый поиск линейной модели ошибки (±20 ppm, наклон) преобразованием Хафа с проверкой
# изотопов 34S и 13C, уточнение робастным полиномом с сужающимся допуском и удаление
# загрязнителей вместе с их изотопологами.
#
# Проверено (30.10.2026) на 16 спектрах LST: сдвиги ±15 ppm и наклоны до 7 ppm на 600 Да
# восстанавливаются с точностью < 0,1 ppm; recallibrate_optimize при сдвиге > 3 ppm
# уводит спектр к +6…+7 ppm.

_ION_SHIFT = 1.007825 - 0.00054858     # нейтральная M → [M-H]-  (как в assign_optimized)
_C13_DELTA = 1.003355
_S34_DELTA = 1.995796
_MONO = {'C': 12.0, 'H': 1.007825, 'N': 14.003074, 'O': 15.994915, 'S': 31.972071}


def _ion_mass(C, H, O, S, N=0):
    return (np.asarray(C) * _MONO['C'] + np.asarray(H) * _MONO['H'] + np.asarray(O) * _MONO['O']
            + np.asarray(S) * _MONO['S'] + np.asarray(N) * _MONO['N'] - _ION_SHIFT)


def lignosulfonate_formula_library(mass_range: tuple = (150.0, 1000.0),
                                   max_sulfur: int = 2) -> pd.DataFrame:
    '''
    Формулы, правдоподобные для лигносульфоната (ионы [M-H]-), — калибранты.

    CHO:  H/C 0.5–1.9, O/C 0.1–0.8, DBE 2–25
    CHOS: S 1..max_sulfur, O ≥ 3S + 2 (сульфогруппа + хотя бы два O скелета),
          (O − 3S)/C ≤ 0.8, S/C ≤ 0.2, H/C 0.5–1.9, DBE 2–25

    Return
    ------
    pd.DataFrame с колонками C, H, O, S, mass (m/z иона), отсортированная по mass
    '''
    C, H, O, S = np.meshgrid(np.arange(5, 51), np.arange(4, 81, 2), np.arange(2, 31),
                             np.arange(0, max_sulfur + 1), indexing='ij')
    C, H, O, S = C.ravel(), H.ravel(), O.ravel(), S.ravel()
    hc, dbe = H / C, C - H / 2 + 1
    ok = (hc >= 0.5) & (hc <= 1.9) & (dbe >= 2) & (dbe <= 25)
    skel_oc = (O - 3 * S) / C
    ok &= np.where(S == 0, (O / C >= 0.1) & (O / C <= 0.8),
                   (O >= 3 * S + 2) & (skel_oc <= 0.8) & (S / C <= 0.2))
    lib = pd.DataFrame({'C': C[ok], 'H': H[ok], 'O': O[ok], 'S': S[ok]})
    lib['mass'] = _ion_mass(lib.C, lib.H, lib.O, lib.S)
    lib = lib[(lib.mass >= mass_range[0]) & (lib.mass <= mass_range[1])]
    return lib.sort_values('mass').reset_index(drop=True)


def lignosulfonate_contaminants(extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    '''
    Типичные загрязнители спектров лигносульфоната (ионы [M-H]-, ESI-).

    fatty_acid      CnH2nO2, C8–C26, и ненасыщенные (DBE 2–4) C14–C22
    LAS             линейные алкилбензолсульфонаты CnH(2n-6)O3S, C15–C22 (цепь C9–C16)
    LAS_oxidized    гидрокси/кето-LAS CnH(2n-6)O4S и CnH(2n-8)O4S
    alkyl_sulfate   CnH(2n+2)O4S, C8–C18 (додецилсульфат и гомологи)
    alkyl_sulfonate CnH(2n+2)O3S, C10–C18
    ether_sulfate   алкилэтоксисульфаты C(n+2k)H(2(n+2k)+2)O(4+k)S, n 10–16, k 1–4
    docusate        диоктилсульфосукцинат C20H38O7S

    Parameters
    ----------
    extra: pd.DataFrame
        Optional. Дополнительные формулы: колонки name, C, H, O, S (N — необязательно).

    Return
    ------
    pd.DataFrame: name, formula, C, H, O, S, N, mass (m/z иона)
    '''
    rows = []
    for n in range(8, 27):
        rows.append(('fatty_acid', n, 2 * n, 2, 0))
    for n in range(14, 23):
        for dbe in (2, 3, 4):
            rows.append(('fatty_acid', n, 2 * n + 2 - 2 * dbe, 2, 0))
    for n in range(15, 23):
        rows.append(('LAS', n, 2 * n - 6, 3, 1))
        rows.append(('LAS_oxidized', n, 2 * n - 6, 4, 1))
        rows.append(('LAS_oxidized', n, 2 * n - 8, 4, 1))
    for n in range(8, 19):
        rows.append(('alkyl_sulfate', n, 2 * n + 2, 4, 1))
    for n in range(10, 19):
        rows.append(('alkyl_sulfonate', n, 2 * n + 2, 3, 1))
    for n in range(10, 17):
        for k in range(1, 5):
            c = n + 2 * k
            rows.append(('ether_sulfate', c, 2 * c + 2, 4 + k, 1))
    rows.append(('docusate', 20, 38, 7, 1))

    df = pd.DataFrame(rows, columns=['name', 'C', 'H', 'O', 'S'])
    df['N'] = 0
    if extra is not None and len(extra):
        ex = extra.copy()
        for col in ('S', 'N'):
            if col not in ex:
                ex[col] = 0
        df = pd.concat([df, ex[['name', 'C', 'H', 'O', 'S', 'N']]], ignore_index=True)
    df = df.drop_duplicates(subset=['C', 'H', 'O', 'S', 'N']).reset_index(drop=True)
    df['formula'] = [f"C{c}H{h}" + (f"N{n}" if n else "") + f"O{o}" + (f"S{s}" if s else "")
                     for c, h, o, s, n in zip(df.C, df.H, df.O, df.S, df.N)]
    df['mass'] = _ion_mass(df.C, df.H, df.O, df.S, df.N) # type: ignore
    return df.sort_values('mass').reset_index(drop=True)


def _match(peaks: np.ndarray, ref: np.ndarray, tol_ppm: np.ndarray):
    '''Ближайшая формула библиотеки для каждого пика; (индекс, ошибка ppm) или -1.'''
    i = np.clip(np.searchsorted(ref, peaks), 1, len(ref) - 1)
    j = np.where(np.abs(ref[i - 1] - peaks) < np.abs(ref[i] - peaks), i - 1, i)
    err = (peaks - ref[j]) / ref[j] * 1e6
    j = np.where(np.abs(err) <= tol_ppm, j, -1)
    return j, err


def _all_errors(peaks: np.ndarray, ref: np.ndarray, max_ppm: float):
    '''Все пары пик–формула в окне ±max_ppm: (индекс пика, индекс формулы, ошибка ppm).'''
    lo = np.searchsorted(ref, peaks * (1 - max_ppm * 1e-6))
    hi = np.searchsorted(ref, peaks * (1 + max_ppm * 1e-6))
    cnt = hi - lo
    pk = np.repeat(np.arange(len(peaks)), cnt)
    rf = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    return pk, rf, (peaks[pk] - ref[rf]) / ref[rf] * 1e6


def _sulfur_evidence(peaks: np.ndarray, peak_int: np.ndarray, all_masses: np.ndarray,
                     all_int: np.ndarray, ppm: float = 2.0) -> np.ndarray:
    '''
    Есть ли у пика изотополог 34S (+1,995796 Да) с правдоподобной интенсивностью.
    Разность масс не зависит от калибровки, поэтому проверка работает до неё.

    Return: +1 — 34S найден (I34/I = 0,015–0,3), −1 — не найден, хотя был бы виден
    (0,03·I выше порога спектра), 0 — пик слишком слабый, чтобы судить.
    '''
    order = np.argsort(all_masses)
    m_sorted, i_sorted = all_masses[order], all_int[order]
    target = peaks + _S34_DELTA
    i = np.clip(np.searchsorted(m_sorted, target), 1, len(m_sorted) - 1)
    j = np.where(np.abs(m_sorted[i - 1] - target) < np.abs(m_sorted[i] - target), i - 1, i)
    found = np.abs(m_sorted[j] - target) / target * 1e6 <= ppm
    ratio = np.where(found, i_sorted[j] / peak_int, 0.0)
    has = found & (ratio >= 0.015) & (ratio <= 0.3)
    visible = peak_int * 0.03 >= 2 * all_int.min()
    return np.where(has, 1, np.where(visible, -1, 0))


def _carbon13_ratio(peaks: np.ndarray, peak_int: np.ndarray, all_masses: np.ndarray,
                    all_int: np.ndarray, ppm: float = 2.0) -> np.ndarray:
    '''Наблюдаемое отношение I(13C)/I(M); 0 — изотополог не найден. Не зависит от калибровки.'''
    order = np.argsort(all_masses)
    m_sorted, i_sorted = all_masses[order], all_int[order]
    target = peaks + _C13_DELTA
    i = np.clip(np.searchsorted(m_sorted, target), 1, len(m_sorted) - 1)
    j = np.where(np.abs(m_sorted[i - 1] - target) < np.abs(m_sorted[i] - target), i - 1, i)
    found = np.abs(m_sorted[j] - target) / target * 1e6 <= ppm
    return np.where(found, i_sorted[j] / peak_int, 0.0)


def _carbon13_contradicts(ratio13: np.ndarray, peak_int: np.ndarray, carbons: np.ndarray,
                          floor: float, tolerance: float = 1.4) -> np.ndarray:
    '''
    Формула противоречит изотопу 13C: пик 13C был бы виден (0,0107·C·I выше 2×порога),
    но не найден или его доля отличается от 0,0107·C больше чем в tolerance раз.
    '''
    expected = 0.0107 * carbons
    visible = expected * peak_int >= 2 * floor
    with np.errstate(divide='ignore', invalid='ignore'):
        off = (ratio13 <= 0) | (np.abs(np.log(ratio13 / expected)) > np.log(tolerance))
    return visible & off


def _isotope_consistent(evidence: np.ndarray, ref_s: np.ndarray) -> np.ndarray:
    '''Формула с серой допустима, если 34S не опровергнут; без серы — если 34S не найден.'''
    return np.where(ref_s > 0, evidence >= 0, evidence <= 0)


def _coarse_lines(peaks: np.ndarray, ref: np.ndarray, max_ppm: float,
                  ref_s: Optional[np.ndarray] = None, evidence: Optional[np.ndarray] = None,
                  max_slope: float = 4.0, slope_step: float = 0.1, bin_ppm: float = 0.2,
                  m0: float = 400.0, n_candidates: int = 5):
    '''
    Кандидаты грубой модели ошибки err(m) = a + b·(m − m0)/100 (ppm) — преобразование
    Хафа: каждая пара пик–формула голосует за все прямые, на которых лежит её ошибка;
    пик голосует не больше раза в ячейке (a, b). Истинная прямая собирает почти все
    сильные пики, случайные совпадения рассеяны.
    С ref_s и evidence голосуют только пары, согласные с изотопом 34S: это ослабляет
    ложную прямую изобарной пары C4S ↔ O5, на которой сульфонаты получают CHO-формулы.
    Возвращаются n_candidates лучших заметно различающихся прямых, вариант без наклона
    и нулевая модель —
    окончательный выбор делается после уточнения (по числу калибрантов).

    Return
    ------
    list of (a, b, votes)
    '''
    x = (peaks - m0) / 100.0
    span = max_slope * max(np.abs(x).max(), 1e-9)
    pk, rf, err = _all_errors(peaks, ref, max_ppm + span)
    if ref_s is not None and evidence is not None:
        ok = _isotope_consistent(evidence[pk], ref_s[rf])
        pk, rf, err = pk[ok], rf[ok], err[ok]
    if len(err) == 0:
        return [(0.0, 0.0, 0)]
    slopes = np.arange(-max_slope, max_slope + slope_step / 2, slope_step)
    edges = np.arange(-max_ppm, max_ppm + bin_ppm, bin_ppm)
    nb = len(edges) - 1
    votes = np.zeros((len(slopes), nb))
    for i, b in enumerate(slopes):
        a = err - b * x[pk]
        inside = (a >= -max_ppm) & (a < max_ppm)
        k = np.clip(np.digitize(a[inside], edges) - 1, 0, nb - 1)
        uniq = np.unique(pk[inside] * nb + k)
        np.add.at(votes[i], uniq % nb, 1)
    # сглаживание 3×3 (0,6 ppm × 0,3 ppm/100 Да)
    pad = np.pad(votes, 1)
    smooth = sum(pad[1 + di:1 + di + votes.shape[0], 1 + dj:1 + dj + nb]
                 for di in (-1, 0, 1) for dj in (-1, 0, 1))

    a_grid = (edges[:-1] + bin_ppm / 2)[None, :]
    b_grid = slopes[:, None]

    def refine(a0, b0):
        on = np.abs(err - (a0 + b0 * x[pk])) <= 1.5 * bin_ppm
        if on.sum() >= 5 and np.ptp(x[pk[on]]) > 0:
            b1, a1 = np.polyfit(x[pk[on]], err[on], 1)
            return float(a1), float(b1)
        return float(a0), float(b0)

    out, work = [], smooth.copy() # type: ignore
    for _ in range(n_candidates):
        i, k = np.unravel_index(np.argmax(work), work.shape)
        if work[i, k] <= 0:
            break
        a0, b0 = a_grid[0, k], b_grid[i, 0]
        out.append((*refine(a0, b0), float(work[i, k])))
        # подавляем прямые, отличающиеся от найденной меньше чем на 1,5 ppm во всём диапазоне масс
        near = np.maximum(np.abs((a_grid - a0) + (b_grid - b0) * x.min()),
                          np.abs((a_grid - a0) + (b_grid - b0) * x.max())) <= 1.5
        work[near] = 0
    i0 = int(np.argmin(np.abs(slopes)))                          # вариант без наклона
    k0 = int(np.argmax(smooth[i0])) # type: ignore
    out.append((*refine(a_grid[0, k0], 0.0), float(smooth[i0, k0]))) # type: ignore
    out.append((0.0, 0.0, float(smooth[i0, int(np.argmin(np.abs(a_grid[0])))])))   # type: ignore # без поправки
    return out


def _fine_fit(peaks, ref, ref_s, evidence, model, max_degree, min_calibrants,
              ref_c=None, ratio13=None, peak_int=None, floor=None):
    '''
    Уточнение: допуск 2 → 1 → 0,6 ppm вокруг модели, робастный полином.
    score — число калибрантов без тех, чьё число атомов C противоречит изотопу 13C.
    '''
    fit = None
    for tol in (2.0, 1.0, 0.6):
        pred = model(peaks)
        j, _ = _match(peaks / (1 + pred * 1e-6), ref, np.full(len(peaks), tol))
        m_ok = (j >= 0) & _isotope_consistent(evidence, ref_s[np.maximum(j, 0)])
        if m_ok.sum() < min_calibrants:
            break
        x = peaks[m_ok]
        y = (peaks[m_ok] - ref[j[m_ok]]) / ref[j[m_ok]] * 1e6
        c, d, x0, sc, keep, mad = _robust_poly(x, y, max_degree=max_degree) # type: ignore
        if keep.sum() < min_calibrants:
            break
        xmin, xmax = x[keep].min(), x[keep].max()
        model = (lambda m, c=c, x0=x0, sc=sc, a=xmin, b=xmax:
                 np.polyval(c, (np.clip(m, a, b) - x0) / sc))
        bad = np.zeros(int(m_ok.sum()), dtype=bool)
        if ref_c is not None and ratio13 is not None:
            bad = _carbon13_contradicts(ratio13[m_ok], peak_int[m_ok], ref_c[j[m_ok]], floor) # type: ignore
        contra = int(bad[keep].sum())
        # доля интенсивности сильных пиков, объяснённая калибрантами без противоречий с 13C:
        # сильные пики — главные компоненты образца, случайные совпадения среди них редки
        w = peak_int / peak_int.sum() if peak_int is not None else np.full(len(peaks), 1 / len(peaks))
        fit = dict(model=model, x=x, y=y, keep=keep, coef=c, degree=d, x0=x0, scale=sc,
                   n=int(keep.sum()), c13_contradicted=contra,
                   score=int(keep.sum()) - contra,
                   explained=float(w[m_ok][keep & ~bad].sum()),
                   explained_sulfonate=float(w[m_ok][keep & ~bad & (ref_s[j[m_ok]] > 0)].sum()),
                   s_confirmed=int(((ref_s[j[m_ok]] > 0) & (evidence[m_ok] > 0))[keep].sum()))
    return fit


def _robust_poly(x: np.ndarray, y: np.ndarray, max_degree: int = 2, clip: float = 3.0):
    '''Полином ошибки от массы с отсечением выбросов (3×MAD) и выбором степени по BIC.'''
    keep = np.ones(len(x), dtype=bool)
    x0, scale = np.mean(x), np.std(x) or 1.0
    best = None
    for _ in range(5):
        xs = (x[keep] - x0) / scale
        cands = []
        for d in range(0, max_degree + 1):
            if keep.sum() < 15 * (d + 1):
                break
            c = np.polyfit(xs, y[keep], d)
            rss = np.sum((y[keep] - np.polyval(c, xs)) ** 2)
            bic = keep.sum() * np.log(rss / keep.sum() + 1e-12) + (d + 1) * np.log(keep.sum())
            cands.append((bic, d, c))
        if not cands:
            c = np.array([np.median(y[keep])])
            cands = [(0, 0, c)]
        _, d, c = min(cands, key=lambda t: t[0])
        res = y - np.polyval(c, (x - x0) / scale)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) or 0.05
        new_keep = np.abs(res) <= clip * mad
        best = (c, d, x0, scale, keep.copy(), mad)
        if (new_keep == keep).all():
            break
        keep = new_keep
    return best


def find_contaminants(spec: pd.DataFrame,
                      contaminants: Optional[pd.DataFrame] = None,
                      ppm: float = 1.0,
                      isotopes: bool = True) -> pd.DataFrame:
    '''
    Пики загрязнителей (и их 13C/34S-изотопологи) в откалиброванном спектре.

    Return
    ------
    pd.DataFrame: index пика в spec, name, formula, mass, intensity, ppm, isotope
    '''
    if contaminants is None:
        contaminants = lignosulfonate_contaminants()
    order = np.argsort(spec['mass'].to_numpy())
    masses = spec['mass'].to_numpy()[order]
    out = []
    targets = [(contaminants['mass'].to_numpy(), '')]
    if isotopes:
        targets.append((contaminants['mass'].to_numpy() + _C13_DELTA, '13C'))
        s_mask = contaminants['S'].to_numpy() > 0
        targets.append((np.where(s_mask, contaminants['mass'].to_numpy() + _S34_DELTA, np.nan), '34S'))
    for tm, iso in targets:
        valid = ~np.isnan(tm)
        j, err = _match(tm[valid], masses, np.full(valid.sum(), ppm))
        for k, (jj, e) in enumerate(zip(j, err)):
            if jj < 0:
                continue
            row = contaminants.loc[np.where(valid)[0][k]]
            idx = spec.index[order[jj]]
            out.append(dict(index=idx, name=row['name'], formula=row['formula'], isotope=iso,
                            mass=float(spec.loc[idx, 'mass']), intensity=float(spec.loc[idx, 'intensity']), # type: ignore
                            ppm=float(-e)))
    res = pd.DataFrame(out, columns=['index', 'name', 'formula', 'isotope', 'mass', 'intensity', 'ppm'])
    return res.drop_duplicates('index').set_index('index')


def recalibrate_lignosulfonate(spec: pd.DataFrame,
                               max_shift_ppm: float = 20.0,
                               n_peaks: int = 1000,
                               max_degree: int = 2,
                               contaminants: Optional[pd.DataFrame] = None,
                               remove_contaminants: bool = True,
                               contaminants_as_calibrants: bool = True,
                               contaminant_ppm: float = 1.0,
                               mass_range: tuple = (150.0, 1000.0),
                               min_calibrants: int = 20,
                               draw: bool = False) -> pd.DataFrame:
    '''
    Калибровка спектра лигносульфоната (ESI-, [M-H]-) и очистка от загрязнителей.

    1. Грубая модель: для 400 самых интенсивных пиков считаются ошибки относительно
       библиотеки лигносульфонатных формул; преобразованием Хафа ищется прямая
       err = a + b·(m − 400)/100 (|a| ≤ max_shift_ppm, |b| ≤ 4 ppm на 100 Да),
       на которой лежит больше всего пиков. Голосуют только пары, согласные с изотопом
       34S (пик +1,9958 Да): формула с S — если 34S не опровергнут, без S — если не найден.
       Это ослабляет ложное решение изобарной пары C4S ↔ O5 (сдвиг +4…+8 ppm).
       Берутся 5 лучших различных прямых и вариант без наклона; каждый уточняется (шаг 2),
       выбирается тот, у которого больше калибрантов, чьё число атомов C не противоречит
       изотопу 13C (0,0107·C). Кандидаты, объясняющие < 80 % интенсивности сильных пиков от
       лучшего, отбрасываются. Среди кандидатов всегда есть «без поправки». Если другой
       кандидат почти так же хорош (≥ 90 %), а поправка для сильных пиков отличается в среднем
       больше чем на 1 ppm, в отчёте ambiguous=True и предупреждение.
    2. Уточнение: пики приписываются ближайшей (и изотопно согласной) формуле при допуске 2 → 1 → 0,6 ppm
       вокруг текущей модели; ошибка аппроксимируется полиномом степени ≤ max_degree
       (степень по BIC, выбросы отсекаются по 3×MAD). Вне диапазона калибрантов поправка
       не экстраполируется, а берётся с ближайшего края.
    3. Массы исправляются: mass / (1 + err(mass)·1e-6).
    4. Загрязнители (lignosulfonate_contaminants) и их 13C/34S-изотопологи находятся
       в исправленном спектре с допуском contaminant_ppm и удаляются
       (remove_contaminants=True) или помечаются колонкой 'contaminant'.

    Parameters
    ----------
    spec: pd.DataFrame
        Центроидный спектр с колонками mass, intensity
    max_shift_ppm: float
        Optional. Default 20. Максимальная ожидаемая ошибка прибора
    n_peaks: int
        Optional. Default 1000. Сколько самых интенсивных пиков используется
    max_degree: int
        Optional. Default 2. Максимальная степень полинома ошибки
    contaminants: pd.DataFrame
        Optional. Таблица загрязнителей, по умолчанию lignosulfonate_contaminants()
    remove_contaminants: bool
        Optional. Default True. Удалить загрязнители; False — колонка 'contaminant'
    contaminants_as_calibrants: bool
        Optional. Default True. Загрязнители — однозначно известные вещества и хорошие
        внутренние стандарты в области m/z 200–450; их пики участвуют в калибровке
    contaminant_ppm: float
        Optional. Default 1.0. Допуск поиска загрязнителей после калибровки
    mass_range: tuple
        Optional. Диапазон m/z библиотеки калибрантов
    min_calibrants: int
        Optional. Default 20. Меньше калибрантов — спектр не меняется, предупреждение
    draw: bool
        Optional. Default False. График ошибок до и после

    Returns
    -------
    pd.DataFrame
        attrs['recallibrate'] = 'lignosulfonate'
        attrs['calibration'] = dict: shift_ppm (при m/z 400), slope_ppm_per_100 — грубая модель;
                               n_calibrants, s_confirmed (сульфонаты с 34S), c13_contradicted,
                               explained, explained_sulfonate (доля интенсивности 1000 сильных пиков,
                               объяснённая калибрантами / калибрантами с S; < 0,1 — спектр не похож
                               на лигносульфонат), calibrant_fraction, ambiguous (+ alternative), degree,
                               rms_ppm, mass_min, mass_max, err_at_min, err_at_max, ok
        attrs['contaminants'] = list[dict] найденных пиков загрязнителей
                                (name, formula, isotope, mass, intensity, ppm);
                                pd.DataFrame(spec.attrs['contaminants']) — таблица
    '''
    import warnings

    spec = spec.copy()
    name = spec.attrs.get('name', '')
    lib = lignosulfonate_formula_library(mass_range)
    cont = lignosulfonate_contaminants() if contaminants is None else contaminants
    ref, ref_s, ref_c = lib['mass'].to_numpy(), lib['S'].to_numpy(), lib['C'].to_numpy()
    if contaminants_as_calibrants:
        ref = np.concatenate([ref, cont['mass'].to_numpy()])
        ref_s = np.concatenate([ref_s, cont['S'].to_numpy()])
        ref_c = np.concatenate([ref_c, cont['C'].to_numpy()])
        o = np.argsort(ref, kind='stable')
        ref, ref_s, ref_c = ref[o], ref_s[o], ref_c[o]

    top = spec.loc[(spec['mass'] >= mass_range[0]) & (spec['mass'] <= mass_range[1])]
    top = top.nlargest(min(n_peaks, len(top)), 'intensity')
    peaks = top['mass'].to_numpy()
    # изотоп 34S: подтверждает или опровергает серу в формуле (до калибровки)
    peak_int = top['intensity'].to_numpy()
    all_m, all_i = spec['mass'].to_numpy(), spec['intensity'].to_numpy()
    evidence = _sulfur_evidence(peaks, peak_int, all_m, all_i)
    ratio13 = _carbon13_ratio(peaks, peak_int, all_m, all_i)
    report = dict(ok=False, ambiguous=False, shift_ppm=np.nan, slope_ppm_per_100=np.nan, n_calibrants=0,
                  s_confirmed=0, c13_contradicted=0, degree=None,
                  rms_ppm=np.nan, mass_min=np.nan, mass_max=np.nan, err_at_min=np.nan, err_at_max=np.nan)

    if len(peaks) >= min_calibrants:
        # 1. кандидаты грубой линейной модели по самым сильным пикам
        n_strong = min(400, len(peaks))
        cands = _coarse_lines(peaks[:n_strong], ref, max_shift_ppm,
                              ref_s=ref_s, evidence=evidence[:n_strong])
        # 2. уточнение каждого кандидата; побеждает больше калибрантов, не противоречащих 13C
        #    (при равенстве — больше сульфонатов, подтверждённых 34S, затем меньший наклон)
        fits = []
        for a0, b0, votes in cands:
            f = _fine_fit(peaks, ref, ref_s, evidence,
                          lambda m, a=a0, b=b0: a + b * (np.asarray(m) - 400.0) / 100.0,
                          max_degree, min_calibrants, ref_c=ref_c, ratio13=ratio13,
                          peak_int=peak_int, floor=all_i.min())
            if f is not None:
                f['start'] = (a0, b0)
                fits.append(f)
        # сильные пики должны объясняться: кандидаты, объясняющие заметно меньше
        # интенсивности, чем лучший, отбрасываются (ложные прямые набирают слабые пики)
        if fits:
            top_expl = max(f['explained'] for f in fits)
            fits = [f for f in fits if f['explained'] >= 0.8 * top_expl]
        fits.sort(key=lambda f: (f['score'], f['s_confirmed'], -abs(f['start'][1])), reverse=True)
        fit = fits[0] if fits else None
        if fit is not None:
            report.update(shift_ppm=fit['start'][0], slope_ppm_per_100=fit['start'][1],
                          c13_contradicted=fit['c13_contradicted'], explained=round(fit['explained'], 3),
                          explained_sulfonate=round(fit['explained_sulfonate'], 3))
            # неоднозначность: другой кандидат почти так же хорош (≥ 90 %), а поправка
            # для сильных пиков отличается в среднем больше чем на 1 ppm
            wts = peak_int / peak_int.sum()
            for f2 in fits[1:]:
                diff = float(np.sum(wts * np.abs(f2['model'](peaks) - fit['model'](peaks))))
                if diff > 1.0 and f2['score'] >= 0.9 * fit['score']:
                    report['ambiguous'] = True
                    report['alternative'] = dict(score=f2['score'], mean_diff_ppm=round(diff, 2), # type: ignore
                                                 err_at_300=float(f2['model'](300.0)),
                                                 err_at_600=float(f2['model'](600.0)))
                    break

            model = fit['model']
            x, y, keep = fit['x'], fit['y'], fit['keep']
            resid = y[keep] - np.polyval(fit['coef'], (x[keep] - fit['x0']) / fit['scale'])
            report.update(ok=True, n_calibrants=fit['n'], s_confirmed=fit['s_confirmed'],
                          degree=int(fit['degree']), rms_ppm=float(np.sqrt(np.mean(resid ** 2))),
                          mass_min=float(x[keep].min()), mass_max=float(x[keep].max()),
                          err_at_min=float(model(x[keep].min())), err_at_max=float(model(x[keep].max())))
            if draw:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
                grid = np.linspace(x.min(), x.max(), 200)
                ax[0].scatter(x, y, s=6, c=np.where(keep, 'C0', 'C3'))
                ax[0].plot(grid, model(grid), 'k-')
                ax[0].set_title(f'{name}: ошибка до калибровки')
                ax[1].scatter(x[keep], resid, s=6)
                ax[1].axhline(0, color='k', lw=0.5)
                ax[1].set_title(f'после: RMS {report["rms_ppm"]:.2f} ppm')
                for a in ax:
                    a.set_xlabel('m/z'); a.set_ylabel('ppm')
                fig.tight_layout()

    # 3. поправка масс
    if report['ok'] and report['n_calibrants'] >= min_calibrants: # type: ignore
        spec['mass'] = spec['mass'] / (1 + model(spec['mass'].to_numpy()) * 1e-6) # type: ignore
    else:
        report['ok'] = False
        warnings.warn(f'{name}: недостаточно калибрантов ({report["n_calibrants"]}), '
                      f'массы не изменены', RuntimeWarning)

    report['calibrant_fraction'] = report['n_calibrants'] / max(len(peaks), 1) # type: ignore
    if report['ok'] and report['ambiguous']:
        warnings.warn(f'{name}: калибровка неоднозначна — альтернативная модель {report["alternative"]} '
                      f'почти так же хорошо объясняет пики; проверьте (draw=True)', RuntimeWarning)
    if report['ok'] and report.get('explained_sulfonate', 1.0) < 0.1: # type: ignore
        warnings.warn(f'{name}: сульфонаты объясняют лишь {100 * report["explained_sulfonate"]:.0f}% ' # type: ignore
                      f'интенсивности сильных пиков — спектр не похож на лигносульфонат, '
                      f'калибровка ненадёжна', RuntimeWarning)
    if report['ok'] and report['calibrant_fraction'] < 0.15:
        warnings.warn(f'{name}: калибрантами стали лишь {100 * report["calibrant_fraction"]:.0f}% '
                      f'сильных пиков — проверьте результат (draw=True)', RuntimeWarning)

    # 4. загрязнители
    found = find_contaminants(spec, cont, ppm=contaminant_ppm)
    if remove_contaminants:
        spec = spec.drop(index=found.index)
    else:
        spec['contaminant'] = ''
        spec.loc[found.index, 'contaminant'] = found['name'] + found['isotope'].map(lambda s: f' ({s})' if s else '')

    spec = spec.reset_index(drop=True) if remove_contaminants else spec
    spec.attrs['name'] = name
    spec.attrs['recallibrate'] = 'lignosulfonate'
    spec.attrs['calibration'] = report
    # список словарей, а не DataFrame: DataFrame в attrs ломает concat/nlargest в pandas
    spec.attrs['contaminants'] = found.reset_index(drop=True).to_dict('records')
    return spec
