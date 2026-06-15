from functools import wraps
from frozendict import frozendict
import copy

from typing import Sequence, Any, Dict, Tuple

def _copy(func):
    """
    Decorator for deep copy pd.DataFrame before apllying methods
    
    Parameters
    ----------
    func: method
        function for decoarate
    
    Return
    ------
    function with deepcopyed pd.DataFrame
    """

    @wraps(func)
    def wrapper(dataframe, *args, **kwargs):
        # Создаем глубокую копию DataFrame
        dataframe_copy = copy.deepcopy(dataframe)
        
        # Вызываем оригинальную функцию с копией
        result = func(dataframe_copy, *args, **kwargs)
        
        return result
    
    return wrapper

def _freeze(func):
    """
    freeze dict in func
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

def _process_elems(elems: Any) -> Dict[str, Tuple[int, int]]:
    """
    Преобразует входные данные в рабочий dict
    """

    if elems is None:
        return {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
    elif isinstance(elems, frozendict):
        return dict(elems)
    elif isinstance(elems, dict):
        return elems
    else:
        raise TypeError(f"Expected dict, frozendict or None, got {type(elems)}")
