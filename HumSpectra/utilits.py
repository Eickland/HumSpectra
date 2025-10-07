import os
import pandas as pd
from pandas import DataFrame
import re
from transliterate import translit


def extract_and_combine_digits_re(text: str) -> int:
    """Извлекает все цифры из строки и объединяет их в одно целое число.

    :param text: Строка для поиска цифр.

    :return Целое число, образованное из цифр в строке. Возвращает 0, если цифр нет.
    """
    digits = re.findall(r'\d', text)  # Находим все цифры в строке
    if digits:
        combined_number_str = ''.join(digits)  # Объединяем цифры в строку
        try:
            return int(combined_number_str)  # Преобразуем строку в целое число
        except ValueError:
            return 0  # Возвращаем 0, если строка не может быть преобразована в число
    else:
        return 0  # Возвращаем 0, если цифры не найдены


def extract_name_from_path(file_path: str) -> str:
    r"""Извлекает имя файла (без расширения) из пути.

    Поддерживает разные разделители каталогов (/, \, //).

    :param file_path: Полный путь к файлу.

    :return: file_name: Имя файла без расширения.  Возвращает пустую строку, если путь недопустимый.
    """
    try:
        # 1. Нормализация пути: Замена двойных слешей на одинарные,
        #    а также приведение к абсолютному пути (это нужно не всегда,
        #    но может помочь избежать проблем).
        normalized_path = os.path.abspath(os.path.normpath(file_path))

        # 2. Извлечение имени файла (вместе с расширением).
        file_name_with_extension = os.path.basename(normalized_path)

        # 3. Разделение имени файла и расширения.
        file_name, file_extension = os.path.splitext(file_name_with_extension)

        # 4. Земена русских символов на английские транслитом.
        file_name = translit(file_name, 'ru', reversed=True)

        return file_name
    
    except Exception as e:
        print(f"Ошибка при обработке пути {file_path}: {e}")
        return ""


def extract_class_from_name(file_name: str) -> str:
    """Извлекает класс образца из имени.

       :param file_name: Имя образца.

       :return: sample_class: Класс образца.
       Функция определяет класс по образца по имени. Определяемые классы: Coal(угольные), Soil(почвенные), Peat(торфяные), L(лигногуматы и лигносульфонаты), 
       ADOM(надшламовые воды) 
       """
    sample_class = "SampleClassError"
    str_name = file_name.replace(" ", "-")
    if "ADOM" in str_name:
        sample_class = "ADOM"
    else:
        str_class = str_name.split(sep="-")[0]
        symbol_class = str_class[0]
        if "C" == symbol_class:
            sample_class = "Coal"
        elif "L" == symbol_class:
            sample_class = "Lg"
        elif "P" == symbol_class:
            sample_class = "Peat"
        elif "S" == symbol_class:
            sample_class = "Soil"
        elif "K" == symbol_class:
            sample_class = "ADOM"
        elif "B" == symbol_class:
            sample_class = "ADOM"
        else:
            raise ValueError("Имя образца не соответствует ни одному представленному классу")
    return sample_class


def extract_subclass_from_name(file_name: str) -> str:
    """Извлекает подкласс образца из имени.

       :param file_name: Имя образца.

       :return: sample_subclass: Класс образца.
       Функция определяет подкласс по образца по имени. Определяемые подклассы: номера карты накопителя для надшламовых вод, 
       Lg(лигногуматы), Lst(лигносульфонаты), (C,P,S)HA (гуминовые кислоты), (C,P,S)HA (фульвокислоты), (C,P,S)HF (нефракционированные гуминовые вещества)
       """
    sample_subclass = "SampleSubClassError"
    sample_class = extract_class_from_name(file_name=file_name)

    str_name = file_name.replace(" ", "-")

    if "ADOM" == sample_class:
        str_subclass = str_name.split(sep="-")[1]

        if "B" != str_name[0]:
            sample_subclass = "Storage " + str(extract_and_combine_digits_re(str_subclass))
        else:
            sample_subclass = "Baikal"

    elif "L" == str_name[0]:
        if "G" == str_name[1]:
            sample_subclass = "Lg"
        else:
            sample_subclass = "Lst"
    else:
        sample_subclass = str_name.split(sep="-")[0]

    return sample_subclass


def check_sep(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: sep: разделитель в строчном виде (example: ",")
    Функция определяет разделитель столбцов в исходном спектре - запятая или точка с запятой
    """

    try:
        with open(path, 'r') as f:
            first_line = f.readline()

    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")

    if first_line.count(';') > first_line.count(','):
        return ';'
    
    else:
        return ','

def check_file_extension(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: file_extension: строка в котором расширение файла
    Функция определяет расширение файла - csv, txt, excel файл и возвращает расширение.
    """

    extension = path.split(sep=".")[-1] 

    return extension   

def check_file_type(path: str) -> str:
    """
    :param path: путь к файлу в строчном виде
    :return: file_type: строка-кодировка типа файла
    Функция определяет тип файла - csv, txt, excel файл и возвращает кодировку
    """

    ext = check_file_extension(path)

    if ext in ["txt","csv"]:
        file_type = "csv_type"
    
    elif ext == "xlsx":
        
        xlsx = pd.ExcelFile(path)
        sheet_num = len(xlsx.sheet_names)

        if sheet_num == 1:
            file_type = "excel_single"

        elif sheet_num > 1:
            file_type = "excel_many"

        else:
            raise ValueError("Неизвестная ошибка")
    
    else:
        raise ValueError("Тип файла не поддерживается")
    
    return file_type

def attributting_order(data: DataFrame,
                       ignore_name: bool,
                       name: str
                       )-> DataFrame:
    """
    :param data: DataFrame, сырой уф спектр
    :param ignore_name: параметр, при включении которого игнорируются встроенные классы и подклассы
    :param name: имя спектра
    :return: Отформатированный уф спектр
    Функция приписывает имя, класс и подкласс (если не игнорируется) спектру
    """

    data_copy = data.copy()

    if not ignore_name:

        data_copy.attrs['name'] = name
        data_copy.attrs['class'] = extract_class_from_name(name)
        data_copy.attrs['subclass'] = extract_subclass_from_name(name)

    else:

        data_copy.attrs['name'] = name

    return data_copy