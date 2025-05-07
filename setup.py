import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="humus-spectra",  # Замените на имя вашего пакета
    version="0.0.1",  # Начальная версия
    author="Kirill",  # Ваше имя
    author_email="mnbv21228@mail.ru",  # Ваш email
    description="Обработка спектров уф и флуоресценции",  # Краткое описание
    long_description=long_description,  # Длинное описание из README.md
    long_description_content_type="text/markdown",  # Тип контента для README
    url="https://github.com/Eickland/Descriptor-calculation-functions-for-fluorescence-and-absorption-spectra-analyses.git",  # URL вашего репозитория
    packages=setuptools.find_packages(),  # Автоматически находит все пакеты в вашем проекте
    classifiers=[  # Классификаторы для PyPI (укажите подходящие)
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.11',  # Минимальная версия Python
    install_requires=[ # Зависимости, которые будут установлены pip
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
)
