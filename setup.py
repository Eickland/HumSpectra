import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HumSpectra",
    version="0.3.35",
    author="Kirill",
    author_email="mnbv21228@mail.ru",
    description="Обработка спектров уф и флуоресценции",
    long_description=long_description, 
    long_description_content_type="text/markdown",
    url="https://github.com/Eickland/Descriptor-calculation-functions-for-fluorescence-and-absorption-spectra-analyses.git",
    packages=setuptools.find_packages(),
    classifiers=[ 
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.11',  
    install_requires=[
        "numpy>=2.0.0",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "transliterate",
        "setuptools",
        "openpyxl",
        "frozendict"
    ],
)
