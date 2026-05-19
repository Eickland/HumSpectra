from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from HumSpectra.parafac import EEMDataLoader, ComponentVisualizer
from HumSpectra.sample import SampleCollection, split_by_category
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20

CACHE_DATA_PATH = Path('') #установите путь к папке, в которой лежит коллекция .pkl
PARAFAC_PATH = Path('') #установите путь к папке, в которой будут результаты parafac

collection_name = "KNP_samples.pkl" #имя коллекции

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,collection_name)))
loader = EEMDataLoader()
eem_tensor,em_waves,ex_waves,sp_names,_ = loader.load_eem_from_samples(collection, filter_tags=None)
data_info = loader.get_data_info()

analyzer = ComponentVisualizer(n_components=3)
factors = analyzer.fit_parafac(
    eem_tensor,
    excitation_wavelengths=ex_waves,
    emission_wavelengths=em_waves,
    sample_names=sp_names)

analyzer.plot_component_loadings(normalization='percentage',group_type='Subclass')
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","component_loadings.png"))

analyzer.plot_component_profiles()
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","component_profiles.png"))

analyzer.plot_fraction_profiles_grouped()
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","fraction_profiles_grouped.png"))

analyzer.plot_all_components_eem(figsize=(5,14))
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","plot_all_components_eem.png"))

analyzer.export_loadings_to_csv(Path.joinpath(PARAFAC_PATH,"data","parafac.csv"),normalization='max')