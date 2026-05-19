from HumSpectra.sample import Sample, SampleCollection
import HumSpectra.fluorescence as fl
import HumSpectra.utilits as ut
import HumSpectra.ultraviolet as uv

from pathlib import Path
import numpy as np
import sys
import pandas as pd

#project_root = Path.cwd()
#sys.path.insert(0, str(project_root))

CACHE_DATA_PATH = Path('') #Путь к папке, куда выложить базу данных
PARAFAC_PATH = Path('') #Путь к папке, где лежат данные PARAFAC 
STAT_PATH = Path('') #Путь к папке, куда выложить таблицу с значениями дескрипторов
INIT_DATA = Path('') #Путь к папке, где лежат исходные данные,а именно папки Fluorescence, UV/absorption, Additional_data

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)

parafac_data = pd.read_csv(Path.joinpath(PARAFAC_PATH,"parafac.csv"))

fluo_path = Path(Path.joinpath((INIT_DATA),"Fluorescence"))

uv_path = Path.joinpath(Path.joinpath(INIT_DATA), "UV","absorption")    
carbon_data = pd.read_csv(Path.joinpath((INIT_DATA), "Additional_data","C_org.csv"))

geodata = pd.read_csv(Path.joinpath((INIT_DATA), "Additional_data","GeoData.csv"))

fluo_spectra_list = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)

uv_abs_spectra_list = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
uv_reflect_spectra_list = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")

for fl_spectra in fluo_spectra_list:
    
    sample_name = fl_spectra.attrs['name']
    
    if sample_name not in collection.samples:
    
        sample = collection.create_sample_from_data(
        sample_id = sample_name,
        sample_subclass = fl_spectra.attrs['subclass'],
        sample_class = fl_spectra.attrs['class'],
        fluorescence_eem = fl_spectra
            )
    
    else:
        
        sample = collection.samples[sample_name]
        sample.fluorescence_eem = fl_spectra
        
    #sample.add_tag("raw_fluo")
    
    fl_spectra_cutted = fl.cut_spectra(fl_spectra,255,450,255,600)
    sample.cutted_fluo_eem = fl_spectra_cutted
    #sample.add_tag("cutted_fluo")
    
    sample.descriptors['Asm_280'] = fl.calc_asm_280(fl_spectra)
    sample.descriptors['Asm_350'] = fl.calc_asm_350(fl_spectra)
    sample.descriptors['HIX'] = fl.calc_humin_index(fl_spectra)
    sample.descriptors['FIX'] = fl.calc_fluo_index(fl_spectra)
                        
for uv_abs in uv_abs_spectra_list:
    
    sample_name = uv_abs.attrs['name']
    
    if sample_name not in collection.samples:
    
        sample = collection.create_sample_from_data(
        sample_id = sample_name,
        sample_subclass = uv_abs.attrs['subclass'],
        sample_class = uv_abs.attrs['class'],
        uv_vis_absorption = uv_abs
            )
    
    else:
        
        sample = collection.samples[sample_name]
        sample.uv_vis_absorption = uv_abs
        
    #sample.add_tag("raw_uv_abs")
    
    if uv_abs.min().min() < 0:
        uv_abs_recall = uv_abs - uv_abs.min() + 0.00001
    
    elif uv_abs.min().min() == 0:
        
        uv_abs_recall = uv_abs + 0.00001
    
    else:
        uv_abs_recall = uv_abs
    
    sample.uv_vis_absorption = uv_abs_recall    
    #sample.add_tag("uv_abs_baseline")
    
    smooth_uv_abs = uv.smooth_uv_spectrum(uv_abs_recall, window_size=9, threshold_factor=1.5)
    smooth_uv_abs = smooth_uv_abs.iloc[30:]
    
    sample.uv_vis_absorption_smooth = smooth_uv_abs
    #sample.add_tag("smooth_uv_abs")   
    
    sample.descriptors['B1'] = uv.calc_cdom_descriptors(smooth_uv_abs)['B1_prime']
    sample.descriptors['B2'] = uv.calc_cdom_descriptors(smooth_uv_abs)['B2_prime']
    sample.descriptors['Ag_275'] = uv.calc_cdom_descriptors(smooth_uv_abs)['ag_275']
    sample.descriptors['Ag_380'] = uv.calc_cdom_descriptors(smooth_uv_abs)['ag_380']
    sample.descriptors['S_380_443'] = uv.calc_cdom_descriptors(smooth_uv_abs)['S_380_443']
    sample.descriptors['B1'] = uv.calc_cdom_descriptors(smooth_uv_abs)['B1_prime']                  
    sample.descriptors['E2E3'] = uv.calc_e2_e3(smooth_uv_abs)
    sample.descriptors['E3E4'] = uv.calc_e3_e4(smooth_uv_abs)
    sample.descriptors['E4E6'] = uv.calc_e4_e6(smooth_uv_abs)                              
    sample.descriptors['lambda'] = uv.calc_lambda_UV(smooth_uv_abs)
                
for uv_refl in uv_reflect_spectra_list:
    
    sample_name = uv_refl.attrs['name'].replace("-R","")
    
    if sample_name not in collection.samples:
    
        sample = collection.create_sample_from_data(
        sample_id = sample_name,
        sample_subclass = uv_refl.attrs['subclass'],
        sample_class = uv_refl.attrs['class'],
        uv_vis_reflection = uv_refl
            )
    
    else:
        
        sample = collection.samples[sample_name]
        sample.uv_vis_reflection = uv_refl
        
    #sample.add_tag("raw_uv_refl")
    
    smooth_uv_refl = uv.smooth_uv_spectrum(uv_refl, window_size=9, threshold_factor=1.5)
    smooth_uv_refl = smooth_uv_refl.iloc[20:]
    
    sample.uv_vis_reflection_smooth = smooth_uv_refl
    #sample.add_tag("smooth_uv_refl")
    
    sample.descriptors['R280_460'] = uv.calc_r280_460(uv_refl)
    sample.descriptors['R412_547'] = uv.calc_r412_547(uv_refl)
    sample.descriptors['R412_670'] = uv.calc_r412_670(uv_refl)
    sample.descriptors['R460_560'] = uv.calc_r460_560(uv_refl)
    sample.descriptors['R490_555'] = uv.calc_r490_555(uv_refl)
    sample.descriptors['Ir_19_20'] = uv.calc_ir_19_20(uv_refl)
    sample.descriptors['Ir_2_18'] = uv.calc_ir_2_18(uv_refl)
    sample.descriptors['Ir_4_5'] = uv.calc_ir_4_5(uv_refl)
    sample.descriptors['Ir_7_8'] = uv.calc_ir_7_8(uv_refl)
                        
for k,row in carbon_data.iterrows():
    
    sample_name = row['Sample']
    
    if sample_name not in collection.samples:
        
        #print(f"Нет спектральных данных образца {sample_name}")
        continue
    
    else:
        
        sample = collection.samples[sample_name]
        sample.org_carbon = row["C_org"]
        
        if sample.org_carbon is None:
            
            raise ValueError("Значение TOC неверно")
        
    #sample.add_tag("org_carbon")
    
    if sample.uv_vis_absorption is not None:
        sample.descriptors['Suva'] = uv.calc_suva(sample.uv_vis_absorption,toc=sample.org_carbon,debug=True)  
                
for k,row in geodata.iterrows():
    
    sample_name = row['Sample']
    
    if sample_name not in collection.samples:
    
        #print(f"Нет спектральных данных образца {sample_name}")
        continue
    
    else:
        
        sample = collection.samples[sample_name]
        sample.pH = row["pH"]
        sample.Eh = row['Eh']
        sample.latitude = row['latitude']
        sample.lontitude = row['longitude']
    
for k,row in parafac_data.iterrows():
    
    sample_name = row['Sample']
    
    if sample_name not in collection.samples:
    
        print(f"Нет спектральных данных образца {sample_name}")
        continue
    
    else:
        
        sample = collection.samples[sample_name]
        
        for column in parafac_data.columns:
            
            sample.descriptors[column] = row[column]
    
    #sample.add_tag("parafac")
    
print(f"Всего образцов в коллекции: {len(collection.samples)}")
collection.save_collection("KNP_samples.pkl")

data = collection.create_flexible_descriptors_table(0.25,sample_attributes=['org_carbon','pH','Eh'])
data.to_csv(Path.joinpath(STAT_PATH,"all_descriptors.csv"))
data.to_excel(Path.joinpath(STAT_PATH,"all_descriptors.xlsx"))

