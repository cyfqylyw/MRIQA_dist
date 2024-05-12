import os
import glob
import numpy as np
import nibabel as nib


def verify_ds_type(ds_name, sub_type):
    """
    validate the dataset information

    Input:
    :param ds_name: dataset name, one of ['OASIS', 'NFBS', 'IXI', 'QTAB', 'ARC', 'ABIDE']
    :param sub_type: 
        if ds_name in ['OASIS', 'NFBS'], sub_type should be 'T1w'
        if ds_name in ['IXI', 'QTAB', 'ARC'], sub_type be in ['T1w', 'T2w']
        if ds_name in ['ABIDE'], sub_type should be in ['ccs', 'cpac', 'dparsf', 'niak']
    """
    ds_list = ['OASIS', 'NFBS', 'IXI', 'QTAB', 'ARC', 'ABIDE']
    if ds_name not in ds_list:
        raise ValueError(f"Illegal dataset name {ds_name}, please apply one of dataset name in {ds_list}")
    
    if ds_name in ['OASIS', 'NFBS'] and sub_type != 'T1w':
        raise ValueError(f"Dataset {ds_name} have no images except for T1w, please set sub_type as 'T1w'.")
    
    if ds_name in ['IXI', 'QTAB', 'ARC'] and sub_type not in ['T1w', 'T2w']:
        raise ValueError(f"sub_type for dataset {ds_name} should be in ['T1w', 'T2w'].")
    
    preprocess_list = ['ccs', 'cpac', 'dparsf', 'niak']
    if ds_name == 'ABIDE' and sub_type not in preprocess_list:
        raise ValueError(f"sub_type for dataset {ds_name} should be in {preprocess_list}.")


def get_dataset_file_list(ds_name, sub_type):
    """
    first validate the dataset information
    then output the list of all MRI files
    """

    verify_ds_type(ds_name, sub_type)
    
    if ds_name == 'OASIS':
        file_paths = []
        for i in range(1, 13):
            disc_folder = './datasets/OASIS/disc{}/'.format(i)
            for root, dirs, files in os.walk(disc_folder):
                for file in files:
                    if file.endswith("_anon.img") and 'RAW' in root:
                        file_paths.append(os.path.join(root, file))

    elif ds_name == 'NFBS':
        NFBS_folder_path = "./datasets/NFBS/library_NFBS/NFBS/stx/1mm/"
        file_paths = glob.glob(os.path.join(NFBS_folder_path, '*.mnc'))

    elif ds_name == 'IXI' and sub_type == 'T1w':
        # IXI (T1w)
        IXI_T1_folder_path = './datasets/IXI-T1/'
        file_paths = glob.glob(os.path.join(IXI_T1_folder_path, '*.nii.gz'))
    
    elif ds_name == 'IXI' and sub_type == 'T2w':
        # IXI (T2w)
        IXI_T2_folder_path = './datasets/IXI-T2/'
        file_paths = glob.glob(os.path.join(IXI_T2_folder_path, '*.nii.gz'))
        file_paths = [x for x in file_paths if x not in ['./datasets/IXI-T2/IXI014-HH-1236-T2.nii.gz', './datasets/IXI-T2/IXI013-HH-1212-T2.nii.gz', './datasets/IXI-T2/IXI012-HH-1211-T2.nii.gz', ]]
        

    elif ds_name == 'QTAB' and sub_type == 'T1w':
        # QTAB (T1w)
        directory = "./datasets/QTAB/ds004146-download/"
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_MP2RAGE.nii.gz"):  # T1w ends up with MP2RAGE
                    file_paths.append(os.path.join(root, file))

    elif ds_name == 'QTAB' and sub_type == 'T2w':
        # QTAB (T2w)
        directory = "./datasets/QTAB/ds004146-download/"
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if "T2w_TSE" in file and file.endswith(".nii.gz"):
                    file_paths.append(os.path.join(root, file))
    
    elif ds_name == 'ARC' and sub_type == 'T1w':
        # ARC (T1w)
        directory = "./datasets/ARC/ds004884-download/"
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_T1w.nii.gz"):
                    if file not in ['sub-M2030_ses-1454_acq-tfl3p2_run-3_T1w.nii.gz']:
                        file_paths.append(os.path.join(root, file))

    elif ds_name == 'ARC' and sub_type == 'T2w':
        # ARC (T2w)
        directory = "./datasets/ARC/ds004884-download/"
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("_T2w.nii.gz"):
                    if file not in ['sub-M2005_ses-1034_acq-spc3_run-2_T2w.nii.gz', 'ub-M2269_ses-767_acq-spc3p2_run-4_T2w.nii.gz', 'sub-M2105_ses-964_acq-spc3_run-11_T2w.nii.gz', 'sub-M2036_ses-1115_acq-spc3_run-3_T2w.nii.gz']:
                        file_paths.append(os.path.join(root, file))

    elif ds_name == 'ABIDE':
        ABIDE_folder_path = f"./datasets/ABIDE/Outputs/{sub_type}/nofilt_noglobal/reho/"
        file_paths = glob.glob(os.path.join(ABIDE_folder_path, '*.nii.gz'))

    file_paths.sort()
    return file_paths


def min_shape(ds_name, sub_type):
    """
    first validate the dataset information
    then output the minimal shape
    """

    verify_ds_type(ds_name, sub_type)

    if ds_name == 'OASIS':
        ms = (256, 256, 128, 1)
    elif ds_name == 'NFBS':
        ms = (193, 229, 193)
    elif ds_name == 'IXI' and sub_type == 'T1w':
        ms = (256, 256, 130)
    elif ds_name == 'IXI' and sub_type == 'T2w':
        ms = (256, 256, 120)
    elif ds_name == 'QTAB' and sub_type == 'T1w':
        ms = (176, 300, 320)
    elif ds_name == 'QTAB' and sub_type == 'T2w':
        ms = (768, 768, 50)
    elif ds_name == 'ARC' and sub_type == 'T1w':
        ms = (160, 256, 256)
    elif ds_name == 'ARC' and sub_type == 'T2w':
        ms = (160, 256, 256)
    elif ds_name == 'ABIDE':
        ms = (61, 73, 61)

    return ms


dics = [
    "./datasets/src/",
    "./datasets/src/OASIS_T1w/",
    "./datasets/src/NFBS_T1w/",
    "./datasets/src/IXI_T1w/",
    "./datasets/src/IXI_T2w/",
    "./datasets/src/QTAB_T1w/",
    "./datasets/src/QTAB_T2w/",
    "./datasets/src/ARC_T1w/",
    "./datasets/src/ARC_T2w/",
    "./datasets/src/ABIDE_ccs/",
    "./datasets/src/ABIDE_cpac/",
    "./datasets/src/ABIDE_dparsf/",
    "./datasets/src/ABIDE_niak/"
]

for dic in dics:
    os.makedirs(dic, exist_ok=True)


ds_type_combinations = [('OASIS', 'T1w'), ('NFBS', 'T1w'), ('IXI', 'T1w'), ('IXI', 'T2w'), ('QTAB', 'T1w'), ('QTAB', 'T2w'), ('ARC', 'T1w'), ('ARC', 'T2w'), ('ABIDE', 'ccs'), ('ABIDE', 'cpac'), ('ABIDE', 'dparsf'), ('ABIDE', 'niak')]

base_path = "./datasets/src/"

for ds_name, sub_type in ds_type_combinations:
    print('\n' * 10)
    print(ds_name, sub_type)

    file_paths = get_dataset_file_list(ds_name, sub_type)
    mshape = min_shape(ds_name, sub_type)[:3]
    ds_type_path = ds_name + '_' + sub_type
    
    for filename in file_paths:
        data = np.array(nib.load(filename).get_fdata())

        if ds_name == 'OASIS':  # with shape (256, 256, 128, 1)
            data = data[:, :, :, 0]
        
        L, W, H = data.shape
        l, w, h = mshape
        l_start, l_end = (L // 2) - (l // 2), (L // 2) - (l // 2) + l
        w_start, w_end = (W // 2) - (w // 2), (W // 2) - (w // 2) + w
        h_start, h_end = (H // 2) - (h // 2), (H // 2) - (h // 2) + h

        region_data = data[l_start:l_end, w_start:w_end, h_start:h_end]
        print(region_data.shape)

        if region_data.shape == mshape:
            fn = filename.split('/')[-1].split('.')[0] + '.npy'
            np.save(os.path.join(base_path, ds_type_path, fn), region_data)
