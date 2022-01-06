import SimpleITK as sitk
import os
import numpy as np
import torch.nn as nn
import torch
import random
import argparse
from shutil import rmtree


def partition_filenames(all_filenames, train_ratio=0.6, val_ratio=0.2, seed_num=44):
  random.seed(seed_num);
  train_filenames = random.sample(all_filenames, int(len(all_filenames) * train_ratio));
  random.seed(seed_num);
  val_filenames = random.sample(list(set(all_filenames) - set(train_filenames)), int(len(all_filenames) * val_ratio));
  test_filenames = list(set(all_filenames) - set(train_filenames).union(set(val_filenames)));
  return train_filenames, val_filenames, test_filenames
  

def resample(array, final_shape, mode='trilinear', input_or_label="input", num_classes=None, label_list=None):
    #Only HxWxD input is accepted!
    resampler = nn.Upsample(size=final_shape, mode=mode)
    if isinstance(array, np.ndarray):
        array = np.expand_dims(np.expand_dims(array, 0), 0)
        
    if input_or_label == "input":
        array = torch.from_numpy(array).float()
        final_array = resampler(array).numpy().reshape(final_shape)
        
    elif input_or_label == "label":
        array = torch.from_numpy(array)
        final_array = np.zeros(final_shape, dtype=np.uint8)
        
        if num_classes:
            for i in range(1, num_classes):
                binary_label = resampler((array == i).float()).numpy().reshape(final_shape) >= 0.5
                final_array[binary_label] = i
                
        elif label_list:
            for i, label_val in enumerate(label_list):
                if i == 0:
                    continue
                binary_label = resampler((array == label_val).float()).numpy().reshape(final_shape) >= 0.5
                final_array[binary_label] = i
            
    return final_array

    
def prepare_colab_for_brats(dimension=2, build=0):
    
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/BRATS2017/Processed' '/content/BRATS2017'")
    else:
        main_path = '/content/drive/My Drive/MiracGamblers/data/BRATS2017'
        hgg_folder = os.path.join(main_path, 'Brats17TrainingData', 'HGG')
        
        cbica_patient_names = [name for name in sorted(os.listdir(hgg_folder)) if 'CBICA' in name]
        tcia_patient_names = [name for name in sorted(os.listdir(hgg_folder)) if 'TCIA' in name]
        
        random.seed(42)
        train_patient_names = random.sample(cbica_patient_names, 66)
        val_patient_names = list(set(cbica_patient_names) - set(train_patient_names))
        test_patient_names = tcia_patient_names
        
        processed_folder = os.path.join(main_path, 'Processed')
        train_folder = os.path.join(processed_folder, 'train')
        val_folder = os.path.join(processed_folder, 'val')
        test_folder = os.path.join(processed_folder, 'test')
        
        for folder in [train_folder, val_folder, test_folder]:
          os.makedirs(os.path.join(folder, 'images'))
          os.makedirs(os.path.join(folder, 'labels'))
        
        for mode in ['train', 'val', 'test']:
          if mode == 'train':
            patient_names = train_patient_names
            folder = train_folder
          elif mode == 'val':
            patient_names = val_patient_names
            folder = val_folder
          elif mode == 'test':
            patient_names = test_patient_names
            folder = test_folder
        
          for patient_name in patient_names:
              
            patient_folder = os.path.join(hgg_folder, patient_name);
        
            t1ce_path = os.path.join(patient_folder, patient_name + '_t1ce.nii.gz');
            t2_path = os.path.join(patient_folder, patient_name + '_t2.nii.gz');
            flair_path = os.path.join(patient_folder, patient_name + '_flair.nii.gz');
            label_path = os.path.join(patient_folder, patient_name + '_seg.nii.gz');
        
            t1ce_array, t2_array, flair_array, label_array = list(map(sitk.ReadImage, [t1ce_path, t2_path, flair_path, label_path]));
            t1ce_array, t2_array, flair_array, label_array = list(map(sitk.GetArrayFromImage, [t1ce_array, t2_array, flair_array, label_array]));
            
            label_array[label_array==4]=3; #https://arxiv.org/pdf/1811.02629.pdf Beginning in 2017, label 3 is eliminated and combined with label 1. Therefore in the original data the only labels are [0,1,2,4]. This line of code is for fixing this problem.
        
            if dimension == 2:
                    
                num_slices = np.shape(t1ce_array)[0]
            
                for slice_num in range(num_slices):
                  t1ce_slice = t1ce_array[slice_num, :, :]
                  t2_slice = t2_array[slice_num, :, :]
                  flair_slice = flair_array[slice_num, :, :]
                  label_slice = label_array[slice_num, :, :]
            
                  combined_slice = np.stack([t1ce_slice, t2_slice, flair_slice], axis=2)
            
                  combined_image, label_image = list(map(sitk.GetImageFromArray, [combined_slice, label_slice]))
            
                  sitk.WriteImage(combined_image, os.path.join(folder, 'images', patient_name + '_slice{0}.nii.gz'.format(slice_num)))
                  sitk.WriteImage(label_image, os.path.join(folder, 'labels', patient_name + '_slice{0}.nii.gz'.format(slice_num)))
                  
            elif dimension == 3:
                
                combined_array = np.stack([t1ce_array, t2_array, flair_array], axis=3);
                combined_image = sitk.GetImageFromArray(combined_array);
                label_image = sitk.GetImageFromArray(label_array);
                
                sitk.WriteImage(combined_image, os.path.join(folder, 'images', patient_name + '.nii.gz'))
                sitk.WriteImage(label_image, os.path.join(folder, 'labels', patient_name + '.nii.gz'))
                
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/BRATS2017/Processed' '/content'")
                
                
def prepare_colab_for_mosmed(build=0):
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/MosMed/Processed' '/content/MosMed'")
    else:
        rmtree('/content/drive/My Drive/MiracGamblers/data/MosMed/Processed', ignore_errors=True)
        main_path = "/content/drive/My Drive/MiracGamblers/data/MosMed/"
        raw_path = os.path.join(main_path, "Raw")
        processed_path = os.path.join(main_path, "Processed")
        train_folder = os.path.join(processed_path, "train")
        test_folder = os.path.join(processed_path, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        desired_spacing = (8.0, 2.0, 2.0)
        
        scan_path = os.path.join(raw_path, "Scans")
        mask_path = os.path.join(raw_path, "Masks")
        
        random.seed(42)
        all_filenames = os.listdir(scan_path)
        train_filenames = random.sample(all_filenames, int(len(all_filenames)*0.8))
        test_filenames = list(set(all_filenames) - set(train_filenames))

        for mode in ['train', 'test']:
            if mode == 'train':
                filenames = train_filenames
                folder = train_folder
            elif mode == 'test':
                filenames = test_filenames
                folder = test_folder
            
            for filename in filenames:
              scan_filepath = os.path.join(scan_path, filename)
              mask_filepath = os.path.join(mask_path, filename[:-7] + "_mask.nii.gz")
              new_filepath = os.path.join(folder, filename[:-7] + ".npz")
            
              scan_img = sitk.ReadImage(scan_filepath)
              scan = sitk.GetArrayFromImage(scan_img)
            
              mask_img = sitk.ReadImage(mask_filepath)
              mask = sitk.GetArrayFromImage(mask_img)
            
              initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
              initial_shape = np.shape(scan)
            
              scaling = tuple(ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)) 
              final_shape = tuple(int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling))
            
              resampled_scan = resample(scan, final_shape)
              resampled_mask = resample(mask, final_shape)
            
              np.savez_compressed(new_filepath, X=resampled_scan.astype(np.float32), Y=resampled_mask.astype(np.uint8))
          
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/MosMed/Processed' '/content'")
        

def prepare_colab_for_covid19(build=0):
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Covid19/Processed' '/content/Covid19'")
    else:
        rmtree('/content/drive/My Drive/MiracGamblers/data/Covid19/Processed', ignore_errors=True)
        main_path = "/content/drive/My Drive/MiracGamblers/data/Covid19/"
        raw_path = os.path.join(main_path, "Raw")
        processed_path = os.path.join(main_path, "Processed")
        train_folder = os.path.join(processed_path, "train")
        val_folder = os.path.join(processed_path, "val")
        test_folder = os.path.join(processed_path, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        #desired_spacing = (6.0, 3.0, 3.0)
        desired_spacing = (6.0, 2.0, 2.0)
        print(desired_spacing)
        
        scan_path = os.path.join(raw_path, "Scans")
        mask_path = os.path.join(raw_path, "Masks")
        
        seed_num=44;
        random.seed(seed_num);
        all_filenames = os.listdir(scan_path);
        train_filenames = random.sample(all_filenames, int(len(all_filenames)*0.6));
        random.seed(seed_num);
        val_filenames = random.sample(list(set(all_filenames) - set(train_filenames)), int(len(all_filenames)*0.2));
        test_filenames = list(set(all_filenames) - set(train_filenames).union(set(val_filenames)));

        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                filenames = train_filenames
                folder = train_folder
            elif mode == 'val':
                filenames = val_filenames
                folder = val_folder
            elif mode == 'test':
                filenames = test_filenames
                folder = test_folder
            
            for filename in filenames:
              scan_filepath = os.path.join(scan_path, filename)
              mask_filepath = os.path.join(mask_path, filename)
              new_filepath = os.path.join(folder, filename[:-7] + ".npz")
            
              scan_img = sitk.ReadImage(scan_filepath)
              scan = sitk.GetArrayFromImage(scan_img)
              
              if "corona" in filename:
                  scan = np.clip(scan, -1250, 250)
                  scan = (scan + 1250) / (1250 + 250) * 255
            
              mask_img = sitk.ReadImage(mask_filepath)
              mask = sitk.GetArrayFromImage(mask_img)
            
              initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
              initial_shape = np.shape(scan)
            
              scaling = tuple(ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)) 
              final_shape = tuple(int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling))
            
              resampled_scan = resample(scan, final_shape, input_or_label="input")
              resampled_mask = resample(mask.astype(np.int16), final_shape, input_or_label="label", num_classes=4)
            
              np.savez_compressed(new_filepath, X=resampled_scan.astype(np.float32), Y=resampled_mask.astype(np.uint8))
          
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Covid19/Processed' '/content/Covid19'")
        
        
def prepare_colab_for_whs(build=0):
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/MM-WHS/Processed' '/content/MM-WHS'")
    else:
        rmtree('/content/drive/My Drive/MiracGamblers/data/MM-WHS/Processed', ignore_errors=True)
        main_path = "/content/drive/My Drive/MiracGamblers/data/MM-WHS/"
        raw_path = os.path.join(main_path, "Raw")
        processed_path = os.path.join(main_path, "Processed")
        train_folder = os.path.join(processed_path, "train")
        val_folder = os.path.join(processed_path, "val")
        test_folder = os.path.join(processed_path, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        #desired_spacing = (1.5, 2.0, 2.0)
        desired_spacing = (1.5, 1.5, 1.5)
        #desired_spacing = (1.5, 1.0, 1.0)
        print(desired_spacing)
        label_list = [0, 205, 420, 500, 550, 600, 820, 850]
        
        filename_list = [filename.strip("_label.nii.gz") for filename in os.listdir(raw_path) if filename.endswith("_label.nii.gz")]
        train_filenames, val_filenames, test_filenames = partition_filenames(filename_list)

        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                filenames = train_filenames
                folder = train_folder
            elif mode == 'val':
                filenames = val_filenames
                folder = val_folder
            elif mode == 'test':
                filenames = test_filenames
                folder = test_folder
            
            for filename in filenames:
                scan_filepath = os.path.join(raw_path, filename + "_image.nii.gz")
                mask_filepath = os.path.join(raw_path, filename + "_label.nii.gz")
                new_filepath = os.path.join(folder, filename + ".npz")
                
                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)
                
                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)
                
                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)
                
                scaling = tuple(ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)) 
                final_shape = tuple(int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling))
                
                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(mask.astype(np.int16), final_shape, input_or_label="label", label_list=label_list)
                
                np.savez_compressed(new_filepath, X=resampled_scan.astype(np.float32), Y=resampled_mask.astype(np.uint8))
          
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/MM-WHS/Processed' '/content/MM-WHS'")
        
        
def prepare_colab_for_pancreas(build=0):
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Pancreas-CT/Processed' '/content/Pancreas-CT'")
    else:
        rmtree('/content/drive/My Drive/MiracGamblers/data/Pancreas-CT/Processed', ignore_errors=True)
        main_path = "/content/drive/My Drive/MiracGamblers/data/Pancreas-CT/"
        raw_path = os.path.join(main_path, "Raw")
        processed_path = os.path.join(main_path, "Processed")
        train_folder = os.path.join(processed_path, "train")
        val_folder = os.path.join(processed_path, "val")
        test_folder = os.path.join(processed_path, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        desired_spacing = (3.0, 3.0, 3.0)
        print(desired_spacing)
        
        scan_path = os.path.join(raw_path, "Scans")
        mask_path = os.path.join(raw_path, "Masks")
        
        patient_list = [filename.strip("label").strip(".nii.gz") for filename in os.listdir(mask_path) if filename.endswith(".nii.gz")]
        train_patients, val_patients, test_patients = partition_filenames(patient_list)

        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                patients = train_patients
                folder = train_folder
            elif mode == 'val':
                patients = val_patients
                folder = val_folder
            elif mode == 'test':
                patients = test_patients
                folder = test_folder
            
            for patient in patients:
                scan_filename = "PANCREAS_" + patient
                mask_filename = "label" + patient
                
                scan_filepath = os.path.join(scan_path, scan_filename + ".nii.gz")
                mask_filepath = os.path.join(mask_path, mask_filename + ".nii.gz")
                new_filepath = os.path.join(folder, patient + ".npz")
                
                scan_img = sitk.ReadImage(scan_filepath)
                scan = sitk.GetArrayFromImage(scan_img)
                
                mask_img = sitk.ReadImage(mask_filepath)
                mask = sitk.GetArrayFromImage(mask_img)
                
                initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
                initial_shape = np.shape(scan)
                
                scaling = tuple(ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)) 
                final_shape = tuple(int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling))
                
                resampled_scan = resample(scan, final_shape, input_or_label="input")
                resampled_mask = resample(mask.astype(np.int16), final_shape, input_or_label="label", num_classes=2)
                
                np.savez_compressed(new_filepath, X=resampled_scan.astype(np.float32), Y=resampled_mask.astype(np.uint8))
          
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Pancreas-CT/Processed' '/content/Pancreas-CT'")
        

def prepare_colab_for_spleen(build=0):
    if not build:
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Spleen/Processed' '/content/Spleen'")
    else:
        rmtree('/content/drive/My Drive/MiracGamblers/data/Spleen/Processed', ignore_errors=True)
        main_path = "/content/drive/My Drive/MiracGamblers/data/Spleen/"
        raw_path = os.path.join(main_path, "Raw")
        processed_path = os.path.join(main_path, "Processed")
        train_folder = os.path.join(processed_path, "train")
        val_folder = os.path.join(processed_path, "val")
        test_folder = os.path.join(processed_path, "test")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        desired_spacing = (5.0, 2.0, 2.0)
        print(desired_spacing)
        
        scan_path = os.path.join(raw_path, "Scans")
        mask_path = os.path.join(raw_path, "Masks")
        
        seed_num=44;
        random.seed(seed_num);
        all_filenames = os.listdir(scan_path);
        train_filenames = random.sample(all_filenames, int(len(all_filenames)*0.6));
        random.seed(seed_num);
        val_filenames = random.sample(list(set(all_filenames) - set(train_filenames)), int(len(all_filenames)*0.2));
        test_filenames = list(set(all_filenames) - set(train_filenames).union(set(val_filenames)));

        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                filenames = train_filenames
                folder = train_folder
            elif mode == 'val':
                filenames = val_filenames
                folder = val_folder
            elif mode == 'test':
                filenames = test_filenames
                folder = test_folder
            
            for filename in filenames:
              scan_filepath = os.path.join(scan_path, filename)
              mask_filepath = os.path.join(mask_path, filename)
              new_filepath = os.path.join(folder, filename[:-7] + ".npz")
            
              scan_img = sitk.ReadImage(scan_filepath)
              scan = sitk.GetArrayFromImage(scan_img)
            
              mask_img = sitk.ReadImage(mask_filepath)
              mask = sitk.GetArrayFromImage(mask_img)
            
              initial_spacing = tuple(np.roll(scan_img.GetSpacing(), 1))
              initial_shape = np.shape(scan)
            
              scaling = tuple(ele1 / ele2 for ele1, ele2 in zip(desired_spacing, initial_spacing)) 
              final_shape = tuple(int(ele1 // ele2) for ele1, ele2 in zip(initial_shape, scaling))
            
              resampled_scan = resample(scan, final_shape, input_or_label="input")
              resampled_mask = resample(mask.astype(np.int16), final_shape, input_or_label="label", num_classes=4)
            
              np.savez_compressed(new_filepath, X=resampled_scan.astype(np.float32), Y=resampled_mask.astype(np.uint8))
          
        os.system("cp -a '/content/drive/My Drive/MiracGamblers/data/Spleen/Processed' '/content/Spleen'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dimension',
        help='process the dataset as whole volume (3) or slices (2)',
        type=int,
        default=2
    )
    parser.add_argument(
        '--build',
        help='process the raw files from scratch if they are not already built',
        type=int,
        default=0
    )
    parser.add_argument(
        '--dataset',
        help='dataset to use: covid19, whs, brats or mosmed'
    )
    #args, _ = parser.parse_known_args()
    options = parser.parse_args()
    
    if options.dataset == 'mosmed':
        prepare_colab_for_mosmed(build=options.build)
    elif options.dataset == 'covid19':
        prepare_colab_for_covid19(build=options.build)
    elif options.dataset == 'whs':
        prepare_colab_for_whs(build=options.build)
    elif options.dataset == 'pancreas':
        prepare_colab_for_pancreas(build=options.build)
    elif options.dataset == 'spleen':
        prepare_colab_for_spleen(build=options.build)
    elif options.dataset == 'brats':
        prepare_colab_for_brats(dimension=options.dimension, build=options.build)
    