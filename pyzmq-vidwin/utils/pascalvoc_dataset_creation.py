# Author - Piyush Yadav
# Insight Centre for Data Analytics
# Package- VidWIN Project

import os
from os import listdir
from os.path import isfile, join
from os import path
import shutil


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def read_voc_train_file(filepath, folderpath,foldertype, foldername, imagedirectory, imagefiles):
    # check if the folder exists  or not
    if path.exists(folderpath+foldertype):
        # create class folder
        create_folder(folderpath+foldertype+'/'+foldername)

    else:
        # create training folder
        create_folder(folderpath+foldertype)
        # create class folder
        create_folder(folderpath+foldertype+'/'+foldername)

    with open(filepath, 'r') as reader:
        # Read and print the entire file line by line
        for line in reader:
            if ' 1' in line:
                line_split = line.split(" ")
                img = line_split[0]+'.jpg'
                if img in imagefiles:
                    shutil.copy(imagedirectory+img, folderpath+foldertype+'/'+foldername)


def read_voc_annotation_directory(directorypath):
    files = [f for f in listdir(directorypath) if isfile(join(directorypath, f))]
    return files

def read_voc_image_directory(directorypath):
    files = [f for f in listdir(directorypath) if isfile(join(directorypath, f))]
    return files


# provide the directory path
voc_annotation_directory_path = '/home/dhaval/piyush/Usecases_dataset/pascal voc dataaset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/'
#voc_annotation_directory_path =  '/home/dhaval/piyush/Usecases_dataset/pascal voc dataaset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/test/'
voc_image_directory_path = '/home/dhaval/piyush/Usecases_dataset/pascal voc dataaset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
voc_dataset_creation_path = '/home/dhaval/piyush/Usecases_dataset/voc_dataset_created/'

directory_files = read_voc_annotation_directory(voc_annotation_directory_path)
image_files = read_voc_annotation_directory(voc_image_directory_path)


for files in directory_files:
    print('File Reading.....', files)
    if '_train.txt' in files:
        file_split = files.split("_")
        read_voc_train_file(voc_annotation_directory_path + files, voc_dataset_creation_path ,'training_data',file_split[0],voc_image_directory_path, image_files)

    if '_val.txt' in files:
        file_split = files.split("_")
        read_voc_train_file(voc_annotation_directory_path + files, voc_dataset_creation_path, 'validation_data',
                            file_split[0], voc_image_directory_path, image_files)

    if '_trainval.txt' in files:
        file_split = files.split("_")
        read_voc_train_file(voc_annotation_directory_path + files, voc_dataset_creation_path, 'trainingvalidation_data',
                            file_split[0], voc_image_directory_path, image_files)


