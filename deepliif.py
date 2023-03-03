### This file contains all necessary code for reproducing the deepliif experiment in our manuscript.

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from time import time
import seaborn as sns
import os

import SN_master.stain_utils as utils
import SN_master.stainNorm_Reinhard as SR
import SN_master.stainNorm_Macenko as SM
import SN_master.stainNorm_Vahadane as SV
from skimage.metrics import structural_similarity as ssim
import spams
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import ImageOps
from utils.image_io import *
import cv2
import seaborn as sns
import scipy
import pandas as pd
from SN_master.stainNorm_Macenko import get_stain_matrix as get_stain_matrix_M
from SN_master.stainNorm_Vahadane import get_stain_matrix as get_stain_matrix_V
from SN_master.stain_utils import standardize_brightness

import skimage
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
import numpy
import PIL
import matplotlib.pyplot as plt
from utils.image_io import *
from itertools import compress
import shutil
from configparser import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
import random
# Python3 program change RGB Color
# Model to HSV Color Model
from mpl_toolkits import mplot3d
def rgb_to_hsv(rgb):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = rgb
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)  # maximum of r, g, b
    cmin = min(r, g, b)  # minimum of r, g, b
    diff = cmax - cmin  # diff of cmax and cmin.

    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0

    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100

    # compute v
    v = cmax * 100
    return [h, s, v]

def OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB
    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    # assert OD.min() >= 0, 'Negative optical density'
    OD = np.clip(OD, a_min = 0, a_max=None)
    return (255 * 10**(-1 * OD)).astype(np.uint8)

def rgb_to_hex(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (r,g,b)


def stain_vector_to_hex(SV, OD=1):
    rgb = OD_to_RGB(OD * SV)
    hex = rgb_to_hex(rgb)
    return hex

def fold(Hues):
    return [(Hue+180)%360-180 for Hue in Hues]

# TODO: Configure runtime vs steps, 55min for 6000 steps?
# TODO: Upload configs
# TODO: Sbatch
# TODO: Download
# TODO: Show color space
# TODO: Calculate correlations

def dapi2gray(rgb):
    """3 channel grayscale to 1 channel grayscale."""
    return np.dot(rgb[...,:3], [0.33, 0.33, 0.33])

IMAGE_FOLDER = r'E:\data\ihcmultiplex\DeepLIIF_Testing_Set'
PATCH_FOLDER = r'E:\data\ihcmultiplex\patches'
IHC_FOLDER = r'E:\data\ihcmultiplex\patches\IHC'
Hem_FOLDER = r'E:\data\ihcmultiplex\patches\Hem'
DAPI_FOLDER = r'E:\data\ihcmultiplex\patches\DAPI'
LAP2_FOLDER = r'E:\data\ihcmultiplex\patches\Lap2'
MARKER_FOLDER = r'E:\data\ihcmultiplex\patches\Marker'
SEG_FOLDER = r'E:\data\ihcmultiplex\patches\Seg'

MACENKO_FOLDER = r'E:\data\ihcmultiplex\patches\Macenko'
VAHADANE_FOLDER = r'E:\data\ihcmultiplex\patches\Vahadane'
OURS_FOLDER = r'E:\data\ihcmultiplex\patches\Ours'
OUTPUT_FOLDER = r'E:\data\ihcmultiplex\patches\output'

PATCH_SIZE = 512
DEEPLIIF_LIST = ['IHC', 'Hem', 'DAPI', 'Lap2', 'Marker', 'Seg']

cfg_example_file = 'ihc_bg.cfg'
cfg_input_folder = '/home/jachen/scratch/PGDIP/DeepLIIF/testset_scf/input'
cfg_output_folder = '/home/jachen/scratch/PGDIP/DeepLIIF/testset_scf/output'

def path_to_np(path):
    img = load(path)
    img_np = pil_to_np(img, with_transpose=False)
    return img_np


def np_to_gray(np_img):
    np_img_uint8 = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img_uint8)


def generate_cfgs(cfg_example_file, cfg_input_folder, cfg_output_folder, PATCH_FOLDER):
    # Read config.ini file
    config_object = ConfigParser()
    config_object.read(cfg_example_file)

    # Get the system section
    systeminfo = config_object["system"]
    systeminfo['input_folder']  = cfg_input_folder
    systeminfo['output_folder']  = cfg_output_folder

    HE_ROI_path = PATCH_FOLDER
    for fname in os.listdir(HE_ROI_path):
        # Update the password
        systeminfo["input"] = fname
        # Write changes back to file
        config_fname = fname.replace('.png', '.cfg')
        with open(config_fname, 'w') as conf:
            config_object.write(conf)

def extract_ROI(IMAGE_FOLDER, DEEPLIIF_LIST, PATCH_SIZE=512):
    for fname in os.listdir(IMAGE_FOLDER):
        image_path = os.path.join(IMAGE_FOLDER, fname)
        image = load(image_path)
        for i in range(int(image.width/PATCH_SIZE)):
            i_crop = image.crop((i*PATCH_SIZE, 0, (i+1)*PATCH_SIZE, PATCH_SIZE))
            folder = os.path.join(PATCH_FOLDER, DEEPLIIF_LIST[i])
            i_crop.save(os.path.join(folder, fname))

def separation_with_stain_vectors_M_V(Macenko_folder, Vahadane_folder, Ours_folder, IHC_folder):
    STEPS = '5999'
    S1 = '_Stained1_' + STEPS + '.png'
    S2 = '_Stained2_' + STEPS + '.png'
    C1 = '_Con1_' + STEPS + '.png'
    C2 = '_Con2_' + STEPS + '.png'

    stain_matrix_M_all = np.zeros((0, 2, 3))
    stain_matrix_V_all = np.zeros((0, 2, 3))
    fnames = []

    for ind, i in enumerate(os.listdir(IHC_folder)):
        im_path = os.path.join(IHC_folder, i)
        fname = im_path.split('.png')[0].split('\\')[-1]

        h = path_to_np(im_path)
        # I_path = os.path.join(Ours_folder, 'HE' + fname + '_Con1_5999.png')
        # I = path_to_np(I_path)
        # h_path = os.path.join(HE_folder, 'HE' + fname + '.png')
        # h = path_to_np(h_path)

        n = SM.Normalizer()
        n.fit(h)
        m = n.hematoxylin(h)
        m_color = n.hematoxylin_color(h)
        m_2 = n.eosin(h)
        m_color_2 = n.eosin_color(h)

        n = SV.Normalizer()
        n.fit(h)
        v = n.hematoxylin(h)
        v_color = n.hematoxylin_color(h)
        v_2 = n.eosin(h)
        v_color_2 = n.eosin_color(h)

        m_img = np_to_gray(m)
        m_img.save(os.path.join(Macenko_folder, fname + '_Con1.png'))
        v_img = np_to_gray(v)
        v_img.save(os.path.join(Vahadane_folder, fname + '_Con1.png'))

        m_img_2 = np_to_gray(m_2)
        m_img_2.save(os.path.join(Macenko_folder, fname + '_Con2.png'))
        v_img_2 = np_to_gray(v_2)
        v_img_2.save(os.path.join(Vahadane_folder, fname + '_Con2.png'))

        m_color_img = np_to_pil(np.moveaxis(m_color * 255, 2, 0))
        m_color_img.save(os.path.join(Macenko_folder, fname + '_Stain1.png'))
        v_color_img = np_to_pil(np.moveaxis(v_color * 255, 2, 0))
        v_color_img.save(os.path.join(Vahadane_folder, fname + '_Stain1.png'))

        m_color_img_2 = np_to_pil(np.moveaxis(m_color_2 * 255, 2, 0))
        m_color_img_2.save(os.path.join(Macenko_folder, fname + '_Stain2.png'))
        v_color_img_2 = np_to_pil(np.moveaxis(v_color_2 * 255, 2, 0))
        v_color_img_2.save(os.path.join(Vahadane_folder, fname + '_Stain2.png'))

        h_standard = standardize_brightness(h)
        stain_M = get_stain_matrix_M(h_standard)
        stain_V = get_stain_matrix_V(h_standard)

        stain_M = np.reshape(stain_M, (1, 2, 3))
        stain_V = np.reshape(stain_V, (1, 2, 3))

        stain_matrix_M_all = np.append(stain_matrix_M_all, stain_M, axis=0)
        stain_matrix_V_all = np.append(stain_matrix_V_all, stain_V, axis=0)

        print(fname)
        print(stain_M)
        print(stain_V)
        fnames.append(fname)


    with open('stain_vector_M.npy', 'wb') as f:
        np.save(f, stain_matrix_M_all, allow_pickle=True)
    with open('stain_vector_V.npy', 'wb') as f:
        np.save(f, stain_matrix_V_all, allow_pickle=True)
    with open('fnames.npy', 'wb') as f:
        np.save(f, fnames, allow_pickle=True)


def extract_results_pgdips(Output_folder, Ours_folder):
    STEPS = '5999'
    S1 = '_Stained1_'+STEPS+'.png'
    S2 = '_Stained2_'+STEPS+'.png'
    C1 = '_Con1_'+STEPS+'.png'
    C2 = '_Con2_'+STEPS+'.png'
    All = '_All_'+STEPS+'.png'
    WB = '_WB_'+STEPS+'.png'
    stain_matrix_P_all = np.zeros((0,2,3))
    Modalities = [S1,S2,C1,C2,All,WB]
    SaveNames1 = ['_Stained1.png', '_Stained2.png', '_Con1.png', '_Con2.png', '_All.png', '_WB.png']
    SaveNames2 = ['_Stained2.png', '_Stained1.png', '_Con2.png', '_Con1.png', '_All.png', '_WB.png']

    for fname in os.listdir(Output_folder):
        try:
            subfolder = os.path.join(Output_folder, fname)
            csv_name = os.path.join(subfolder, fname+'_out.csv')
            df = pd.read_csv(csv_name)
            last_epoch = df.iloc[-1,:]
            if last_epoch['S1R'] > last_epoch['S2R']:
                stain_P = np.array([[[last_epoch['S1R'], last_epoch['S1G'], last_epoch['S1B']],
                          [last_epoch['S2R'], last_epoch['S2G'], last_epoch['S2B']]]], dtype=float)
                for i, m in enumerate(Modalities):
                    src = os.path.join(subfolder, fname+m)
                    dst = os.path.join(Ours_folder, fname+SaveNames1[i])
                    shutil.copy(src, dst)
            else:
                stain_P = np.array([[[last_epoch['S2R'], last_epoch['S2G'], last_epoch['S2B']],
                          [last_epoch['S1R'], last_epoch['S1G'], last_epoch['S1B']]]], dtype=float)
                for i, m in enumerate(Modalities):
                    src = os.path.join(subfolder, fname+m)
                    dst = os.path.join(Ours_folder, fname+SaveNames2[i])
                    shutil.copy(src, dst)
            stain_matrix_P_all = np.append(stain_matrix_P_all, stain_P, axis = 0)
            # print(stain_P)
        except:
            print(fname)
        # im_path = os.path.join(DAPI_folder, i)
        # fname = im_path.split('DAPI')[2].split('.png')[0]



    with open('stain_vector_P_bg.npy', 'wb') as f:
        np.save(f, stain_matrix_P_all, allow_pickle=True)
    print(1)

    with open('stain_vector_P_bg.npy', 'rb') as f:
        c = np.load(f,allow_pickle=True)

def calculate_correlations(Macenko_folder, Vahadane_folder, Ours_folder):
    DEEPLIIF_LIST = ['IHC', 'Hem', 'DAPI', 'Lap2', 'Marker', 'Seg']
    IMAGE_FOLDER = r'E:\data\ihcmultiplex\DeepLIIF_Testing_Set'
    PATCH_FOLDER = r'E:\data\ihcmultiplex\patches'
    IHC_FOLDER = r'E:\data\ihcmultiplex\patches\IHC'
    Hem_FOLDER = r'E:\data\ihcmultiplex\patches\Hem'
    DAPI_FOLDER = r'E:\data\ihcmultiplex\patches\DAPI'
    LAP2_FOLDER = r'E:\data\ihcmultiplex\patches\Lap2'
    MARKER_FOLDER = r'E:\data\ihcmultiplex\patches\Marker'
    SEG_FOLDER = r'E:\data\ihcmultiplex\patches\Seg'

    mc_all = []
    vc_all = []
    pc_all = []
    ms_all = []
    vs_all = []
    ps_all = []

    mc_m_all = []
    vc_m_all = []
    pc_m_all = []
    ms_m_all = []
    vs_m_all = []
    ps_m_all = []

    filenames = []

    for fname in os.listdir(IHC_FOLDER):
        img_name = fname.strip('.png')
        filenames.append(img_name)
        ihc_path = os.path.join(IHC_FOLDER, fname)
        I = path_to_np((ihc_path))
        dapi_path = os.path.join(DAPI_FOLDER, fname)
        D = path_to_np(dapi_path)

        hem_path = os.path.join(Hem_FOLDER, fname)
        H = path_to_np(hem_path)
        marker_path = os.path.join(MARKER_FOLDER, fname)
        M = path_to_np(marker_path)

        mh_path = os.path.join(Macenko_folder, img_name + '_Con1.png')
        mh = path_to_np(mh_path)
        md_path = os.path.join(Macenko_folder, img_name + '_Con2.png')
        md = path_to_np(md_path)
        mhc_path = os.path.join(Macenko_folder, img_name + '_Stain1.png')
        mhc = path_to_np(mhc_path)
        mdc_path = os.path.join(Macenko_folder, img_name + '_Stain2.png')
        mdc = path_to_np(mdc_path)

        vh_path = os.path.join(Vahadane_folder, img_name + '_Con1.png')
        vh = path_to_np(vh_path)
        vd_path = os.path.join(Vahadane_folder, img_name + '_Con2.png')
        vd = path_to_np(vd_path)
        vhc_path = os.path.join(Vahadane_folder, img_name + '_Stain1.png')
        vhc = path_to_np(vhc_path)
        vdc_path = os.path.join(Vahadane_folder, img_name + '_Stain2.png')
        vdc = path_to_np(vdc_path)

        ph_path = os.path.join(Ours_folder, img_name + '_Con1.png')
        ph = path_to_np(ph_path)
        pdab_path = os.path.join(Ours_folder, img_name + '_Con2.png')
        pdab = path_to_np(pdab_path)
        phc_path = os.path.join(Ours_folder, img_name + '_Stained1.png')
        phc = path_to_np(phc_path)
        pdc_path = os.path.join(Ours_folder, img_name + '_Stained2.png')
        pdc = path_to_np(pdc_path)

        Db = dapi2gray(D)
        Db = Db * 255.0 / np.max(D)
        Mb = dapi2gray(M)
        Mb = Mb * 255.0 / np.max(Mb)
        Db, Mb = 255-Db, 255-Mb
        Db, Mb = Db * 255.0 / np.max(Db), Mb * 255.0/np.max(Mb)

        mc = scipy.stats.pearsonr(Db.ravel(), mh.ravel())[0]
        vc = scipy.stats.pearsonr(Db.ravel(), vh.ravel())[0]
        pc = scipy.stats.pearsonr(Db.ravel(), ph.ravel())[0]

        ms = ssim(Db, mh,
                  data_range=Db.max() - Db.min())
        vs = ssim(Db, vh,
                  data_range=Db.max() - Db.min())
        ps = ssim(Db, ph,
                  data_range=Db.max() - Db.min())

        mc_m = scipy.stats.pearsonr(Mb.ravel(), md.ravel())[0]
        vc_m = scipy.stats.pearsonr(Mb.ravel(), vd.ravel())[0]
        pc_m = scipy.stats.pearsonr(Mb.ravel(), pdab.ravel())[0]

        ms_m = ssim(Mb, md,
                  data_range=Mb.max() - Mb.min())
        vs_m = ssim(Mb, vd,
                  data_range=Mb.max() - Mb.min())
        ps_m = ssim(Mb, pdab,
                  data_range=Mb.max() - Mb.min())

        mc_all.append(mc)
        vc_all.append(vc)
        pc_all.append(pc)

        ms_all.append(ms)
        vs_all.append(vs)
        ps_all.append(ps)

        mc_m_all.append(mc_m)
        vc_m_all.append(vc_m)
        pc_m_all.append(pc_m)

        ms_m_all.append(ms_m)
        vs_m_all.append(vs_m)
        ps_m_all.append(ps_m)

        plt.clf()
        fig, axes = plt.subplots(2, 5, figsize=(12, 4), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(I.astype('uint16'))
        ax[1].imshow(H.astype('uint16'))
        ax[2].imshow(mhc.astype('uint16'))
        ax[3].imshow(vhc.astype('uint16'))
        ax[4].imshow(phc.astype('uint16'))
        ax[5].imshow(D.astype('uint16'))
        ax[6].imshow(M.astype('uint16'))
        ax[7].imshow(mdc.astype('uint16'))
        ax[8].imshow(vdc.astype('uint16'))
        ax[9].imshow(pdc.astype('uint16'))
        ax[0].title.set_text('IHC')
        ax[1].title.set_text('Hematoxylin')
        ax[2].title.set_text(f'Macenko_{ms:.3f}')
        ax[3].title.set_text(f'Vahadane_{vs:.3f}')
        ax[4].title.set_text(f'PGDIPS_{ps:.3f}')
        ax[5].title.set_text('DAPI')
        ax[6].title.set_text('Marker')
        ax[7].title.set_text(f'Macenko_{ms_m:.3f}')
        ax[8].title.set_text(f'Vahadane_{vs_m:.3f}')
        ax[9].title.set_text(f'PGDIPS_{ps_m:.3f}')
        plt.savefig('ssim_hematoxylin'+fname+'.png')
        plt.close()

        print(fname)


    plt.clf()
    sns.set(font_scale=1.2)
    sns.set_style("white")
    sns.set_palette("pastel")
    # sns.despine(offset=5, trim=True)
    boxplot = sns.violinplot(x='method', y='pearsonr', data=df)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('Pearson correlation', fontsize=12)
    plt.savefig('violin_pear.png', dpi=600)

    plt.clf()
    # sns.set(font_scale=1.2)
    # sns.set_style("white")
    # sns.set_palette("pastel")

    boxplot = sns.violinplot(x='method', y='ssim', data=df)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('SSIM', fontsize=12)
    plt.savefig('violin_ssim.png', dpi=600)

    corr_all = mc_all + vc_all + pc_all
    ssim_all = ms_all + vs_all + ps_all
    IDs = filenames * 3
    corr_m_all = mc_m_all + vc_m_all + pc_m_all
    ssim_m_all = ms_m_all + vs_m_all + ps_m_all
    method = ['Macenko'] * len(mc_all) + ['Vahadane'] * len(vc_all) + ['PGDIPS'] * len(pc_all)
    df_corr = pd.DataFrame({'method': method, 'ID': filenames, 'pearsonr_hem': corr_all, 'ssim_hem': ssim_all,
                            'pearsonr_dab': corr_m_all, 'ssim_dab': ssim_m_all})

    plt.clf()
    sns.set(font_scale=1.2)
    sns.set_style("white")
    sns.set_palette("pastel")
    # sns.despine(offset=5, trim=True)
    boxplot = sns.violinplot(x='method', y='pearsonr_hem', data=df_corr)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('Pearson correlation', fontsize=12)
    plt.savefig('violin_pear_hem.png', dpi=600)

    plt.clf()
    sns.set(font_scale=1.2)
    sns.set_style("white")
    sns.set_palette("pastel")
    # sns.despine(offset=5, trim=True)
    boxplot = sns.violinplot(x='method', y='ssim_hem', data=df_corr)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('Structral similarity', fontsize=12)
    plt.savefig('violin_ssim_hem.png', dpi=600)

    plt.clf()
    sns.set(font_scale=1.2)
    sns.set_style("white")
    sns.set_palette("pastel")
    # sns.despine(offset=5, trim=True)
    boxplot = sns.violinplot(x='method', y='pearsonr_dab', data=df_corr)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('Pearson correlation', fontsize=12)
    plt.savefig('violin_pear_dab.png', dpi=600)

    plt.clf()
    sns.set(font_scale=1.2)
    sns.set_style("white")
    sns.set_palette("pastel")
    # sns.despine(offset=5, trim=True)
    boxplot = sns.violinplot(x='method', y='ssim_dab', data=df_corr)
    boxplot.set_xlabel('', fontsize=12)
    boxplot.set_ylabel('Structral similarity', fontsize=12)
    plt.savefig('violin_ssim_dab.png', dpi=600)

    df_corr.to_csv('correlations_ihc.csv',index=False)

if __name__ == '__main__':
	# extract patches from the DeepLIIF dataset
    extract_ROI(IMAGE_FOLDER, DEEPLIIF_LIST, PATCH_SIZE)
    # generate cfg files for each patch for deconvolution jobs on hpc. NOTE: shell code for submitting jobs to hpc is not included here
    generate_cfgs(cfg_example_file, cfg_input_folder, cfg_output_folder, IHC_FOLDER)
    # deconvolute DeepLIIF patches using implementations from https://github.com/wanghao14/Stain_Normalization
    separation_with_stain_vectors_M_V(MACENKO_FOLDER, VAHADANE_FOLDER, OURS_FOLDER, IHC_FOLDER)
    # extract results from PGDIPS outputs
    extract_results_pgdips(OUTPUT_FOLDER, OURS_FOLDER)
    # calculate metrics and generate plots
    calculate_correlations(MACENKO_FOLDER, VAHADANE_FOLDER, OURS_FOLDER)


