#packages
import os
import h5py
import shutil
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy import table
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, ImageNormalize)
import csv 
from astropy.table import Table
import pandas as pd
from sklearn.utils import shuffle
import umap
from sklearn.preprocessing import StandardScaler

#global variables
cutout_size = 64

#hd5 file
cutout_dir = os.path.expandvars("$SCRATCH") + "/"
hf = h5py.File(cutout_dir + "cutouts_filtered.h5", "r")

#tile list
with open(cutout_dir + 'tiles_5channel_41tiles.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
tile_ids = [i[0] for i in data]



#for first notebook: Create Filtered Cutouts.ipynb

def create_cutouts(img, wt, x, y, band):
    ''' Creates the image and weight cutouts given a tile, the position of the center and the band '''
    
    img_cutout = Cutout2D(img.data, (x, y), cutout_size, mode="partial", fill_value=0).data
    if np.count_nonzero(np.isnan(img_cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(img_cutout) == 0: # Don't use this cutout
        return (None, None)
    wt_cutout = Cutout2D(wt.data, (x, y), cutout_size, mode="partial", fill_value=0).data
    img_cutout[np.isnan(img_cutout)] = 0
    wt_cutout[np.isnan(wt_cutout)] = 0
    
    img_lower = np.percentile(img_cutout, 1)
    img_upper = np.percentile(img_cutout, 99)
    
    if img_lower == img_upper:
        img_norm = np.zeros((cutout_size, cutout_size))
    else:
        img_norm = (img_cutout - np.min(img_cutout)) / (img_upper - img_lower)
    if (band != "u" or band != "r") and img_upper != img_lower: # Alter weights for PS1
        wt_norm = (wt_cutout - np.min(wt_cutout)) / (img_upper - img_lower)
    else: # Do not alter weights for CFIS
        wt_norm = wt_cutout
        
    return (img_norm, wt_norm)

#for the second notebook: Multi-Channel Autoencoder
def get_test_cutouts(index, n_cutouts, bands="all", start=0):
    ''' Input: hf file, tile index, number of cutouts needed, dimensions, band and cutout start marker
        Output: the img and weight cutouts for the test set as (n_cutouts, pix, pix, channels) '''
    n = 0
    if bands == "all":
        band_indices = [0, 1, 2, 3, 4]                
    elif bands == "cfis":
        band_indices = [0,2]        
    elif bands == "ps1":
        band_indices = [1, 3, 4]
    else:
        print('WARNING: unrecognized band')
        band_indices = [0]
        
    l = len(band_indices)
    sources = np.zeros((n_cutouts, cutout_size, cutout_size, l))
    weights = np.zeros((n_cutouts, cutout_size, cutout_size, l))
        
    img_group = hf.get(tile_ids[index] + "/IMAGES")
    wt_group = hf.get(tile_ids[index] + "/WEIGHTS")
    for i in range(start, len(img_group)):
        sources[n,:,:,:] = np.array(img_group.get(f"c{i}"))[:,:,band_indices]
        weights[n,:,:,:] = np.array(wt_group.get(f"c{i}"))[:,:,band_indices]
        n += 1
        if n == n_cutouts:
            return (sources, weights)
        
def get_cutouts(tile_indices, batch_size, bands="all"):
    ''' Input: hf file, tile indices, batch size, dimensions, band and bands
        Output: the img and weight cutouts for the test set as (batch_size, pix, pix, channels) '''
    b = 0 # counter for batch
    if bands == "all":
        band_indices = [0, 1, 2, 3, 4]
    elif bands == "cfis":
        band_indices = [0, 2]
        l = len(band_indices)
    elif bands == "ps1":
        band_indices = [1, 3, 4]
    else:
        print('WARNING: unrecognized band')
        band_indices = [0]   
 
    l = len(band_indices)
    sources = np.zeros((batch_size, cutout_size, cutout_size, l))
    weights = np.zeros((batch_size, cutout_size, cutout_size, l))
    while True:
        for i in tile_indices:
            img_group = hf.get(tile_ids[i] + "/IMAGES")
            wt_group = hf.get(tile_ids[i] + "/WEIGHTS")
            n_cutouts = len(img_group)
            for n in range(n_cutouts):
                sources[b,:,:,:] = np.array(img_group.get(f"c{n}"))[:,:,band_indices]
                weights[b,:,:,:] = np.array(wt_group.get(f"c{n}"))[:,:,band_indices]
                b += 1
                if b == batch_size:
                    b = 0
                    yield (sources, sources)

def train_autoencoder(model, train_indices, val_indices, n_epochs, batch_size, bands="all"):
    '''Training autoencoders given the compiled model'''
    n_cutouts_train = 0
    for i in train_indices:
        img_group = hf.get(tile_ids[i] + "/IMAGES")        
        n_cutouts_train += len(img_group)
    
    n_cutouts_val = 0    
    for i in val_indices:
        img_group = hf.get(tile_ids[i] + "/IMAGES")        
        n_cutouts_val += len(img_group)
    
    train_steps = n_cutouts_train // batch_size
    val_steps = n_cutouts_val // batch_size
    
    history = model.fit(get_cutouts(train_indices, batch_size, bands), 
                        epochs=n_epochs, steps_per_epoch=train_steps, 
                        validation_data=get_cutouts(val_indices, batch_size, bands), 
                        validation_steps=val_steps)
    return model, history

#### PLOTTING ####
def plot_loss_curves(history, figname):
    '''Plots loss curves and saves it in ../Loss Curves subdir'''
    plt.plot(history["loss"], color="g", label="Training")
    plt.plot(history["val_loss"], color="b", label="Validation")
    plt.title("Loss Curves for Training/Validation Sets")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../Loss Curves/" + figname)
    
def plot_images(images, band):
    ''' plots 9 images for 1 band'''
    if len(bands)==5:
        fig, axes = plt.subplots(3,3, figsize=(8,8))
        i = 0
        for row in range(3):
            for col in range(3):
                norm = ImageNormalize(images[i][:,:,band], interval=ZScaleInterval())
                axes[row][col].imshow(images[i][:,:,band], norm=norm)
                i += 1
    #cfis
    if len(bands)==2:
            #plots source, reconstructed and residuals for 1 image (start = index) for all bands
        fig, axes = plt.subplots(images.shape[0],len(bands), figsize=(2,20))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.45)
        for row in range(images.shape[0]):
            for col in range(len(bands)):
                norm = ImageNormalize(images[row,:,:,col], interval=ZScaleInterval())
                im = axes[row][col].imshow(images[row,:,:,col], norm=norm)
                #fig.colorbar(im, fraction=0.045, ax=axes[row][col])
                if row == 0:
                    axes[row][col].set_title(bands[col])
        #plt.savefig("../Plots/" + figname)
    if len(bands)==3:
            #plots source, reconstructed and residuals for 1 image (start = index) for all bands
        fig, axes = plt.subplots(images.shape[0],len(bands), figsize=(4,50))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.45)
        for row in range(images.shape[0]):
            for col in range(len(bands)):
                norm = ImageNormalize(images[row,:,:,col], interval=ZScaleInterval())
                im = axes[row][col].imshow(images[row,:,:,col], norm=norm)
                #fig.colorbar(im, fraction=0.045, ax=axes[row][col])
                if row == 0:
                    axes[row][col].set_title(bands[col])
        #plt.savefig("../Plots/" + figname)
            
def plot_1_cutout(images, weights, figname, bands, start=0):
    '''plots source, reconstructed and residuals for 1 image (start = index) for all bands and weight maps'''
    fig, axes = plt.subplots(images.shape[0],len(bands), figsize=(14,8))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.45)
    for row in range(3):
        for col in range(len(bands)):
            norm = ImageNormalize(images[row, start,:,:,col], interval=ZScaleInterval())
            im = axes[row][col].imshow(images[row, start,:,:,col], norm=norm)
            fig.colorbar(im, fraction=0.045, ax=axes[row][col])
            if row == 0:
                axes[row][col].set_title(bands[col])
    plt.savefig("../Plots/" + figname)
    plt.show()
    
    #weights
    fig, axes = plt.subplots(1,5, figsize=(14,8))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.6)
    for i in range(5):
        norm = ImageNormalize(weights[start,:,:,i], interval=ZScaleInterval())
        im = axes[i].imshow(weights[start,:,:,i], norm=norm)
        fig.colorbar(im, fraction=0.045, ax=axes[i])
        axes[i].set_title(bands[i])
    #plt.savefig(f"../Plots/Weights 185.270 c{start} 64p.png")
    
def plot_mosaic(images, figname): 
    '''plotting 50 images, 5 bands'''
    fig, axes = plt.subplots(50,len(bands), figsize=(5,50))
    #fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.45)
    i=0
    for row in range(50):
        for col in range(len(bands)):
            norm = ImageNormalize(images[row,:,:,col], interval=ZScaleInterval())
            im = axes[row][col].imshow(images[row,:,:,col], norm=norm)
            #fig.colorbar(im, fraction=0.045, ax=axes[row][col])
            axes[row][col].set_yticks([]);axes[row][col].set_xticks([])
            if row == 0:
                axes[row][col].set_title(bands[col])
            i += 1
    plt.savefig("../Plots/" + figname)
    
def min_max_pixels(images, bands, start=0):
    for i in range(len(bands)):
        print(bands[i])
        print("Min pixel value: " + str(np.min(images[start,:,:,i])))
        print("Max pixel value: " + str(np.max(images[start,:,:,i])))
        
def plot_hist(images, wts, figname, bands, start=0):
    '''plots histogram of source, reconstructed and residuals for 1 image (start = index) for all bands '''
    if len(bands) == 5:
        fig, axes = plt.subplots(images.shape[0],len(bands), figsize=(20,8))
        for row in range(images.shape[0]):
            for col in range(len(bands)):
                mean = np.mean(images[row,start,:,:,col])
                std = np.std(images[row,start,:,:,col])
                if row == 2:
                    x = images[0,start,:,:,col]
                    xr = images[1,start,:,:,col]
                    axes[row][col].hist((np.sqrt(wts[start,:,:,col])*(x-xr)).ravel())
                else:
                    axes[row][col].hist(images[row,start,:,:,col].ravel())
                axes[row][col].set_ylim(top=4000)
                #xlim = axes[row][col].get_xlim()[1]
                #ylim = axes[row][col].get_ylim()[1]
                #axes[row][col].annotate(r"$\mu={:.4f}$".format(mean), (0.7*xlim, 0.7*ylim))
                #axes[row][col].annotate(r"$\sigma={:.4f}$".format(std), (0.7*xlim, 0.6*ylim))
                if row == 0:
                    axes[row][col].set_title(bands[col])
        plt.savefig("../Histograms/" + figname)
    
    if len(bands) == 2: 
        fig, axes = plt.subplots(images.shape[0],len(bands), figsize=(5,10))
        for row in range(images.shape[0]):
            for col in range(len(bands)):
                mean = np.mean(images[row,:,:,col])
                std = np.std(images[row,:,:,col])
                #if row == 2:
                 #   x = images[0,:,:,col]
                 #   xr = images[1,:,:,col]
                 #   axes[row][col].hist((np.sqrt(wts[:,:,col])*(x-xr)).ravel())
                #else:
                axes[row][col].hist(images[row,:,:,col].ravel())
                axes[row][col].set_ylim(top=4000)
                #xlim = axes[row][col].get_xlim()[1]
                #ylim = axes[row][col].get_ylim()[1]
                #axes[row][col].annotate(r"$\mu={:.4f}$".format(mean), (0.7*xlim, 0.7*ylim))
                #axes[row][col].annotate(r"$\sigma={:.4f}$".format(std), (0.7*xlim, 0.6*ylim))
                if row == 0:
                    axes[row][col].set_title(bands[col])
        plt.savefig("../Histograms/" + figname)
        
#For the third notebook: Lens binary Classifier

def get_confirmed_cutouts(hf_pos, filter_dict, label_dir, label_subdir):
    '''Cutouts '''
    n_cutouts = 0
    for k in list(hf_pos.get(label_subdir).keys()):
        f = list(hf_pos.get(label_subdir + k).keys())[0]
        img_subgroup = hf_pos.get(label_subdir + k + "/" + f + "/IMAGES")
        n_cutouts += len(img_subgroup)
       
    confirmed_cutouts = np.zeros((n_cutouts, cutout_size, cutout_size, 5))
    n_tiles = len(list(hf_pos.get(label_subdir).keys()))
    count = 0
    tile_ids = list(hf_pos.get(label_subdir).keys())
    for n in range(n_tiles):
        tile_id = tile_ids[n]
        f = list(hf_pos.get(label_subdir + tile_id).keys())[0]
        df = pd.read_csv(label_dir + label_subdir + f + "/" + tile_id + "_labels.csv")
        img_subgroup = hf_pos.get(label_subdir + tile_id + "/" + f + "/IMAGES")
        n_labels = len(img_subgroup)
        for i in range(n_labels):
            cutout = np.zeros((cutout_size, cutout_size, 5))
            dataset_name = tile_id + str(i)
            filts = [f + "/" for f in list(hf_pos.get(label_subdir + tile_id).keys())]
            filt_indices = [filter_dict.get(f) for f in filts]
            for (j, ind) in enumerate(filt_indices):
                cutout[:,:,ind] = hf_pos.get(label_subdir + tile_id + "/" + filts[j] + "IMAGES/" + dataset_name)
            confirmed_cutouts[count,:,:,:] = cutout
            count += 1
    return confirmed_cutouts

#only if negatie cutouts were not created
def create_cutout(fits_file, x, y):
    cutout = Cutout2D(fits_file[0].data, (x, y), cutout_size, mode="partial", fill_value=0).data
    if np.count_nonzero(np.isnan(cutout)) >= 0.05*cutout_size**2 or np.count_nonzero(cutout) == 0: # Don't use this cutout
        return None
    cutout[np.isnan(cutout)] = 0
    lower = np.percentile(cutout, 1)
    upper = np.percentile(cutout, 99)
    if lower == upper:
        cutout_norm = np.zeros((cutout_size, cutout_size))
    else:
        cutout_norm = (cutout - np.min(cutout)) / (upper - lower)
    return cutout_norm

def get_negative_cutouts():
    n_negative = 6370
    u_fits = fits.open(u_image, memmap=True)
    #g_fits = fits.open(g_image, memmap=True)
    r_fits = fits.open(r_image, memmap=True)
    #i_fits = fits.open(i_image, memmap=True)
    #z_fits = fits.open(z_image, memmap=True)
    cat = Table.read(dest + f"CFIS.{tile_id}.r.cat", format="ascii.sextractor")
    n = 0
    for i in range(len(cat)):
        cutout = np.zeros((cutout_size, cutout_size, 5))
        if cat["FLAGS"][i] != 0 or cat["MAG_AUTO"][i] >= 99.0 or cat["MAGERR_AUTO"][i] <= 0 or cat["MAGERR_AUTO"][i] >= 1:
            continue
        x = cat["X_IMAGE"][i]
        y = cat["Y_IMAGE"][i]
        
        u_cutout = create_cutout(u_fits, x, y)
        if u_cutout is None:
            continue
        #g_cutout = create_cutout(g_fits, x, y)
        #if g_cutout is None:
        #    continue
        r_cutout = create_cutout(r_fits, x, y)
        if r_cutout is None:
            continue
        #i_cutout = create_cutout(i_fits, x, y)
        #if i_cutout is None:
        #    continue
        #z_cutout = create_cutout(z_fits, x, y)
        #if z_cutout is None:
        #    continue
        cutout[:,:,0] = u_cutout
        #cutout[:,:,1] = g_cutout
        cutout[:,:,2] = r_cutout
        #cutout[:,:,3] = i_cutout
        #cutout[:,:,4] = z_cutout
        hf_neg.create_dataset(f"cutout{n}", data=cutout)
        n += 1
        if n == n_negative:
            u_fits.close()
            #g_fits.close()
            r_fits.close()
            #i_fits.close()
            #z_fits.close()
            return
def train_classifier(model, n_epochs, batch_size):
    num_cutouts_train_neg = int(0.7*len(hf_neg))
    neg_start_train = 0
    neg_end_train = num_cutouts_train_neg
    neg_start_val = num_cutouts_train_neg
    neg_end_val = int(0.9*len(hf_neg))

    num_cutouts_train_pos = int(0.7*len(confirmed_cutouts))
    pos_start_train = 0
    pos_end_train = num_cutouts_train_pos
    pos_start_val = num_cutouts_train_pos
    pos_end_val = int(0.9*len(confirmed_cutouts))

    train_steps = (neg_end_train + pos_end_train) // batch_size
    val_steps = ((neg_end_val - neg_start_val) + (pos_end_val - pos_start_val)) // batch_size
    neg_weight = (num_cutouts_train_neg + num_cutouts_train_pos) / num_cutouts_train_neg
    pos_weight = (num_cutouts_train_neg + num_cutouts_train_pos) / num_cutouts_train_pos
    class_weight = {0: neg_weight, 1: pos_weight}
    history = model.fit(get_cutouts(pos_start_train, pos_end_train, neg_start_train, neg_end_train, batch_size), 
                        epochs=n_epochs, steps_per_epoch=train_steps, 
                        validation_data=get_cutouts(pos_start_val, pos_end_val, neg_start_val, neg_end_val, batch_size), 
                        validation_steps=val_steps, callbacks=[callback], class_weight=class_weight)
    return model, history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    plt.hist(y_predict)
    print("Lowest 10 scores:")
    print(sorted(y_predict)[:10])
    print()
    print("Highest 10 scores:")
    print()
    print(sorted(y_predict)[-10:])
    conf = tf.math.confusion_matrix(y_test, y_predict)
    print(f"Confusion Matrix:\n {conf}")
    print("Test loss: %.3f" % test_loss)
    print("Test accuracy: %3f" % test_acc)