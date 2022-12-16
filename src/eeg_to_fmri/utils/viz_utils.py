import tensorflow.compat.v1 as tf

import numpy as np

from eeg_to_fmri.metrics import bnn, quantitative_metrics

from eeg_to_fmri.learning import losses

from eeg_to_fmri.data import eeg_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from scipy import spatial

from numpy.linalg import norm

from nilearn import plotting, image

from scipy.ndimage import rotate

from pathlib import Path

import pathlib

from scipy.stats import gamma, ttest_ind, ttest_1samp

import copy

uncertainty_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/uncertainty/"
uncertainty_losses_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/uncertainty_losses/"
gamma_plots_directory = str(Path.home())+"/eeg_to_fmri/src/results_plots/gamma/"

def get_models_and_shapes(eeg_file='../../optimized_nets/eeg/eeg_30_partitions.json', 
                        bold_file='../../optimized_nets/bold/bold_30_partitions.json',
                        decoder_file='../../optimized_nets/decoder/decoder_30_partitions.json'):

    json_file = open(eeg_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    eeg_network = tf.keras.models.model_from_json(loaded_model_json)

    json_file = open(bold_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    bold_network = tf.keras.models.model_from_json(loaded_model_json)

    json_file = open(decoder_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    decoder_network = tf.keras.models.model_from_json(loaded_model_json)

    return eeg_network, bold_network, decoder_network


def _plot_mean_std(reconstruction_loss, distance, tset="train", n_partitions=30, model="M", ax=None):

    inds_ids = []
    inds_mean = np.zeros(len(reconstruction_loss)//n_partitions)
    inds_std = np.zeros(len(reconstruction_loss)//n_partitions)

    #compute mean 
    for ind in range(inds_mean.shape[0]):
        inds_ids += ['Ind_' + str(ind+1)]
        inds_mean[ind] = np.mean(reconstruction_loss[ind:ind+n_partitions])
        inds_std[ind] = np.std(reconstruction_loss[ind:ind+n_partitions])

    print(tset + " set", "mean: ", np.mean(reconstruction_loss))
    print(tset + " set", "std: ", np.std(reconstruction_loss))


    ax.errorbar(inds_ids, inds_mean, inds_std, linestyle='None', elinewidth=0.5, ecolor='r', capsize=10.0, markersize=10.0, marker='o')
    ax.set_title(distance + " on " + tset + " set " + " (" + model + ")")
    ax.set_xlabel("Individuals")
    if("Cosine" in distance):
        ax.set_ylabel("Correlation")
    else:
        ax.set_ylabel("Distance")

def _plot_mean_std_loss(synthesized_bold, bold, distance_function, distance_name, set_name, model_name, n_partitions=30, ax=None):
    reconstruction_loss = np.zeros((synthesized_bold.shape[0], 1))

    for instance in range(len(reconstruction_loss)):
        instance_synth = synthesized_bold[instance]
        instance_bold = bold[instance]

        instance_synth = instance_synth.reshape((1, instance_synth.shape[0], instance_synth.shape[1], instance_synth.shape[2]))
        instance_bold = instance_bold.reshape((1, instance_bold.shape[0], instance_bold.shape[1], instance_bold.shape[2]))

        reconstruction_loss[instance] = distance_function(instance_synth, instance_bold).numpy()

    _plot_mean_std(reconstruction_loss, distance=distance_name, tset=set_name, model=model_name, n_partitions=n_partitions, ax=ax)



def plot_mean_std_loss(eeg_train, bold_train, 
                        eeg_val, bold_val, 
                        eeg_test, bold_test, 
                        encoder_network, decoder_network, 
                        distance_name, distance_function,
                        model_name, n_partitions=30):

    n_plotted = 1

    n_plots = int(type(eeg_train) is np.ndarray and type(bold_train) is np.ndarray) + \
        int(type(eeg_val) is np.ndarray and type(bold_val) is np.ndarray) + \
        int(type(eeg_test) is np.ndarray and type(bold_test) is np.ndarray)

    plt.figure(figsize=(20,5))

    if(type(eeg_train) is np.ndarray and type(bold_train) is np.ndarray):
        ax1 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_train = encoder_network.predict(eeg_train)
        synthesized_bold_train = decoder_network.predict(shared_eeg_train)
        _plot_mean_std_loss(synthesized_bold_train, bold_train, distance_function, distance_name, "train", model_name, n_partitions=n_partitions, ax=ax1)

    if(type(eeg_val) is np.ndarray and type(bold_val) is np.ndarray):
        ax2 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_val = encoder_network.predict(eeg_val)
        synthesized_bold_val = decoder_network.predict(shared_eeg_val)
        _plot_mean_std_loss(synthesized_bold_val, bold_val, distance_function, distance_name, "validation", model_name, n_partitions=n_partitions, ax=ax2)

    

    if(type(eeg_test) is np.ndarray and type(bold_test) is np.ndarray):
        ax3 = plt.subplot(1,n_plots,n_plotted)
        n_plotted += 1

        shared_eeg_test = encoder_network.predict(eeg_test)
        synthesized_bold_test = decoder_network.predict(shared_eeg_test)
        _plot_mean_std_loss(synthesized_bold_test, bold_test, distance_function, distance_name, "test", model_name, n_partitions=n_partitions, ax=ax3)

    plt.show()

def plot_loss_results(eeg_train, bold_train, eeg_val, bold_val, eeg_test, bold_test, eeg_network, decoder_network, model_name, n_partitions=30):

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Log Cosine", losses.get_reconstruction_log_cosine_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Log Cosine Voxels Mean", losses.get_reconstruction_log_cosine_voxel_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Cosine", losses.get_reconstruction_cosine_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Cosine Voxels Mean", losses.get_reconstruction_cosine_voxel_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Euclidean", losses.get_reconstruction_euclidean_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Euclidean Per Volume", losses.get_reconstruction_euclidean_volume_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "Mean Absolute Error Per Volume", losses.get_reconstruction_absolute_volume_loss,
    model_name, n_partitions=n_partitions)

    plot_mean_std_loss(eeg_train, bold_train, 
    eeg_val, bold_val, 
    eeg_test, bold_test, 
    eeg_network, decoder_network, 
    "KL Loss", losses.get_reconstruction_kl_loss,
    model_name, n_partitions=n_partitions)


######################################################################################################################################################
#
#                                                            PLOT VOXELS REAL AND SYNTHESIZED
#
######################################################################################################################################################


def plot_view_mask(img, timestep=4, vmin=None, vmax=None, resampling_factor=4, symmetric_cmap=False, save_file="/tmp/plot.html"):
    img = image.index_img(img, timestep)

    if(vmin is None):
        vmin=np.amin(img.get_data())
    if(vmax is None):
        vmax=np.amax(img.get_data())

    view = plotting.view_img(img, 
                            threshold=None,
                            colorbar=True,
                            annotate=False,
                            draw_cross=False,
                            cut_coords=[0, 0,  0],
                            black_bg=True,
                            bg_img=False,
                            cmap="inferno",
                            symmetric_cmap=symmetric_cmap,
                            vmax=vmax,
                            vmin=vmin,
                            dim=-2,
                            resampling_interpolation="nearest")

    view.save_as_html(save_file)


def _plot_voxel(real_signal, synth_signal, rows=1, columns=2, index=1, y_bottom=None, y_top=None):
    ax = plt.subplot(rows, columns, index)
    ax.plot(list(range(0, len(real_signal)*2, 2)), real_signal, color='b')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    if(y_bottom==None and y_top==None):
        y_bottom_real = np.amin(real_signal)
        y_top_real = np.amax(real_signal)
        y_bottom_synth = np.amin(synth_signal)
        y_top_synth = np.amax(synth_signal)
        
    ax.set_ylim(y_bottom_real, y_top_real)
    
    if(index == 1):
        ax.set_title("Real BOLD Signal", y=0.99999)

        
    
    ax = plt.subplot(rows, columns, index+1)
    ax.plot(list(range(0, len(synth_signal)*2, 2)), synth_signal, color='r')
    ax.set_xlabel("Seconds")
    ax.set_ylabel("BOLD intensity")
    
    ax.set_ylim(y_bottom_synth, y_top_synth)
    
    if(index == 1):
        ax.set_title("Synthesized BOLD Signal")

def _plot_voxels(real_set, synth_set, individual=0, voxels=None, y_bottom=None, y_top=None, title_pos=0.999, pad=0.1, normalized=False):
    n_voxels=len(voxels)
    fig = plt.figure(figsize=(20,n_voxels*2))

    fig.suptitle('Top-' + str(len(voxels)) + ' correlated voxels', fontsize=16, y=title_pos)

    if(individual != None):
        real_set = real_set[individual] 
        synth_set = synth_set[individual]

    index=1
    if(voxels):
        for voxel in voxels:
            real_voxel = real_set[voxel]
            synth_voxel = synth_set[voxel]

            if(normalized):
                real_voxel = real_voxel/norm(real_voxel)
                synth_voxel = synth_voxel/norm(synth_voxel)


            _plot_voxel(real_voxel, synth_voxel, 
                        rows=n_voxels, index=index, 
                        y_bottom=y_bottom, y_top=y_top)
            index += 2

    fig.tight_layout(pad=pad)

    plt.show()

def rank_best_synthesized_voxels(real_signal, synth_signal, top_k=10, ignore_static=True, verbose=0):
    sort_voxels = {}
    n_voxels = real_signal.shape[0]
    
    for voxel in range(n_voxels):
        #ignore voxels that are constant over time
        if(ignore_static and all(x==real_signal[voxel][0] for x in real_signal[voxel])):
            continue
        voxel_a = real_signal[voxel].reshape((real_signal[voxel].shape[0]))
        voxel_b = synth_signal[voxel].reshape((synth_signal[voxel].shape[0]))
        distance_cosine = spatial.distance.cosine(voxel_a/norm(voxel_a), voxel_b/norm(voxel_b))
        if(verbose>1):
            print("Distance:", distance_cosine)

        sort_voxels[voxel] = distance_cosine

    sort_voxels = dict(sorted(sort_voxels.items(), key=lambda kv: kv[1]))
    
    if(verbose>0):
        print(list(sort_voxels.values())[0:top_k])

    return list(sort_voxels.keys())[0:top_k]


##########################################################################################################
#
#                                                HEATMAP
#
##########################################################################################################

def heat_map(real_bold_set, synth_bold_set, individual=8, timestep=0, normalize=False):
    real_mapping = np.copy(real_bold_set[individual][:, timestep, 0])
    synth_mapping = np.copy(synth_bold_set[individual][:, timestep, 0])
    
    if(normalize):
        for voxel in range(len(real_bold_set[individual])):
            real_bold_set[individual][voxel] = real_bold_set[individual][voxel]/norm(real_bold_set[individual][voxel])            
            synth_bold_set[individual][voxel] = synth_bold_set[individual][voxel]/norm(synth_bold_set[individual][voxel])
                
    real_mapping.resize((50, 52))
    synth_mapping.resize((50, 52))
    
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    
    ax1.imshow(real_mapping, cmap='gnuplot2', interpolation='nearest')#, cmap="YlGnBu")
    
    ax1.set_xticks([], [])
    ax1.set_yticks([],[])
    ax1.set_title('Real Bold Signal')
    
    ax2.imshow(synth_mapping, cmap='gnuplot2', interpolation='nearest')#, cmap="YlGnBu")
    
    ax2.set_xticks([], [])
    ax2.set_yticks([],[])
    ax2.set_title('Synthesized Bold Signal')
    
    plt.show()

def plot_epistemic_aleatoric_uncertainty(setting, model, array_set, volume, xslice, yslice, zslice, T=10):
    save_file=uncertainty_plots_directory+setting+"/"+"v_"+ str(volume)+"_x_"+str(xslice)+"_y_"+str(yslice)+"_z_"+str(zslice)+".pdf"

    pathlib.Path(uncertainty_plots_directory+setting).mkdir(parents=True, exist_ok=True) 

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(15,6))
    axes[0][0].imshow(rotate(array_set[volume,xslice,:,:,:], 90),cmap=plt.cm.nipy_spectral)
    axes[0][0].set_xticks([])
    axes[0][0].set_yticks([])
    axes[0][1].imshow(rotate(model(array_set[volume:volume+1])[0].numpy()[0,xslice,:,:,:], 90, axes=(0,1)),cmap=plt.cm.nipy_spectral)
    axes[0][1].set_xticks([])
    axes[0][1].set_yticks([])
    axes[0][2].imshow(rotate(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,xslice,:,:,:], 90, axes=(0,1)))
    axes[0][2].set_xticks([])
    axes[0][2].set_yticks([])
    axes[0][3].imshow(rotate(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,xslice,:,:,:], 90, axes=(0,1)))
    axes[0][3].set_xticks([])
    axes[0][3].set_yticks([])

    axes[1][0].imshow(rotate(array_set[volume,:,yslice,:,:], 90),cmap=plt.cm.nipy_spectral)
    axes[1][0].set_xticks([])
    axes[1][0].set_yticks([])
    axes[1][1].imshow(rotate(model(array_set[volume:volume+1])[0].numpy()[0,:,yslice,:,:], 90, axes=(0,1)),cmap=plt.cm.nipy_spectral)
    axes[1][1].set_xticks([])
    axes[1][1].set_yticks([])
    axes[1][2].imshow(rotate(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,yslice,:,:], 90, axes=(0,1)))
    axes[1][2].set_xticks([])
    axes[1][2].set_yticks([])
    axes[1][3].imshow(rotate(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,yslice,:,:], 90, axes=(0,1)))
    axes[1][3].set_xticks([])
    axes[1][3].set_yticks([])

    axes[2][0].imshow(array_set[volume,:,:,zslice,:],cmap=plt.cm.nipy_spectral, aspect="auto")
    axes[2][0].set_xticks([])
    axes[2][0].set_yticks([])
    axes[2][1].imshow(model(array_set[volume:volume+1])[0].numpy()[0,:,:,zslice,:],cmap=plt.cm.nipy_spectral, aspect="auto")
    axes[2][1].set_xticks([])
    axes[2][1].set_yticks([])
    axes[2][2].imshow(bnn_utils.aleatoric_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,:,zslice,:], aspect="auto")
    axes[2][2].set_xticks([])
    axes[2][2].set_yticks([])
    axes[2][3].imshow(bnn_utils.epistemic_uncertainty(model, array_set[volume:volume+1], T=T).numpy()[0,:,:,zslice,:], aspect="auto")
    axes[2][3].set_xticks([])
    axes[2][3].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_file, format="pdf")


def gamma_epoch_plot(setting, parameters_history, epochs=10):
    save_file=gamma_plots_directory+setting+"/"
    pathlib.Path(save_file).mkdir(parents=True, exist_ok=True) 

    plt.style.use("default")

    for epoch in range(epochs):
        plt.figure()
        distribution = gamma.pdf(np.linspace (0, 100, 200), 
                                  a=np.abs(parameters_history[epoch][0])+1e-9, 
                                  scale=1/(np.abs(parameters_history[epoch][1])+1e-9))
        plt.plot(np.linspace (0, 1, 200), distribution)
        plt.title("$\\lambda = \\frac{1}{\\sigma^2} \\sim $Ga($\\alpha$,$\\beta$)", size=12)
        plt.ylabel("$f_{\\lambda} = f_{\\frac{1}{\\sigma^2}}$", size=15)
        plt.rc('font', family='serif')
        plt.tight_layout()
        
        plt.savefig(save_file+"epoch_"+str(epoch)+".pdf", format="pdf")


def uncertainty_losses_plot(setting, losses_history, loss_i, epochs=10):
    save_file=uncertainty_losses_plots_directory+setting+"/"
    pathlib.Path(save_file).mkdir(parents=True, exist_ok=True) 

    plt.style.use("default")

    plt.figure()
    fig, axes = plt.subplots(1,4,figsize=(30,6))

    axes[0].plot(losses_history[:,0], color="black")
    axes[0].grid(True)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("$\mathcal{L}_"+str(loss_i)+"$", size=18)
    axes[0].set_xticks(range(0,epochs))
    axes[0].set_xticklabels(range(1,epochs+1))
    axes[1].plot(losses_history[:,1], color="blue")
    axes[1].grid(True)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("$\mathcal{L}_"+str(2)+"$", size=18)
    axes[1].set_xticks(range(0,epochs))
    axes[1].set_xticklabels(range(1,epochs+1))
    axes[2].plot(losses_history[:,2], color="orange")
    axes[2].grid(True)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("$\mathcal{L}_"+str(1)+"$", size=18)
    axes[2].set_xticks(range(0,epochs))
    axes[2].set_xticklabels(range(1,epochs+1))
    axes[3].plot(losses_history[:,3], color="green")
    axes[3].grid(True)
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("$||y_i-f^{W}(x_i)||_2^2$", size=18)
    axes[3].set_xticks(range(0,epochs))
    axes[3].set_xticklabels(range(1,epochs+1))

    plt.rc('font', family='serif', size=15)
    plt.tight_layout()
    plt.savefig(save_file+"losses_convergence_val.pdf", format="pdf")



##########################################################################################################
#
#                                                3 Dimensinoal plot with 2d slices
#
##########################################################################################################

def plot_3D_representation_projected_slices(instance, factor=3, h_resolution=1, v_resolution=1, threshold=0.37, cmap=plt.cm.nipy_spectral, uncertainty=False, res_img=None, legend_colorbar="redidues", max_min_legend=["Good","Bad"], normalize_residues=False, normalize_explanations=False, slice_label=True, save=False, save_path=None, save_format="pdf"):
    """
        
    Inputs:
        instance: Numpy.ndarray - of shape (X,Y,Z,1)
        factor: float - that resamples the Z axis slices
        h_resolution: int - resolution in the horizontal dimension
        v_resolution: int - resolution in the vertical dimension
        save: bool - whether to save the figure
        save_path: str - path to save the figure, save has to be True
    Returns:
        matplotlib.Figure - The figure to plot, no saving option implemented
    """
    label = "$Z_{"

    #this is a placeholder for the residues plot
    if(res_img is None):
        res_img=instance
        res_img_none=False
    else:
        res_img_none=True
    
    fig = plt.figure(figsize=(25,7))
    gs = GridSpec(2, 7, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    axes = fig.add_subplot(gs[:,0:2], projection='3d', proj_type='ortho')
    
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_over("w")
    
    x, y = np.mgrid[0:instance[:,:,0].shape[0], 0:instance[:,:,0].shape[1]]
    
    #normalization
    if(not normalize_residues):
        instance = (instance[:,:,:,:]-np.amin(instance[:,:,:,:]))/(np.amax(instance[:,:,:,:])-np.amin(instance[:,:,:,:]))
        res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
        if(uncertainty):
            _instance=copy.deepcopy(instance)
        instance[np.where(res_img < threshold)]= 1.001
    elif(normalize_explanations):
        res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
        if(uncertainty):
            _instance=copy.deepcopy(instance)
        instance[np.where(res_img < threshold)]= 1.001
        instance[np.where(res_img >= threshold)] = (instance[np.where(res_img >= threshold)]-np.mean(instance[np.where(res_img >= threshold)]))/np.std(instance[np.where(res_img >= threshold)])
        instance[np.where(res_img >= threshold)] = (instance[np.where(res_img >= threshold)]-np.amin(instance[np.where(res_img >= threshold)]))/(np.amax(instance[np.where(res_img >= threshold)])-np.amin(instance[np.where(res_img >= threshold)]))
    else:
        res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
        if(uncertainty):
            _instance=copy.deepcopy(instance)
        instance[np.where(res_img < threshold)]= 1.001

    for axis in range((instance[:,:,:].shape[2])//factor):
        img = rotate(instance[:,:,axis*factor,0], 90)

        ax = axes.plot_surface(x,y,np.ones(x.shape)+5*(axis),
                                facecolors=cmap(img), cmap=cmap, 
                                shade=False, antialiased=True, zorder=0,
                                cstride=v_resolution, rstride=h_resolution)
        if(slice_label):
            axes.text(60, 60, 3+5*(axis), label+str(axis*factor)+"}$", size=13, zorder=100)

    if(uncertainty):
        instance=_instance

    #axes.text(60, 60, 20+5*(instance[:,:,:].shape[2])//factor, "3-Dimensional sliced representation", size=13, zorder=100)
    if(slice_label):
        pos=[0.13, 0.15, 0.01, 0.7]
    elif(not slice_label and res_img_none):
        pos=[0.12, 0.15, 0.01, 0.7]
    else:
        pos=[0.09, 0.15, 0.01, 0.7]
    cbaxes = fig.add_axes(pos)  # This is the position for the colorbar
    #to give a normalized bar that goes from 0.0 to 1.0
    if(normalize_residues or normalize_explanations):
        norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cb = plt.colorbar(ax, cax = cbaxes, norm=norm)
    else:
        cb = plt.colorbar(ax, cax = cbaxes)

    if(not slice_label):
        text_legend = fig.add_axes([pos[0]-0.007]+pos[1:])
        text_legend.axis("off")
        text_legend.text(-0.25,0.45, legend_colorbar, size=17, rotation=90)  # This is the position for the colorbar
        cb.ax.get_yaxis().set_ticks([0,1])
        cb.ax.get_yaxis().set_ticklabels(max_min_legend, size=17)


    axes.axis("off")
    axes.view_init(elev=5.5, azim=7)#(100,150)
    axes.dist = 5
    
    #plot each slice in 2 Dimensional plot
    row = 1
    col = 6
    for axis in range((instance[:,:,:].shape[2])//factor):
        axes = fig.add_subplot(gs[row,col])
        
        img = rotate(instance[:,:,axis*factor,0], 90)
        
        axes.imshow(cmap(img))
        if(slice_label):
            axes.text(28, 1, label+str(axis*factor)+"}$", size=13,
                 va="baseline", ha="left", multialignment="left",)

        axes.axis("off")

        col -= 1
        if(col == 1):
            col=6
            row-=1

    plt.rcParams["font.family"] = "serif"

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig


def plot_3D_representation_projected_slices_alpha(instance, factor=3, h_resolution=1, v_resolution=1, threshold=0.37, cmap=plt.cm.nipy_spectral, cmap_background=plt.cm.binary, uncertainty=False, res_img=None, alpha_img=None, legend_colorbar="redidues", max_min_legend=["Good","Bad"], normalize_residues=False, slice_label=True, save=False, save_path=None, save_format="pdf"):
    """
        
    Inputs:
        instance: Numpy.ndarray - of shape (X,Y,Z,1)
        factor: float - that resamples the Z axis slices
        h_resolution: int - resolution in the horizontal dimension
        v_resolution: int - resolution in the vertical dimension
        save: bool - whether to save the figure
        save_path: str - path to save the figure, save has to be True
    Returns:
        matplotlib.Figure - The figure to plot, no saving option implemented
    """
    label = "$Z_{"

    assert alpha_img is not None

    #this is a placeholder for the residues plot
    if(res_img is None):
        res_img=instance
        res_img_none=False
    else:
        res_img_none=True
    
    fig = plt.figure(figsize=(25,7))
    gs = GridSpec(2, 7, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    axes = fig.add_subplot(gs[:,0:2], projection='3d', proj_type='ortho')
    
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap_background = copy.copy(mpl.cm.get_cmap(cmap_background))
    cmap.set_over("w")
    cmap_background.set_over("w")
    #cmap.set_under("w")
    #cmap.set_bad("w")
    
    x, y = np.mgrid[0:instance[:,:,0].shape[0], 0:instance[:,:,0].shape[1]]
    
    #normalization
    if(not normalize_residues):
        instance = (instance[:,:,:,:]-np.amin(instance[:,:,:,:]))/(np.amax(instance[:,:,:,:])-np.amin(instance[:,:,:,:]))
        res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
        if(uncertainty):
            _instance=copy.deepcopy(instance)
        instance[np.where(res_img < threshold)]= 1.001
    else:
        res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
        if(uncertainty):
            _instance=copy.deepcopy(instance)
        instance[np.where(res_img < threshold)]= 1.001

    for axis in range((instance[:,:,:].shape[2])//factor):
        img = rotate(instance[:,:,axis*factor,0], 90)

        ax = axes.plot_surface(x,y,np.ones(x.shape)+5*(axis),
                                facecolors=cmap(img), cmap=cmap, 
                                shade=False, antialiased=True, zorder=0,
                                cstride=v_resolution, rstride=h_resolution)
        if(slice_label):
            axes.text(60, 60, 3+5*(axis), label+str(axis*factor)+"}$", size=13, zorder=100)

    if(uncertainty):
        instance=_instance

    #axes.text(60, 60, 20+5*(instance[:,:,:].shape[2])//factor, "3-Dimensional sliced representation", size=13, zorder=100)
    if(slice_label):
        pos=[0.13, 0.15, 0.01, 0.7]
    elif(not slice_label and res_img_none):
        pos=[0.12, 0.15, 0.01, 0.7]
    else:
        pos=[0.09, 0.15, 0.01, 0.7]
    cbaxes = fig.add_axes(pos)  # This is the position for the colorbar
    #to give a normalized bar that goes from 0.0 to 1.0
    if(normalize_residues):
        norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cb = plt.colorbar(ax, cax = cbaxes, norm=norm)
    else:
        cb = plt.colorbar(ax, cax = cbaxes)

    if(not slice_label):
        text_legend = fig.add_axes([pos[0]-0.007]+pos[1:])
        text_legend.axis("off")
        text_legend.text(-0.25,0.45, legend_colorbar, size=17, rotation=90)  # This is the position for the colorbar
        cb.ax.get_yaxis().set_ticks([0,1])
        cb.ax.get_yaxis().set_ticklabels(max_min_legend, size=17)


    axes.axis("off")
    axes.view_init(elev=5.5, azim=7)#(100,150)
    axes.dist = 5
    
    #plot each slice in 2 Dimensional plot
    row = 1
    col = 6
    for axis in range((instance[:,:,:].shape[2])//factor):
        axes = fig.add_subplot(gs[row,col])
        img = rotate(instance[:,:,axis*factor,0], 90)
        mask = rotate(alpha_img[:,:,axis*factor,0], 90)

        #alpha=(mask>0.5).astype("float32")
        #alpha+=(mask<0.1).astype("float32")
        #alpha[np.where(alpha==0.0)]=0.2

        axes.imshow(cmap_background(img))
        axes.imshow(cmap(img, alpha=mask))

        if(slice_label):
            axes.text(28, 1, label+str(axis*factor)+"}$", size=13,
                 va="baseline", ha="left", multialignment="left",)

        axes.axis("off")

        col -= 1
        if(col == 1):
            col=6
            row-=1

    plt.rcParams["font.family"] = "serif"

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig



def comparison_plot_3D_representation_projected_slices(res1, res2, pvalues, res_img, model1="Model1", model2="Model2", factor=3, h_resolution=1, v_resolution=1, threshold=0.37, slice_label=True, save=False, save_path=None, red_blue=False, save_format="pdf"):
    """
        
    Inputs:
        res1: Numpy.ndarray - of shape (X,Y,Z,1)
        res2: Numpy.ndarray - of shape (X,Y,Z,1)
        pvalues: Numpy.ndarray - of shape (X,Y,Z,1)
        res_img: Numpy.ndarray - of shape (X,Y,Z,1)
        factor: float - that resamples the Z axis slices
        h_resolution: int - resolution in the horizontal dimension
        v_resolution: int - resolution in the vertical dimension
        save: bool - whether to save the figure
        save_path: str - path to save the figure, save has to be True
    Returns:
        matplotlib.Figure - The figure to plot, no saving option implemented
    """
    label = "$Z_{"

    #colormap definition
    if(red_blue):
        cp1 = np.linspace(0,1)
        cp2 = np.linspace(0,1)
        Cp1, Cp2 = np.meshgrid(cp1,cp2)
        C0 = np.full_like(Cp1, Cp1*Cp2*((Cp1)+(Cp2))/2)
        p_values_range=Cp2#place holder that can stay to emulate pvalues
        Legend = np.dstack((np.concatenate((Cp2, p_values_range*Cp1[:,::-1], ), axis=1)[:,::-1],
                            np.concatenate((C0,C0[:,::-1]), axis=1),
                            np.concatenate((p_values_range*Cp1, Cp2), axis=1)[:,::-1]))
        cmap=ListedColormap(Legend)
    else:
        cp1 = np.linspace(0,1)
        cp2 = np.linspace(0,1)
        Cp1, Cp2 = np.meshgrid(cp1,cp2)
        C0 = np.full_like(Cp1, Cp1*Cp2*((Cp1)+(Cp2))/2)
        Cp2_ = np.triu(Cp2)
        np.fill_diagonal(Cp2_, 0)
        p_values_range=Cp2#place holder that can stay to emulate pvalues
        Legend = np.dstack((np.concatenate((p_values_range*Cp1, Cp2), axis=1)[:,::-1],
                            np.concatenate((Cp2, p_values_range*Cp1[:,::-1], ), axis=1)[:,::-1],
                            np.concatenate((Cp2, Cp2), axis=1)[:,::-1]))
        cmap=ListedColormap(Legend)

    #normalization
    res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
    res_img[np.where(res_img < threshold)]= -1


    #assign colors
    instance =np.zeros(res1[:,:,:,0].shape+(3,))
    res1=np.abs(res1)+1e-9
    res2=np.abs(res2)+1e-9
    res1[np.where(res1>1.0)]=0.9999#1.0
    res2[np.where(res2>1.0)]=0.9999#1.0

    def _cmap_(res1, res2, voxel, pvalue=0.0, red_blue=False):
        if(voxel==-1):
            return (0.999,0.999,0.999)
        elif(pvalue>0.05):
            pvalue=np.clip(pvalue, a_min=1e-9, a_max=0.2)
            return (pvalue,pvalue,pvalue)
        if(red_blue):
            if(res1<res2):
                return (1+1e-30-res1, 0.0001, pvalue+1e-9)
            return (pvalue+1e-9, 0.0001, 1+1e-30-res2)
        else:
            if(res1<res2):
                return (0.0001, 1+1e-30-res1, 1+1e-30-res1)
            return (1+1e-30-res2, 0.0001, 1+1e-30-res2)

    for voxel1 in range(instance.shape[0]):
        for voxel2 in range(instance.shape[1]):
            for voxel3 in range(instance.shape[2]):
                instance[voxel1,voxel2,voxel3] = np.array(list(_cmap_(res1[voxel1,voxel2,voxel3,0], res2[voxel1,voxel2,voxel3,0], 
                                                                res_img[voxel1,voxel2,voxel3,0],
                                                                pvalue=pvalues[voxel1,voxel2,voxel3,0],
                                                                red_blue=red_blue)))

    fig = plt.figure(figsize=(25,17))
    gs = GridSpec(41, 7, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    axes = fig.add_subplot(gs[:41,0:2], projection='3d', proj_type='ortho')
    x, y = np.mgrid[0:res_img[:,:,0].shape[0], 0:res_img[:,:,0].shape[1]]

    for axis in range((instance[:,:,:].shape[2])//factor):
        img = rotate(instance[:,:,axis*factor,:], 90)

        ax = axes.plot_surface(x,y,np.ones(x.shape)+5*(axis),
                                facecolors=img,
                                shade=False, antialiased=True, zorder=0,
                                cstride=v_resolution, rstride=h_resolution)
        if(slice_label):
            axes.text(60, 60, 3+5*(axis), label+str(axis*factor)+"}$", size=13, zorder=100)
    axes.axis("off")
    axes.view_init(elev=5.5, azim=7)#(100,150)
    axes.dist = 5

    #colorbar
    cax = fig.add_subplot(gs[30:33,3:5])
    cax.imshow(Legend, extent=[0,100,0,100], aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([5,95])
    cax.annotate('', xy=(0, -0.5), xycoords='axes fraction', xytext=(1, -0.5), 
                arrowprops=dict(arrowstyle="<->, head_width=0.4", color='black'))
    cax.annotate(model1, xy=(0, -0.56), xycoords='axes fraction', xytext=(-0.1, -0.56), size=20, color="red")
    cax.annotate(model2, xy=(0, -0.56), xycoords='axes fraction', xytext=(1.03, -0.56), size=20, color="blue")
    cax.set_yticklabels([r"$p=0.0$",r"$p=1.0$"], size=20)
    

    #plot each slice in 2 Dimensional plot
    row = 1
    col = 6
    for axis in range((instance[:,:,:].shape[2])//factor):
        if(row==1):
            _row=20
        else:
            _row=10
        axes = fig.add_subplot(gs[_row:_row+10,col])
        img = rotate(instance[:,:,axis*factor,:], 90)
        axes.imshow(img)
        if(slice_label):
            axes.text(28, 1, label+str(axis*factor)+"}$", size=13,
                 va="baseline", ha="left", multialignment="left",)
        axes.axis("off")
        col -= 1
        if(col == 1):
            col=6
            row-=1

    plt.rcParams["font.family"] = "serif"
    fig.set_tight_layout(True)

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig



def comparison_plot_uncertainty(res1, res2, pvalues, res_img, model1="Model1", model2="Model2", factor=3, h_resolution=1, v_resolution=1, threshold=0.37, slice_label=True, save=False, save_path=None, red_blue=False, save_format="pdf"):
    """
        
    Inputs:
        res1: Numpy.ndarray - of shape (X,Y,Z,1)
        res2: Numpy.ndarray - of shape (X,Y,Z,1)
        pvalues: Numpy.ndarray - of shape (X,Y,Z,1)
        res_img: Numpy.ndarray - of shape (X,Y,Z,1)
        factor: float - that resamples the Z axis slices
        h_resolution: int - resolution in the horizontal dimension
        v_resolution: int - resolution in the vertical dimension
        save: bool - whether to save the figure
        save_path: str - path to save the figure, save has to be True
    Returns:
        matplotlib.Figure - The figure to plot, no saving option implemented
    """
    label = "$Z_{"

    #colormap definition
    if(red_blue):
        cp1 = np.linspace(0,1)
        cp2 = np.linspace(0,1)
        Cp1, Cp2 = np.meshgrid(cp1,cp2)
        C0 = np.full_like(Cp1, Cp1*Cp2*((Cp1)+(Cp2))/2)
        p_values_range=Cp2#place holder that can stay to emulate pvalues
        Legend = np.dstack((np.concatenate((Cp2, p_values_range*Cp1[:,::-1], ), axis=1)[:,::-1],
                            np.concatenate((C0,C0[:,::-1]), axis=1),
                            np.concatenate((p_values_range*Cp1, Cp2), axis=1)[:,::-1]))
        cmap=ListedColormap(Legend)
    else:
        cp1 = np.linspace(0,1)
        cp2 = np.linspace(0,1)
        Cp1, Cp2 = np.meshgrid(cp1,cp2)
        C0 = np.full_like(Cp1, Cp1*Cp2*((Cp1)+(Cp2))/2)
        Cp2_ = np.triu(Cp2)
        np.fill_diagonal(Cp2_, 0)
        p_values_range=Cp2#place holder that can stay to emulate pvalues
        Legend = np.dstack((np.concatenate((p_values_range*Cp1, Cp2), axis=1)[:,::-1],
                            np.concatenate((Cp2, p_values_range*Cp1[:,::-1], ), axis=1)[:,::-1],
                            np.concatenate((Cp2, Cp2), axis=1)[:,::-1]))
        cmap=ListedColormap(Legend)

    #normalization
    res_img = (res_img[:,:,:,:]-np.amin(res_img[:,:,:,:]))/(np.amax(res_img[:,:,:,:])-np.amin(res_img[:,:,:,:]))
    res_img[np.where(res_img < threshold)]= -1


    #assign colors
    instance =np.zeros(res1[:,:,:,0].shape+(3,))
    res1=np.abs(res1)+1e-9
    res2=np.abs(res2)+1e-9
    res1[np.where(res1>1.0)]=0.9999#1.0
    res2[np.where(res2>1.0)]=0.9999#1.0

    def _cmap_(res1, res2, voxel, pvalue=0.0, red_blue=False):
        if(voxel==-1):
            return (0.999,0.999,0.999)
        elif(pvalue>0.05):
            pvalue=np.clip(pvalue, a_min=1e-9, a_max=0.2)
            return (pvalue,pvalue,pvalue)
        if(red_blue):
            if(res1<res2):
                return (1+1e-30-res1, 0.0001, pvalue+1e-9)
            return (pvalue+1e-9, 0.0001, 1+1e-30-res2)
        else:
            if(res1<res2):
                return (0.0001, 1+1e-30-res1, 1+1e-30-res1)
            return (1+1e-30-res2, 0.0001, 1+1e-30-res2)

    for voxel1 in range(instance.shape[0]):
        for voxel2 in range(instance.shape[1]):
            for voxel3 in range(instance.shape[2]):
                instance[voxel1,voxel2,voxel3] = np.array(list(_cmap_(res1[voxel1,voxel2,voxel3,0], res2[voxel1,voxel2,voxel3,0], 
                                                                res_img[voxel1,voxel2,voxel3,0],
                                                                pvalue=pvalues[voxel1,voxel2,voxel3,0],
                                                                red_blue=red_blue)))

    fig = plt.figure(figsize=(25,17))
    gs = GridSpec(41, 5, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    #colorbar
    cax = fig.add_subplot(gs[30:33,2:3])
    cax.imshow(Legend, extent=[0,100,0,100], aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([5,95])
    cax.annotate('', xy=(0, -0.5), xycoords='axes fraction', xytext=(1, -0.5), 
                arrowprops=dict(arrowstyle="<->, head_width=0.4", color='black'))
    cax.annotate(model1, xy=(0, -0.56), xycoords='axes fraction', xytext=(-0.4, -0.56), size=20, color="red")
    cax.annotate(model2, xy=(0, -0.56), xycoords='axes fraction', xytext=(1.03, -0.56), size=20, color="blue")
    cax.set_yticklabels([r"$p=0.0$",r"$p=1.0$"], size=20)
    

    #plot each slice in 2 Dimensional plot
    row = 1
    col = 4
    for axis in range((instance[:,:,:].shape[2])//factor):
        if(row==1):
            _row=20
        else:
            _row=10
        axes = fig.add_subplot(gs[_row:_row+10,col])
        img = rotate(instance[:,:,axis*factor,:], 90)
        axes.imshow(img)
        if(slice_label):
            axes.text(28, 1, label+str(axis*factor)+"}$", size=13,
                 va="baseline", ha="left", multialignment="left",)
        axes.axis("off")
        col -= 1
        if(col == -1):
            col=4
            row-=1

    plt.rcParams["font.family"] = "serif"
    fig.set_tight_layout(True)

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig




def plot_analysis_uncertainty(runs, res_img, evaluations, xlabel=r"$Var[res]$", ylabel=r"$H$", threshold=0.37, save=False, save_path=None, save_format="pdf"):
    """
    Plot from: Bayesian DCT Uncertainty Quantification

    Examples:

    >>> from utils import viz_utils, preprocess_data
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from pathlib import Path
    >>>
    >>> home = str(Path.home())
    >>> 
    >>> n_individuals=getattr(data_utils, "n_individuals_"+dataset)
    >>> with tf.device('/CPU:0'):
    >>>     train_data, test_data = preprocess_data.dataset(dataset, n_individuals=n_individuals,
    ...                                                interval_eeg=interval_eeg, 
    ...                                                ind_volume_fit=False,
    ...                                                standardize_fmri=True,
    ...                                                iqr=False,
    ...                                                verbose=True)
    >>>     eeg_train, fmri_train =train_data
    >>>     eeg_test, fmri_test = test_data
    >>>
    >>> H=[2,5,7,10,13,15,18,20]
    >>> sinusoids_res = np.zeros((8,64,64,30,1))
    >>> path_res = home+"/eeg_to_fmri/metrics/01_topographical_attention_random_fourier_features_attention_style_variational_VonMises_dependent_h_"
    >>> for sin in range(len(H)):
    >>>     sinusoids_res[sin] = np.mean(np.squeeze(np.load(path_res+str(H[sin])+"_30x30x15_res_2.0/metrics/res_seed_11.npy"), axis=1), axis=0)
    >>>
    >>> resolutions=["3x3x1","6x6x3","12x12x6","15x15x7","18x18x9","20x20x10","25x25x12","30x30x15"]
    >>> resolutions_res = np.zeros((8,64,64,30,1))
    >>> path_res = home+"/eeg_to_fmri/metrics/01_topographical_attention_random_fourier_features_attention_style_variational_VonMises_dependent_h_15_"
    >>> for res in range(len(resolutions)):
    >>>     resolutions_res[res] = np.mean(np.squeeze(np.load(path_res+resolutions[res]+"_res_2.0/metrics/res_seed_11.npy"), axis=1), axis=0)
    >>>
    >>> viz_utils.plot_analysis_uncertainty(sinusoids_res, fmri_train, H, xlabel=r"$Var[res]$", ylabel=r"$H$", threshold=0.37, save=False, save_path=None, save_format="pdf")
    >>> viz_utils.plot_analysis_uncertainty(resolutions_res, fmri_train, resolutions, xlabel=r"$Var[res]$", ylabel=r"$R$", threshold=0.37, save=False, save_path=None, save_format="pdf")


    """
    img = np.mean(res_img, axis=0)

    cp1 = np.linspace(0,1)
    cp2 = np.linspace(0,1)
    Cp1, Cp2 = np.meshgrid(cp1,cp2)
    C0 = np.full_like(Cp1, Cp1*Cp2*((Cp1)+(Cp2))/2)
    Cp2_ = np.triu(Cp2)
    p_values_range=Cp2#place holder that can stay to emulate pvalues
    Legend = np.dstack((
                        (p_values_range*Cp1)[:,:][:,15:],
                        (p_values_range*Cp1)[:,::-1][:,15:],
                        Cp2[:,15:],
                        ))
    cmap=ListedColormap(Legend)

    
    def _cmap_(analysis, voxel, list_range, threshold=0., threshold_q=1e-1, epsilon=1e-3):
        """
        analysis \in \mathbb{R}^H
        voxel \in \mathbb{R}

        """
        if(voxel==-1):
            return (0.999,0.999,0.999)

        if(np.std(analysis) > threshold and analysis[np.argmin(analysis)] < threshold_q):
            return (((np.argmin(analysis)+1)/analysis.shape[0])-(analysis[np.argmin(analysis)])-epsilon,
                    1.-((np.argmin(analysis)+1)/analysis.shape[0])-(analysis[np.argmin(analysis)])-epsilon,
                    1.-analysis[np.argmin(analysis)]-epsilon)

        return (epsilon, epsilon, epsilon)


    threshold_q=np.quantile(np.abs(runs), 0.5)
    img = (img[:,:,:,:]-np.amin(img[:,:,:,:]))/(np.amax(img[:,:,:,:])-np.amin(img[:,:,:,:]))
    img[np.where(img < threshold)]= -1
    instance=np.zeros(res_img[0,:,:,:,0].shape+(3,))
    for voxel1 in range(instance.shape[0]):
        for voxel2 in range(instance.shape[1]):
            for voxel3 in range(instance.shape[2]):
                instance[voxel1,voxel2,voxel3] = np.array(list(_cmap_(np.abs(runs[:,voxel1,voxel2,voxel3,0]),
                                                                img[voxel1,voxel2,voxel3,0],
                                                                evaluations, threshold=1e-1, 
                                                                threshold_q=threshold_q, epsilon=1e-1)))

    fig = plt.figure(figsize=(50,10))
    gs = GridSpec(2, 49, figure=fig, wspace=0.01, hspace=0.05)
    #colorbar
    cax = fig.add_subplot(gs[:,3:4])
    cax.imshow(rotate(Legend, 90, axes=(0,1)), extent=[0,100,0,100], aspect="auto")
    cax.set_yticks([5 , 20, 30 , 42, 55, 65, 80 ,90])
    cax.set_xticks([5,95])
    cax.annotate('', xy=(1.4, 0), xycoords='axes fraction', xytext=(1.4, 1), arrowprops=dict(arrowstyle="<-, head_width=0.4", color='black'))
    cax.set_xticklabels([r"<{:0.3f}".format(1e-3), r"$\infty$"], size=20)
    cax.set_yticklabels(evaluations, size=20)
    cax.set_xlabel(xlabel, size=20)
    cax.set_ylabel(ylabel, size=20)
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position("right")




    #plot each slice in 2 Dimensional plot
    row = 1
    col = 5
    factor=3
    for axis in range((instance[:,:,:].shape[2])//factor):

        axes = fig.add_subplot(gs[row:row+1,col*5:col*5+5])
        img = rotate(instance[:,:,axis*factor,:], 90)
        axes.imshow(img)
        axes.axis("off")
        col -= 1
        if(col == 0):
            col=5
            row-=1

    plt.rcParams["font.family"] = "serif"
    fig.set_tight_layout(True)
    
    if(save):
        fig.savefig(save_path, format=save_format)

    return fig




def single_display_gt_pred_espistemic_aleatoric(im1, im2, im3, im4, name="default", xslice=14, threshold=0.37, cmap=plt.cm.nipy_spectral, save=False, save_path=None, save_format="pdf"):
    """
    Single display plot

    """
    
    def normalize_img(img, threshold=0.37):
        img = (img[:,:,:,:]-np.amin(img[:,:,:,:]))/(np.amax(img[:,:,:,:])-np.amin(img[:,:,:,:]))
        img[np.where(img < threshold)]= 1.001

        return img
    
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_over("w")

    fig = plt.figure(figsize=(25,7))
    gs = GridSpec(2, 9, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)

    axes = fig.add_subplot(gs[:,0])
    if(len(name)<10):
        axes.text(0.75,0.3,name,rotation="vertical",size=40)
    else:
        axes.text(0.75,0.13,name,rotation="vertical",size=40)
    
    axes.axis("off")
    #ground truth
    axes = fig.add_subplot(gs[:,1:3])
    axes.imshow(rotate(im1[0,:,:,xslice,0], 90, axes=(0,1)),
              cmap=cmap)
    axes.axis("off")
    #predicted view
    axes = fig.add_subplot(gs[:,3:5])
    axes.imshow(rotate(im2[0,:,:,xslice,0], 90, axes=(0,1)),
              cmap=cmap)
    axes.axis("off")
    #epistemic uncertainty
    axes = fig.add_subplot(gs[:,5:7])
    axes.imshow(rotate(im3[0,:,:,xslice,0], 90, axes=(0,1)),
              cmap=cmap)
    axes.axis("off")
    #aleatoric uncertainty
    axes = fig.add_subplot(gs[:,7:9])
    axes.imshow(rotate(im4[0,:,:,xslice,0], 90, axes=(0,1)),
              cmap=cmap)
    axes.axis("off")
    
    plt.rcParams["font.family"] = "serif"
    plt.tight_layout()
    
    if(save):
        fig.savefig(save_path, format=save_format)
        
    return fig


def whole_display_gt_pred_espistemic_aleatoric(im1, im2, im3, im4, cmap=plt.cm.nipy_spectral, factor=5, save=False, save_path=None, save_format="pdf"):
    """
    Whole brain display - for Bayesian uncertainty Quantification
    """
    cmap = copy.copy(mpl.cm.get_cmap(cmap))
    cmap.set_over("w")

    fig = plt.figure(figsize=(22,30))
    gs = GridSpec(im1.shape[3]//factor, 9, figure=fig, wspace=0.01, hspace=0.05)#, wspace=-0.4)
    
    for xslice in reversed(range(factor,im1.shape[3],factor)):
        
        axes = fig.add_subplot(gs[(im1.shape[3]-xslice)//factor-1,0])
        if(xslice==factor):
            x_shift=0.2
        else:
            x_shift=0.15
        axes.text(0.75,x_shift, "slice "+str(xslice)+"/"+str(im1.shape[3]), size=35, rotation="vertical")
        axes.axis("off")
        #ground truth
        axes = fig.add_subplot(gs[(im1.shape[3]-xslice)//factor-1,1:3])
        axes.imshow(rotate(im1[0,:,:,xslice,0], 90, axes=(0,1)),
                  cmap=cmap)
        if(xslice==im1.shape[3]-factor):
            axes.set_title("Ground Truth", size=35)
        axes.axis("off")
        #predicted view
        axes = fig.add_subplot(gs[(im1.shape[3]-xslice)//factor-1,3:5])
        axes.imshow(rotate(im2[0,:,:,xslice,0], 90, axes=(0,1)),
                  cmap=cmap)
        if(xslice==im1.shape[3]-factor):
            axes.set_title("Predicted", size=35)
        axes.axis("off")
        #epistemic uncertainty
        axes = fig.add_subplot(gs[(im1.shape[3]-xslice)//factor-1,5:7])
        axes.imshow(rotate(im3[0,:,:,xslice,0], 90, axes=(0,1)),
                  cmap=cmap)
        if(xslice==im1.shape[3]-factor):
            axes.set_title("Epistemic", size=35)
        axes.axis("off")
        #aleatoric uncertainty
        axes = fig.add_subplot(gs[(im1.shape[3]-xslice)//factor-1,7:9])
        axes.imshow(rotate(im4[0,:,:,xslice,0], 90, axes=(0,1)),
                  cmap=cmap)
        if(xslice==im1.shape[3]-factor):
            axes.set_title("Aleatoric", size=35)
        axes.axis("off")

    plt.rcParams["font.family"] = "serif"
    plt.tight_layout()

    if(save):
        fig.savefig(save_path, format=save_format)
        
    return fig



##########################################################################################################
#
#                                         LRP - Plotting Relevances
#
##########################################################################################################



def R_channels(R, X, ch_names=None, save=False, save_path=None, save_format="pdf"):
    """

    Inputs:
        * R - np.ndarray
        * X - np.ndarray
    """
    if(ch_names is None):
        ch_names=list(range(X.shape[1]))

    fig = plt.figure(figsize=(27,45))

    n=16
    m=8

    gs = GridSpec(n, m, figure=fig, wspace=0.1, hspace=0.01)

    R_channels = np.mean(R, axis=0)[:,:,:,0]
    norm_R_channels = (R_channels)
    instances = np.mean(X, axis=0)[:,:,:,0]

    i=0
    j=0

    for channel in range(R_channels.shape[0]):


        axes = fig.add_subplot(gs[i,j])
        axes.imshow(R_channels[channel,:,:], 
                    cmap=plt.cm.inferno,
                    vmin=np.amin(R_channels),
                    vmax=np.amax(R_channels),
                    aspect=0.05)
        axes.set_title(ch_names[channel]+" : "+r"$\frac{\sum_i R_{iC}}{\sum_i \sum_c R_{ic}} = $"+\
                               str(np.sum(norm_R_channels[channel,:,:])/np.sum(norm_R_channels[:,:,:])), size=8)
        axes.set_xticks([])
        axes.set_yticks([])

        axes = fig.add_subplot(gs[i,j+1])
        axes.imshow(instances[channel,:,:],
                    aspect=0.05,
                    vmin=np.amin(instances),
                    vmax=np.amax(instances))
        axes.set_title(ch_names[channel])
        axes.set_xticks([])
        axes.set_yticks([])

        j+=2
        if(j >= m):
            i+=1
            j=0
        if(i >= n):
            break
        continue

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] =9
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()
        
        


def R_analysis_channels(R, channels, ch_names=None, save=False, save_path=None, save_format="pdf"):
    """

    Inputs:
        * R - np.ndarray
        * channels - int32
    """
    if(ch_names is None):
        ch_names=list(range(channels))

    pop_channels = np.sum(np.sum(R[:,:,:,:,0], axis=3), axis=2)

    pvalues=np.zeros((pop_channels.shape[1], pop_channels.shape[1]))
    for channel1 in range(pop_channels.shape[1]):
        for channel2 in range(pop_channels.shape[1]):
            if(channel1 == channel2):
                continue
            pvalues[channel1,channel2] = ttest_ind(pop_channels[:,channel1], pop_channels[:,channel2]).pvalue


    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(channels, channels+6, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[0,:])
    axes.imshow(np.std(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(channels)))
    axes.set_xticklabels(ch_names)
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[1,:])
    axes.imshow(np.amax(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[2,:])
    axes.imshow(np.amin(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()

    #plot pvalues
    axes = fig.add_subplot(gs[3:,:])
    stat_sign = (pvalues >= 0.05).astype("float32")
    stat_sign[np.where(stat_sign == 0)] = pvalues[np.where(stat_sign == 0)]
    stat_sign[np.where(stat_sign == 1)] = 0.05
    axes.imshow(stat_sign,
               cmap=plt.cm.Greens_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks(list(range(channels)))
    axes.set_yticklabels(ch_names)
    axes.set_ylabel("Electrodes")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()





def R_analysis_freqs(R, freqs, save=False, save_path=None, save_format="pdf"):
    """

    Inputs:
        * R - np.ndarray
        * freqs - int32
    """
    pop_freqs = np.sum(np.sum(R[:,:,:,:,0], axis=3), axis=1)

    pvalues=np.zeros((pop_freqs.shape[1], pop_freqs.shape[1]))
    for freq1 in range(pop_freqs.shape[1]):
        for freq2 in range(pop_freqs.shape[1]):
            if(freq1 == freq2):
                continue
            pvalues[freq1,freq2] = ttest_ind(pop_freqs[:,freq1], pop_freqs[:,freq2]).pvalue

    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(freqs-50, freqs+6, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[0,:])
    axes.imshow(np.std(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(5,freqs,5)))
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[1,:])
    axes.imshow(np.amax(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[2,:])
    axes.imshow(np.amin(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()

    #plot pvalues
    axes = fig.add_subplot(gs[3:,:])
    stat_sign = (pvalues >= 0.05).astype("float32")
    stat_sign[np.where(stat_sign == 0)] = pvalues[np.where(stat_sign == 0)]
    stat_sign[np.where(stat_sign == 1)] = 0.05
    axes.imshow(stat_sign,
               cmap=plt.cm.Greens_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks(list(range(5,freqs,5)))
    axes.set_ylabel("Hz")

    plt.rcParams["font.family"] = "serif"
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()

        

def R_analysis_times(R, times, save=False, save_path=None, save_format="pdf"):
    """

    Inputs:
        * R - np.ndarray
        * times - int32
    """
    

    pop_times = np.sum(np.sum(R[:,:,:,:,0], axis=2), axis=1)

    pvalues=np.zeros((pop_times.shape[1], pop_times.shape[1]))
    for time1 in range(pop_times.shape[1]):
        for time2 in range(pop_times.shape[1]):
            if(time1 == time2):
                continue
            pvalues[time1,time2] = ttest_ind(pop_times[:,time1], pop_times[:,time2]).pvalue

    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(3*times, times, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[0,:])
    axes.imshow(np.std(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(times)))
    axes.set_xticklabels(list(range(0,2*times,2)))
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[1,:])
    axes.imshow(np.amax(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[2,:])
    axes.imshow(np.amin(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()

    #plot pvalues
    axes = fig.add_subplot(gs[3:,:])
    stat_sign = (pvalues >= 0.05).astype("float32")
    stat_sign[np.where(stat_sign == 0)] = pvalues[np.where(stat_sign == 0)]
    stat_sign[np.where(stat_sign == 1)] = 0.05
    axes.imshow(stat_sign,
               cmap=plt.cm.Greens_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks(list(range(0,times,1)))
    axes.set_yticklabels(list(range(0,2*times,2)))
    axes.set_ylabel("Seconds")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 22
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()
        



def R_analysis_dimensions(R, ch_names=None, save=False, save_path=None, save_format="pdf"):
    """
    Inputs:
        * R - np.ndarray
    """

    pop_channels = np.sum(np.sum(R[:,:,:,:,0], axis=3), axis=2)
    pop_freqs = np.sum(np.sum(R[:,:,:,:,0], axis=3), axis=1)
    pop_times = np.sum(np.sum(R[:,:,:,:,0], axis=2), axis=1)

    channels=pop_channels.shape[1]
    freqs=pop_freqs.shape[1]
    times=pop_times.shape[1]

    if(ch_names is None):
        ch_names=list(range(channels))

    fig = plt.figure(figsize=(25,10))
    gs = GridSpec(23, 128, figure=fig, wspace=0.1, hspace=0.001)

    axes = fig.add_subplot(gs[0:4,:])
    axes.text(0.5, 0.5, "Channels",ha="center", va="center", size=22)
    axes.axis('off')

    #plot variance
    axes = fig.add_subplot(gs[4,:])
    axes.imshow(ttest_1samp(pop_channels, np.mean(pop_channels),axis=0).pvalue.reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(channels)))
    axes.set_xticklabels(ch_names,size=10)
    axes.set_yticks([0])
    axes.set_yticklabels([r"$p$-value"])
    axes.xaxis.tick_top()

    #plot variance
    axes = fig.add_subplot(gs[5,:])
    axes.imshow(np.std(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[6,:])
    axes.imshow(np.amax(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[7,:])
    axes.imshow(np.amin(pop_channels,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()



    axes = fig.add_subplot(gs[8:11,:])
    axes.text(0.5, 0.5, "Frequencies",ha="center", va="center", size=22)
    axes.axis('off')

    axes = fig.add_subplot(gs[11,:])
    axes.imshow(ttest_1samp(pop_freqs, np.mean(pop_freqs),axis=0).pvalue.reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(0, freqs, 5)))
    axes.set_yticks([0])
    axes.set_yticklabels([r"$p$-value"])
    axes.xaxis.tick_top()

    #plot variance
    axes = fig.add_subplot(gs[12,:])
    axes.imshow(np.std(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[13,:])
    axes.imshow(np.amax(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[14,:])
    axes.imshow(np.amin(pop_freqs,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()



    axes = fig.add_subplot(gs[15:18,:])
    axes.text(0.5, 0.5, "Times",ha="center", va="center", size=22)
    axes.axis('off')

    axes = fig.add_subplot(gs[18,:])
    axes.imshow(ttest_1samp(pop_times, np.mean(pop_times),axis=0).pvalue.reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(times)))
    axes.set_xticklabels(list(range(6,2*times+6,2)))
    axes.set_yticks([0])
    axes.set_yticklabels([r"$p$-value"])
    axes.xaxis.tick_top()

    #plot variance
    axes = fig.add_subplot(gs[19,:])
    axes.imshow(np.std(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$Var[R]$"])
    axes.xaxis.tick_top()

    #plot max relevance
    axes = fig.add_subplot(gs[20,:])
    axes.imshow(np.amax(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$max(R)$"])
    axes.xaxis.tick_top()

    #plot min relevance
    axes = fig.add_subplot(gs[21,:])
    axes.imshow(np.amin(pop_times,axis=0).reshape(1,-1),
               cmap=plt.cm.Blues_r,
               aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([0])
    axes.set_yticklabels([r"$min(R)$"])
    axes.xaxis.tick_top()

    plt.rcParams["font.family"] = "serif"
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()


def R_analysis_times_freqs(R, times, freqs, func=np.std, save=False, save_path=None, save_format="pdf"):
    """
    Inputs:
        * R - np.ndarray
        * times - int32
        * freqs - int32
    """
    pop = (np.sum(R[:,:,:,:,0], axis=1),)

    if(func is metrics.ttest_1samp_r):
        pop = pop+(np.mean(pop),)
    
    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(freqs-50, freqs+6, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[:,:])
    axes.imshow(func(*pop,axis=0),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_yticks(list(range(5,freqs,5)))
    axes.set_xticks(list(range(times)))
    axes.set_xticklabels(list(range(6,2*times+6,2)))
    axes.set_ylabel(r"$Hz$")
    axes.set_xlabel("Seconds")
    axes.invert_yaxis()
    
    plt.rcParams["font.family"] = "serif"
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()


def R_analysis_channels_freqs(R, channels, freqs, func=np.std, ch_names=None, save=False, save_path=None, save_format="pdf"):
    """
    Inputs:
        * R - np.ndarray
        * times - int32
        * freqs - int32
    """
    pop = (np.sum(R[:,:,:,:,0], axis=3),)

    if(func is metrics.ttest_1samp_r):
        pop = pop+(np.mean(pop),)
    
    if(ch_names is None):
        ch_names=list(range(channels))
        
    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(freqs-50, freqs+6, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[:,:])
    axes.imshow(func(*pop,axis=0).T,
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_yticks(list(range(5,freqs,5)))
    axes.set_xticks(list(range(channels)))
    axes.set_xticklabels(ch_names,size=8)
    axes.set_ylabel(r"$Hz$")
    axes.set_xlabel("Electrodes")
    axes.invert_yaxis()
    
    plt.rcParams["font.family"] = "serif"
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()



def R_analysis_times_channels(R, times, channels, func=np.std, ch_names=None, save=False, save_path=None, save_format="pdf"):
    """
    Inputs:
        * R - np.ndarray
        * times - int32
        * channels - int32
    """
    pop = (np.sum(R[:,:,:,:,0], axis=2),)
    
    if(func is metrics.ttest_1samp_r):
        pop = pop+(np.mean(pop),)

    if(ch_names is None):
        ch_names=list(range(channels))
        
    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(channels-50, channels+6, figure=fig, wspace=0.1, hspace=0.001)

    #plot variance
    axes = fig.add_subplot(gs[:,:])
    axes.imshow(func(*pop,axis=0),
               cmap=plt.cm.Blues,
               aspect="auto")
    axes.set_xticks(list(range(times)))
    axes.set_xticklabels(list(range(6, 2*times+6,2)))
    axes.set_yticks(list(range(channels)))
    axes.set_yticklabels(ch_names,size=8)
    axes.set_ylabel(r"$Hz$")
    axes.set_xlabel("Electrodes")
    axes.invert_yaxis()
    
    plt.rcParams["font.family"] = "serif"
    
    if(save):
        fig.savefig(save_path, format=save_format)
    else:
        fig.show()






def plot_eeg_channels(colors=None, scores=None, edges=None, edge_threshold=0.5, edge_width=3., dataset="01", plot_names=False):
    """
    Plot EEG cap 10-20 system


    Good for attention scores and check which channels are related
    """
    #circle1 = plt.Circle((0, 0), 0.2, color='r')
    #circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
    head = plt.Circle((0.5,0.5), 0.47, linestyle='-',edgecolor='black',fill=True, facecolor="white", zorder=0)

    nose1 = plt.Circle((0.5,1.0), 0.025, linestyle='-',edgecolor='black', linewidth=1., fill=True,facecolor="white", zorder=-3)
    nose2 = plt.Circle((0.48, 0.97), 0.025, linestyle='-',edgecolor='black', linewidth=1., fill=True,facecolor="white", zorder=-2)
    nose3 = plt.Circle((0.52,0.97), 0.025, linestyle='-',edgecolor='black', linewidth=1., fill=True,facecolor="white", zorder=-2)
    nose = plt.Polygon(np.array([[0.473,1.], [0.527,1.0], [0.473,0.9], [0.527,0.9]]), linestyle='-', fill=True, facecolor="white", zorder=-1)

    lear = plt.Circle((0.1,0.5), 0.1, linestyle='-',edgecolor='black', linewidth=1., fill=True,facecolor="white", zorder=-1)
    rear = plt.Circle((0.9,0.5), 0.1, linestyle='-',edgecolor='black', linewidth=1., fill=True,facecolor="white", zorder=-1)

    channel_circle=[]
    channel_names=getattr(eeg_utils, "channels_"+dataset)
    for channel in channel_names:
        if(not channel in eeg_utils.channels_coords_10_20.keys()):
            continue
        if(colors is None):
            facecolor="white"
        else:
            facecolor=colors[channel]

        linewidth=1.
        if(np.any(edges[channel_names.index(channel),:]>edge_threshold) or \
            np.any(edges[:,channel_names.index(channel)]>edge_threshold)):
            linewidth=3.

        channel_circle+=[plt.Circle(eeg_utils.channels_coords_10_20[channel], 0.025, 
                                    linestyle='-',edgecolor='black',
                                    linewidth=linewidth,
                                    fill=True,facecolor=facecolor, zorder=5)]




    fig,axes=plt.subplots(figsize=(10,10))

    axes.add_patch(head)
    axes.add_patch(nose)
    axes.add_patch(nose1)
    axes.add_patch(nose2)
    axes.add_patch(nose3)
    axes.add_patch(lear)
    axes.add_patch(rear)
    #channels
    for channel in channel_circle:
        axes.add_patch(channel)
    
    
    if(plot_names):
        for channel in channel_names:
            if(not channel in eeg_utils.channels_coords_10_20.keys()):
                continue

            color="black"
            if(scores is None):
                color="black"
            else:
                if(scores[channel] > 0.5):
                    color="white"

            text_shift=0.005*len(channel)

            axes.text(eeg_utils.channels_coords_10_20[channel][0]-text_shift,
                      eeg_utils.channels_coords_10_20[channel][1]-0.006, 
                        channel, color=color, size=8, zorder=6)

    #attention scores lines
    for channel1 in range(len(channel_names)):
        for channel2 in range(len(channel_names)):
            if(channel1 == channel2 or not channel_names[channel1] in eeg_utils.channels_coords_10_20.keys() or not channel_names[channel2] in eeg_utils.channels_coords_10_20.keys()):
                continue

            if(edges[channel1,channel2] > edge_threshold):

                axes.plot([eeg_utils.channels_coords_10_20[channel_names[channel1]][0], eeg_utils.channels_coords_10_20[channel_names[channel2]][0]], 
                            [eeg_utils.channels_coords_10_20[channel_names[channel1]][1], eeg_utils.channels_coords_10_20[channel_names[channel2]][1]], 
                            linewidth=edge_width,
                            color="black",
                            zorder=3)

    axes.axis('off')
    axes.margins(0.001, 0.01)

    plt.rcParams["font.family"] = "serif"

    return fig



def plot_attention_eeg(attention, dataset="01", cmap=mpl.cm.Blues, edge_threshold=3., plot_names=False, save=False, save_path=None, save_format="pdf"):

    channels_names=getattr(eeg_utils, "channels_"+dataset)
    edges=attention

    channel_scores=np.mean(edges, axis=1)
    channel_scores=(channel_scores-np.amin(channel_scores))/(np.amax(channel_scores)-np.amin(channel_scores))

    channel_colors=dict(zip(channels_names, cmap(channel_scores)))
    channel_scores=dict(zip(channels_names, channel_scores))

    fig=plot_eeg_channels(colors=channel_colors,
                            scores=channel_scores,
                            edges=edges,
                            edge_threshold=edge_threshold, 
                            dataset=dataset, plot_names=plot_names)

    if(save):
        fig.savefig(save_path, format=save_format)

    return fig