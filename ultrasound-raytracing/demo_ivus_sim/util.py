from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.signal
import numpy as np
from scipy.spatial.transform import Rotation
from functools import partial
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from tqdm import tqdm

def show_xducer(field, Th, scatterers=None, r_cath=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # draw catheter
    if r_cath is not None:
        height = 4e-3
        theta = np.linspace(0, 2 * np.pi, 50)
        x = np.linspace(-height/2, height/2, 50)
        theta, x = np.meshgrid(theta, x)
        y = r_cath * np.cos(theta)
        z = r_cath * np.sin(theta)
        ax.plot_surface(x*1e3, y*1e3, z*1e3, color='gray', alpha=0.25, rstride=1, cstride=1, edgecolor=None)

    # draw array

    rect = field.xdc_get(Th,'rect')

    if np.ndim(rect)==1:
        rect = np.atleast_2d(rect).T
    
    
    
    for rect_ind in range(np.shape(rect)[1]):
    
        rect_verts = [
            [rect[10,rect_ind], rect[11,rect_ind], rect[12,rect_ind]],
            [rect[13,rect_ind], rect[14,rect_ind], rect[15,rect_ind]],
            [rect[16,rect_ind], rect[17,rect_ind], rect[18,rect_ind]],
            [rect[19,rect_ind], rect[20,rect_ind], rect[21,rect_ind]]
        ]
    
        faces = [np.array(rect_verts)*1e3]
    
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', edgecolors='r', linewidths=0.25, alpha=1))

    # draw scatterers
    if scatterers is not None:
        for scatterer in scatterers:
            ax.scatter(scatterer[0]*1e3, scatterer[1]*1e3, scatterer[2]*1e3, c='k', marker='o')

    
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.tight_layout()
    plt.show()


def define_impulse_resp(fc, fbw, fs):

    bw = fc * fbw

    cutoff = scipy.signal.gausspulse('cutoff', fc=fc, bw=fbw, tpr=-40, bwr=-3)
    adj_cutoff = np.ceil(cutoff * fs) / fs
    
    t_acc = np.arange(-adj_cutoff, adj_cutoff + 1 / fs, 1 / fs)
    _, h_acc = scipy.signal.gausspulse(t_acc, fc=fc, bw=fbw, retquad=True, bwr=-3)
    t_acc = t_acc - np.min(t_acc)

    return t_acc, h_acc


def mask_scatterers_local(
    scatterers_global,
    translation_vec,          # world
    rotation_mat,             # 3x3
    x_len,                    # half-length along axial (+/- x_local)
    imaging_depth,            # radial depth (in YZ)
    rotation_is_body_to_world=True,
):
    # update: scatterers_global: Nx4 (x,y,z,refl)

    pts_w = scatterers_global[:, :3]
    refl  = scatterers_global[:, 3]

    d = (pts_w - translation_vec).astype(np.float32)   # world-centered

    R = rotation_mat
    if rotation_is_body_to_world:
        # row-vector world→local
        p_locals_all = d @ R      # NOTE: no .T here
    else:
        # R is world→body
        p_locals_all = d @ R.T

    x_local = p_locals_all[:, 0]
    r_local = np.hypot(p_locals_all[:, 1], p_locals_all[:, 2])  # radial (YZ)

    mask = (np.abs(x_local) <= x_len) & (r_local <= imaging_depth)

    p_locals = p_locals_all[mask]
    return np.column_stack((p_locals, refl[mask]))


def rotate_array(field, Th):
    
    rect = field.xdc_get(Th,'rect')
    if np.ndim(rect)==1:
        rect = np.atleast_2d(rect).T
    
    rect2 = []

    elm_centers = np.zeros((len(np.unique(rect[0,:])),3))
    
    
    for ind in range(np.shape(rect)[1]):
    
        rect_elm  = [rect[0,ind]+1] + list(rect[10:22,ind]) + [rect[4,ind]] + list(rect[2:4,ind]) + list(rect[7:10,ind])
        rect2.append(rect_elm)
        
        phys_elm_ind = int(rect[0,ind]) 
        elm_centers[phys_elm_ind] = rect[23:,ind]

        # flip x and y
        elm_centers[phys_elm_ind] = elm_centers[phys_elm_ind][[1,0,2]]
    
  
    
    rect2 = np.array(rect2)
    
    # flip x and y axes so x axis is the cylinder axis
    xvals = rect2[:,[1,4,7,10,16]]
    rect2[:,[1,4,7,10,16]] = rect2[:,[2,5,8,11,17]]
    rect2[:,[2,5,8,11,17]] = xvals

    Th2 = field.xdc_rectangles(rect2, elm_centers, [0,0,1e6])

    return Th2, elm_centers

def rotate_and_zOffset_array(field, Th, r_array):
    # this is just for visualizaiton
    # field ii seems to reset the z offset to zero at initialization

    rect = field.xdc_get(Th,'rect')
    if np.ndim(rect)==1:
        rect = np.atleast_2d(rect).T

    rect2 = []

    elm_centers = np.zeros((len(np.unique(rect[0,:])),3))


    for ind in range(np.shape(rect)[1]):


        rect_elm  = [rect[0,ind]+1] + list(rect[10:22,ind]) + [rect[4,ind]] + list(rect[2:4,ind]) + list(rect[7:10,ind])
        rect2.append(rect_elm)

        phys_elm_ind = int(rect[0,ind])
        elm_centers[phys_elm_ind] = rect[23:,ind] + np.array([0,0,r_array])

        # flip x and y
        elm_centers[phys_elm_ind] = elm_centers[phys_elm_ind][[1,0,2]]



    rect2 = np.array(rect2)

    # add z offset
    rect2[:,[3,6,9,12,18]] += r_array

    # flip x and y axes so x axis is the cylinder axis
    xvals = rect2[:,[1,4,7,10,16]]
    rect2[:,[1,4,7,10,16]] = rect2[:,[2,5,8,11,17]]
    rect2[:,[2,5,8,11,17]] = xvals

    Th2 = field.xdc_rectangles(rect2, elm_centers, [0,0,1e6])

    return Th2


def recon_tx(t,Ascan,t_total):

    interp_type='linear'

    Ascan_h = scipy.signal.hilbert(Ascan,axis=0)
    intfun = interp1d(t,Ascan_h,interp_type,
                          bounds_error=False,fill_value=0j)

    pixval = np.squeeze(intfun(t_total))
    pixval[np.where(np.isnan(pixval))] = 0

    return pixval

def polar_to_cartesian(im_polar, r_min, r_max, dtheta_range_global):
    
    theta_size, r_size = im_polar.shape
    cart_size = r_size * 2

    x = np.linspace(-r_max, r_max, cart_size)
    y = np.linspace(-r_max, r_max, cart_size)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X) % (2 * np.pi)

    R_idx = (R - r_min) / (r_max - r_min) * (r_size - 1)
    Theta_idx = Theta / (2 * np.pi + dtheta_range_global) * theta_size

    cart_img = map_coordinates(im_polar, [Theta_idx.ravel(), R_idx.ravel()],
                               order=1, mode='constant', cval=0.0)
    cart_img = cart_img.reshape(cart_size, cart_size)

    return cart_img, x, y


def recon_img(RFdata_2D, elm_centers, pixel_coords, tx_pulse_delay, tx_inds, rx_inds, c0, t):

    pix_vals = []
    
    for tx_ind in tx_inds:
    
        tx_elm_pos = np.tile(elm_centers[tx_ind, :3], (len(pixel_coords), 1))
        t_tx = np.linalg.norm(pixel_coords - tx_elm_pos, axis=1) / c0
    
        for rx_ind in rx_inds:
            Ascan = RFdata_2D[:, tx_ind, rx_ind]
    
            rx_elm_pos = np.tile(elm_centers[rx_ind, :3], (len(pixel_coords), 1))
            t_rx = np.linalg.norm(pixel_coords - rx_elm_pos, axis=1) / c0
    
            t_total = t_tx + t_rx + 2 * tx_pulse_delay
    
            pixval = recon_tx(t,Ascan,t_total)
            pix_vals.append(pixval)
    
    return pix_vals
