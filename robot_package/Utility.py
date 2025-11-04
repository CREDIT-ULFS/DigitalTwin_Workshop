import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sym
import pyvista as pv
import pyvistaqt as pvqt
from tabulate import tabulate
from numpy import sin as s, cos as c, pi
import symengine
from tqdm import tqdm


from .Robot_plotter import RobotPlotter
from .Inverse_Kinematics import InverseKinematics

def MAC(phi_1, phi_2, output_type = 'matrix'):
    """
    Calculates modal assurance criterion matrix.

    :param phi_1: modal matrix or modeshapes 1, shape: ``(n_locations, n_modes)``
    :type phi_1: array(float)
    :param phi_2: modal matrix or modeshapes 1, shape: ``(n_locations, n_modes)``
    :type phi_2: array(float)
    :param output_type: output type - 'matrix' or 'diagonal'
    :type output_type: str('matrix', 'diagonal')
    :return: MAC values
    """
    if phi_1.shape[0] != phi_2.shape[0]:
        raise Exception('Input dimensions are not compatible.')
    if phi_1.ndim == 1:
        phi_1 = phi_1[:,np.newaxis]
    if phi_2.ndim == 1:
        phi_2 = phi_2[:,np.newaxis]

    MAC_mat = (np.abs(np.einsum('ri,ik->rk',np.conj(phi_1).T,phi_2))**2 / (np.einsum('ri,ir->r',np.conj(phi_1).T,phi_1)[:,np.newaxis] * np.einsum('ri,ir->r',np.conj(phi_2).T,phi_2))).real
    if output_type == 'matrix':
        return MAC_mat
    if output_type == 'diagonal':
        return np.diagonal(MAC_mat)
    else:
        raise Exception('Unknown output type.')
    
def corelate_modes(A1, A2, cutoff=0.5):
    """
    Correlate modes between two matrices A1 and A2 using the Modal Assurance Criterion (MAC).
    Parameters:
    A1 (numpy.ndarray): The first matrix containing mode shapes.
    A2 (numpy.ndarray): The second matrix containing mode shapes.
    cutoff (float, optional): The threshold value for MAC to consider a mode as correlated. Default is 0.5.
    Returns:
    tuple: A tuple containing:
        - sorted_indices (numpy.ndarray): An array of tuples where each tuple contains the indices of correlated modes from A1 and A2.
        - matrix (numpy.ndarray): The MAC matrix computed from A1 and A2.
    """
    
    matrix = MAC(A1, A2)
    indices = []
    used_rows = set()
    used_cols = set()

    # Find all indices and their values above the cutoff
    candidates = [(i, j, matrix[i, j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] > cutoff]

    # Sort candidates by value in descending order to prioritize larger values
    candidates.sort(key=lambda x: x[2], reverse=True)

    # used_rows.add(3)
    # used_rows.add(5)
    for i, j, value in candidates:
        if i not in used_rows and j not in used_cols:
            indices.append((i, j))
            used_rows.add(i)
            used_cols.add(j)
    ind_sort = np.argsort(np.array(indices)[:,0])
    sorted_indices = np.array(indices)[ind_sort]
    
    return sorted_indices, matrix

def plot_MAC(A1, A2, label1 = 'Experimental mode', label2 = 'Numerical mode', title = 'MAC', cutoff = 0.7):
    fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)
    corelated, MAC_mat = corelate_modes(A1, A2, cutoff = cutoff)
    ax.imshow(MAC_mat, cmap='coolwarm', clim=[0,1])
    for i in range(A1.shape[1]):
        for j in range(A2.shape[1]):
            if MAC(A1,A2)[i,j] > 0.6:
                ax.text(j, i, f'{MAC(A1, A2)[i,j]:.2f}', ha='center', va='center', color='k', bbox=dict(facecolor='white', alpha=0.6), fontsize=8)
            else:
                ax.text(j, i, f'{MAC(A1, A2)[i,j]:.2f}', ha='center', va='center', color='k', bbox=dict(facecolor='white', alpha=0.6), fontsize=8)
    for i,j in corelated:
        ax.add_patch(plt.Circle([j,i], radius=0.5, edgecolor='k',lw=2, fill=False))
    ax.set_xlabel(label2)
    ax.set_ylabel(label1)
    ax.set_xticks(range(0,A2.shape[1]), range(1,A2.shape[1]+1))
    ax.set_yticks(range(0,A1.shape[1]), range(1,A1.shape[1]+1))
    plt.title(title)
    plt.tight_layout()
    # plt.savefig('MAC_T'+T+'.png')

def compare_freqs(f1,f2,A1,A2, limit_modes = 8, cutoff = 0.7, labels = ['Experimental', 'Predicted']):
    corelated, _ = corelate_modes(A1, A2, cutoff = cutoff)
    print('Corelated modes:', corelated)
    fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)
    barwidth = 0.35
    exp_ind = np.array([_[0] for _ in corelated])
    num_ind = np.array([_[1] for _ in corelated])
    exp_freq = f1[exp_ind]
    num_freq = f2[num_ind]
    ax.bar(np.arange(len(exp_freq))-barwidth/2, exp_freq, barwidth, label=labels[0])
    ax.bar(np.arange(len(num_freq))+barwidth/2, num_freq, barwidth, label=labels[1])
    ax.set_xticks(range(len(corelated)))
    ax.set_xticklabels(range(1,len(corelated)+1))
    ax.set_xlabel('Mode')
    ax.set_ylabel('Frequency [Hz]')
    ax.legend()
    plt.tight_layout()
    plt.show()
        