# -*- coding: utf-8 -*-
"""
Comparison of anisotropy indices for transverse isotropic materials

@author: Nicolas Guarin-Zapata
"""
import numpy as np
import matplotlib.pyplot as plt


def aniso(C11, C12, C13, C33, C44):
    """Generalization of the Zener ratio to quantify anisotropy

    This generalization was presented in [1]_.

    References
    ----------
    .. [1] Kanit, T., N’Guyen, F., Forest, S., Jeulin, D., Reed, M., &
        Singleton, S. (2006). Apparent and effective physical
        properties of heterogeneous materials: Representativity of
        samples of two materials from food industry. Computer Methods
        in Applied Mechanics and Engineering, 195(33), 3960-3982.
    """
    Y44 = (C44 + C11 - C12)/3.0
    Y11 = (C11*2 + C33)/3.0
    Y12 = (C12 + C13*2)/3.0
    a = 2*Y44/(Y11 - Y12)
    return a


def christ_stiff(theta, C11, C12, C13, C33, C44):
    """Christophel stiffness for tensor C and angle theta
    """
    C66 = 0.5*(C11 - C12)
    M = ((C11 - C44)*np.sin(theta)**2 - (C33 - C44)*np.cos(theta)**2)**2 +\
        (C13 + C44)**2*np.sin(2*theta)**2
    vqP = np.sqrt(0.5*(C11*np.sin(theta)**2 + C33*np.cos(theta)**2 + C44 +
        np.sqrt(M)))
    vqS = np.sqrt(0.5*(C11*np.sin(theta)**2 + C33*np.cos(theta)**2 + C44 -
        np.sqrt(M)))
    vS = np.sqrt(C66*np.sin(theta)**2 + C44*np.cos(theta)**2)
    return vqP, vqS, vS


def aniso_ratio(C11, C12, C13, C33, C44):
    theta = np.linspace(0, np.pi/2, 300)
    aqP = np.zeros_like(C11)
    aqS = np.zeros_like(C11)
    aS = np.zeros_like(C11)
    for cont in range(len(C11)):
        vqP, vqS, vS = christ_stiff(theta, C11[cont], C12[cont], C13[cont],
                               C33[cont], C44[cont])
        mqP = np.mean(vqP)
        mqS = np.mean(vqS)
        mS = np.mean(vS)
        aqP[cont] = np.mean((vqP - mqP)/mqP)
        aqP[cont] = np.std(vqP/mqP)
        aqS[cont] = np.mean((vqS - mqS)/mqS)
        aqS[cont] = np.std(vqS/mqS)
        aS[cont] = np.mean((vS - mS)/mS)
        aS[cont] = np.std(vS/mS)
    return aqP, aqS, aS


if __name__ == "__main__":
    plt.style.use("seaborn")
    fpath = "../transverse_isotropic.tsv"
    cols = range(2,12)
    mats = np.loadtxt(fpath, skiprows=1, usecols=(0,), dtype=str)
    props = np.loadtxt(fpath, skiprows=1, usecols=cols, delimiter='\t')
    C11 = 0.5*(props[:, 0] + props[:, 1])
    C12 = 0.5*(props[:, 2] + props[:, 3])
    C13 = 0.5*(props[:, 4] + props[:, 5])
    C33 = 0.5*(props[:, 6] + props[:, 7])
    C44 = 0.5*(props[:, 8] + props[:, 9])
    a = aniso(C11, C12, C13, C33, C44)
    aqP, aqS, aS = aniso_ratio(C11, C12, C13, C33, C44)
    x = np.array(range(len(a)))
    plt.bar(x, np.abs(1 - a)/np.max(np.abs(1 - a)), align='center',
            width=0.4)
    plt.bar(x + 0.4, (aqP + aqS + aS)/3/np.max((aqP + aqS + aS)/3),
            align='center', width=0.4)
    plt.xticks(x, mats, rotation='vertical')
    plt.xlim(-0.5, len(a) - 0.5)
    plt.ylabel("Anisotropy index")
    plt.tight_layout()
    plt.savefig("aniso_index.png", dpi=300)
    plt.show()