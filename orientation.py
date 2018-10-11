import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt

# beta distribution
def beta_dist(x, a, b):
    f = x**(a-1.)*(1.-x)**(b-1.)/beta(a,b)
    return f

# integral function for beta distribution
def i_n(n, a, b):
    ival = beta(a+n,b)/beta(a,b)
    return ival

# calculate angular moments for a beta distribution
def angmom_beta(a, b):
    i1 = i_n(1,a,b)
    i2 = i_n(2,a,b)
    i3 = i_n(3,a,b)
    i4 = i_n(4,a,b)

    a1 = 4*(i2-i1)+1.
    a2 = 2*(i1-i2)
    a3 = 8*(2*i4-4*i3+3*i2-i1)+1.
    a4 = 6*(i4-2*i3+i2)
    a5 = 2*(-4*i4+8*i3-5*i2+i1)
    angmom = [a1, a2, a3, a4, a5]
    return angmom

def angmom_gauss(sig_deg):
    sig_rad = sig_deg*np.pi/180.
    r = np.exp(-2.*sig_rad**2.)
    a1 = (1./4.)*(1.+r)**2.
    a2 = (1./4.)*(1.-r**2.)
    a3 = (3./8.+1./2.*r+1./8*r**4.)**2.
    a4 = (3./8.-1./2.*r+1./8*r**4.)*(3./8.+1./2.*r+1./8*r**4.)
    a5 = 1./8.*(3./8.+1./2.*r+1./8*r**4.)*(1.-r**4.)
    a6 = 0.
    a7 = 1./2*r*(1.+r)
    angmom = [a1, a2, a3, a4, a5, a6, a7]
    return angmom

# calculate averaged polarizability covariances symmetric particle
def avg_polcov(angmom, alpha_a, alpha_c):
    # get angular moments and select j terms
    a1 = angmom[0]
    a2 = angmom[1]
    a3 = angmom[2]
    a4 = angmom[3]
    a5 = angmom[4]
    j1 = np.abs(alpha_a-alpha_c)**2.
    j2 = np.conj(alpha_a)*(alpha_a-alpha_c)

    # calculate covariance matrix elements (+kdp-like element)
    avar_hh = np.abs(alpha_a)**2.-2.*np.real(j2)*a2+j1*a4
    avar_vv = np.abs(alpha_a)**2.-2.*np.real(j2)*a1+j1*a3
    avar_hv = j1*a5
    acov_hh_vv = np.abs(alpha_a)**2.+j1*a5-j2*a1-np.conj(j2)*a2
    adp = np.real(alpha_a-alpha_c)*(a1-a2)

    return avar_hh, avar_vv, avar_hv, acov_hh_vv, adp

# calculate radar variables from covariances
def radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, adp):
    zhh = 4.*wavl**4./(np.pi**4.*0.93)*avar_hh
    zvv = 4.*wavl**4./(np.pi**4.*0.93)*avar_vv
    zhv = 4.*wavl**4./(np.pi**4.*0.93)*avar_hv

    zdr = 10.*np.log10(zhh/zvv)
    ldr = 10.*np.log10(zhv/zhh)
    kdp = 180.*1.e-3/wavl*adp
    rhohv = np.abs(acov_hh_vv/np.sqrt(avar_hh*avar_vv))

    return zdr, ldr, kdp, rhohv, 10.*np.log10(zhh)
