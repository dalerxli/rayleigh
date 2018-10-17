import numpy as np
from scipy.special import beta
import matplotlib.pyplot as plt

# beta distribution
def beta_dist(x, a, b):
    f = x**(a-1.)*(1.-x)**(b-1.)/beta(a,b)
    return f

# gaussian distribution
def gauss_dist(x, sig_deg):
    f = 2./(sig_deg*np.sqrt(2.*np.pi))*np.exp(-x**2./(2.*sig_deg**2.))
    return f

# integral function for beta distribution
def i_n(n, a, b):
    ival = beta(a+n,b)/beta(a,b)
    return ival

# half-integer function for beta distribution
def j_n(n, a, b):
    jval = beta(a+float(n)/2.,b)/beta(a,b)
    return jval

# fixed-angle (delta distribution) moments
def angmom_fixed(theta, phi):
    ct = np.cos(theta)
    cp = np.cos(phi)
    st = np.sin(theta)
    s2t = np.sin(2.*theta)
    sp = np.sin(phi)

    a1 = ct**2.
    a2 = st**2.*sp**2.
    a3 = ct**4.
    a4 = st**4.*sp**4.
    a5 = sp**2.*st**2.*ct**2.
    a6 = sp*s2t
    a7 = sp*ct**2.*s2t
    a8 = sp**3.*st**2.*s2t
    
    angmom = [a1, a2, a3, a4, a5, a6, a7, a8]
    return angmom

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

# angular moments for beta distribution in theta, fixed phi
def angmom_beta_phi(a, b, phi):
    # 'integer' moments
    i1 = i_n(1,a,b)
    i2 = i_n(2,a,b)
    i3 = i_n(3,a,b)
    i4 = i_n(4,a,b)

    a1 = 4*(i2-i1)+1.
    a2 = 4*np.sin(phi)**2.*(i1-i2)
    a3 = 8*(2*i4-4*i3+3*i2-i1)+1.
    a4 = 16*np.sin(phi)**4.*(i4-2*i3+i2)
    a5 = 4*np.sin(phi)**2.*(-4*i4+8*i3-5*i2+i1)

    # 'half-integer' moments
    j1 = j_n(1,a,b)
    j3 = j_n(3,a,b)
    j5 = j_n(5,a,b)
    j7 = j_n(7,a,b)
    j9 = j_n(9,a,b)
    j11 = j_n(11,a,b)
    j13 = j_n(13,a,b)
    j15 = j_n(15,a,b)
    j17 = j_n(17,a,b)
    j19 = j_n(19,a,b)

    a6 = np.sin(phi)*(4*j1-8*j3-2*j5+4*j7-j9/2.+j11-j13/4.+j15/2.)
    a26 = 4*(4*j3-12*j5+6*j7+6*j9-9./2.*j11+3./2.*j13-5./4.*j15+3./4.*j17+j19/2.)
    a7 = a6-np.sin(phi)*a26
    a8 = np.sin(phi)**3.*a26

    angmom = [a1, a2, a3, a4, a5, a6, a7, a8]
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
    a6 = angmom[5]
    a7 = angmom[6]
    a8 = angmom[7]
    j1 = np.abs(alpha_a-alpha_c)**2.
    j2 = np.conj(alpha_a)*(alpha_a-alpha_c)

    # calculate covariance matrix elements (+kdp-like element)
    avar_hh = np.abs(alpha_a)**2.-2.*np.real(j2)*a2+j1*a4
    avar_vv = np.abs(alpha_a)**2.-2.*np.real(j2)*a1+j1*a3
    avar_hv = j1*a5
    acov_hh_vv = np.abs(alpha_a)**2.+j1*a5-j2*a1-np.conj(j2)*a2
    acov_hh_hv = 0.5*(j1*a8-np.conj(j2)*a6)
    acov_vv_hv = 0.5*(j1*a7-np.conj(j2)*a6)
    adp = np.real(alpha_a-alpha_c)*(a1-a2)

    return [avar_hh, avar_vv, avar_hv, acov_hh_vv, acov_hh_hv, acov_vv_hv, adp]

# calculate radar variables from covariances
def radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, acov_hh_hv, adp):
    zhh = 4.*wavl**4./(np.pi**4.*0.93)*avar_hh
    zvv = 4.*wavl**4./(np.pi**4.*0.93)*avar_vv
    zhv = 4.*wavl**4./(np.pi**4.*0.93)*avar_hv

    zdr = 10.*np.log10(zhh/zvv)
    ldr = 10.*np.log10(zhv/zhh)
    kdp = 180.*1.e-3/wavl*adp
    rhohv = np.abs(acov_hh_vv/np.sqrt(avar_hh*avar_vv))
    rhoxh = np.abs(acov_hh_hv/np.sqrt(avar_hh*avar_hv))

    return zdr, ldr, kdp, rhohv, 10.*np.log10(zhh), rhoxh
