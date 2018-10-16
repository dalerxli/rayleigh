import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import orientation as ort
import single_particle as sp

# cosine transform
def fc(v):
    ang = np.arccos(1.-2.*v)
    return ang

# set particle and radar properties
wavl = 32.1
eps_ice = complex(3.16835, 0.0089)
dmax = 0.2
thick = 0.01

# calculate polarizabilities
alpha_a, alpha_a, alpha_c = sp.oblate_polz(eps_ice, dmax/2., thick/2.)

# set orientation distribution
amp1 = 41
mod1 = 0.1
a1 = (amp1-2)*mod1+1
b1 = amp1*(1-mod1)+2*mod1-1
w1 = 1.

amp2 = 41
mod2 = 0.05
a2 = (amp2-2)*mod2+1
b2 = amp2*(1-mod2)+2*mod2-1
w2 = 1.-w1

#a = 7.
#b = 7.
sig_deg = 180.*np.sqrt(2./(b1-1.))/np.pi
#sig_deg = 35.
print sig_deg
angmom1 = ort.angmom_beta(a1, b1)
angmom2 = ort.angmom_beta(a2, b2)
angmom = (w1*np.array(angmom1)+w2*np.array(angmom2))
angmom_g = ort.angmom_gauss(sig_deg)
print angmom_g
print angmom1

# calculate radar variables
avar_hh, avar_vv, avar_hv, acov_hh_vv, adp = ort.avg_polcov(angmom, alpha_a, alpha_c)
zdr, ldr, kdp, rhohv, zh = ort.radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, adp)
print zdr, ldr, kdp, rhohv, zh

avar_hh, avar_vv, avar_hv, acov_hh_vv, adp = ort.avg_polcov(angmom_g, alpha_a, alpha_c)
zdr, ldr, kdp, rhohv, zh = ort.radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, adp)
print zdr, ldr, kdp, rhohv, zh

# plot beta distribution
x = (1.-np.cos(np.linspace(fc(0.), fc(1.), 201)))/2.
ang = np.arccos(1.-2.*x)*180./np.pi
beta_av = (w1*ort.beta_dist(x, a1, b1)+w2*ort.beta_dist(x, a2, b2))
#plt.plot(x, beta_av, 'r--', lw=3.)
plt.plot(ang, beta_av, 'r--', lw=3.)
ax = plt.gca()
ax.set_ylim([0., np.max(beta_av)*1.1])
ax.set_xlim([0., 180.])
ax.set_xlabel('Angle from zenith (degrees)')
plt.savefig('beta.png', dpi=45)
