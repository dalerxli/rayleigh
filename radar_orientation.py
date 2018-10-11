import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import orientation as ort
import single_particle as sp

# set particle and radar properties
wavl = 32.1
eps_ice = complex(3.16835, 0.0089)
dmax = 0.2
thick = 0.01

# calculate polarizabilities
alpha_a, alpha_a, alpha_c = sp.oblate_polz(eps_ice, dmax/2., thick/2.)

# set orientation distribution
amp1 = 61
mod1 = 0.01
a1 = (amp1-2)*mod1+1
b1 = amp1*(1-mod1)+2*mod1-1

amp2 = 91
mod2 = 0.03
a2 = (amp2-2)*mod2+1
b2 = amp2*(1-mod2)+2*mod2-1

#a = 7.
#b = 7.
sig_deg = 14.97
angmom1 = ort.angmom_beta(a1, b1)
angmom2 = ort.angmom_beta(a2, b2)
angmom = (np.array(angmom1)+np.array(angmom2))/2.
angmom_g = ort.angmom_gauss(sig_deg)

# calculate radar variables
avar_hh, avar_vv, avar_hv, acov_hh_vv, adp = ort.avg_polcov(angmom, alpha_a, alpha_c)
zdr, ldr, kdp, rhohv, zh = ort.radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, adp)
print zdr, ldr, kdp, rhohv, zh

avar_hh, avar_vv, avar_hv, acov_hh_vv, adp = ort.avg_polcov(angmom_g, alpha_a, alpha_c)
zdr, ldr, kdp, rhohv, zh = ort.radar(wavl, avar_hh, avar_vv, avar_hv, acov_hh_vv, adp)
print zdr, ldr, kdp, rhohv, zh

# plot beta distribution
x = np.linspace(0., 1., 201)
beta_av = 0.5*(ort.beta_dist(x, a1, b1)+ort.beta_dist(x, a2, b2))
plt.plot(x, beta_av, 'r--', lw=3.)
ax = plt.gca()
ax.set_ylim([0., np.max(beta_av)*1.1])
ax.set_xlim([0., 1.])
plt.savefig('beta.png', dpi=45)
