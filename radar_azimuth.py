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
thick = 0.02

# calculate polarizabilities
#alpha_a, alpha_a, alpha_c = sp.oblate_polz(eps_ice, dmax/2., thick/2.)
alpha_c, alpha_a, alpha_a = sp.prolate_polz(eps_ice, dmax/2., thick/2.)
print alpha_c, alpha_a

# set orientation distribution
amp = 51
ang_mode = 0.3
mod = (1.-np.cos(ang_mode))/2.
print mod
#mod = 0.2
a = (amp-2)*mod+1
b = amp*(1-mod)+2*mod-1
nphi = 201
phi = np.linspace(0., 2.*np.pi, nphi)
#angmom = ort.angmom_fixed(ang_mode, phi)
angmom = ort.angmom_beta_phi(a, b, phi)

#print angmom

# calculate radar variables
alpha_cov = ort.avg_polcov(angmom, alpha_a, alpha_c)
avar_hh = alpha_cov[0]
avar_vv = alpha_cov[1]
avar_hv = alpha_cov[2]
acov_hh_vv = alpha_cov[3]
acov_hh_hv = alpha_cov[4]
adp = alpha_cov[6]
zdr, ldr, kdp, rhohv, zh, rhoxh = ort.radar(wavl, avar_hh, avar_vv, avar_hv,
                                            acov_hh_vv, acov_hh_hv, adp)

# plot distribution
plt.figure(0)
v = (1.-np.cos(np.linspace(1.-2.*np.arccos(0.), 1.-2.*np.arccos(1.), 201)))/2.
theta = np.arccos(1.-2.*v)
beta = ort.beta_dist(v, a, b)

plt.plot(theta*180./np.pi, beta, 'b-', lw=3.)
ax = plt.gca()
#ax.set_ylim([0., 1.])
ax.set_xlim([0., 180.])
plt.savefig('dist.png')

# plot radar variables
fig = plt.figure(1)
ax = fig.add_subplot(2,2,1)
plt.plot(phi*180./np.pi, zdr, 'r-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,2)
plt.plot(phi*180./np.pi, ldr, 'r-', lw=3.)
ax.set_ylim([-45., 0.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,3)
plt.plot(phi*180./np.pi, rhohv, 'r-', lw=3.)
ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,4)
plt.plot(phi*180./np.pi, rhoxh, 'r-', lw=3.)
ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

print rhoxh

plt.savefig('radar_azimuth.png')
