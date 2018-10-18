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
thick = 0.001

# calculate polarizabilities
#alpha_a, alpha_a, alpha_c = sp.oblate_polz(eps_ice, dmax/2., thick/2.)
alpha_c, alpha_a, alpha_a = sp.prolate_polz(eps_ice, dmax/2., thick/2.)
print alpha_c, alpha_a

# set orientation distribution
amp = 71
ang_mode = np.pi*0.
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
acov_vv_hv = alpha_cov[5]
adp = alpha_cov[6]
zdr, ldr, kdp, rhohv, zh, rhoxh = ort.radar(wavl, avar_hh, avar_vv, avar_hv,
                                            acov_hh_vv, acov_hh_hv,
                                            acov_vv_hv, adp)

# compare to simultaneous tr
psi_dp = 0.5*np.pi
svar_h, svar_v, scov_hv = ort.cov_sim(alpha_cov, psi_dp)
zdr_str, ldrj, kdpj, rhohv_str, zh_str, rhoxhj = ort.radar(wavl, svar_h, svar_v,
                                              avar_hv, scov_hv, acov_hh_hv,
                                              acov_vv_hv, adp)
phi_dp = 180./np.pi*np.angle(scov_hv)
# plot distribution
plt.figure(0)
v = np.linspace(0., 1., 1000)
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
plt.plot(phi*180./np.pi, zdr_str, 'b-', lw=3.)
#ax.set_ylim([0., 1.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,2)
plt.plot(phi*180./np.pi, phi_dp, 'r-', lw=3.)
ax.set_ylim([-180., 0.])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,3)
plt.plot(phi*180./np.pi, rhohv, 'r-', lw=3.)
plt.plot(phi*180./np.pi, rhohv_str, 'b-', lw=3.)
ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

ax = fig.add_subplot(2,2,4)
plt.plot(phi*180./np.pi, rhoxh, 'r-', lw=3.)
ax.set_ylim([0., 1.1])
ax.set_xlim([0., 360.])

plt.savefig('radar_azimuth.png')
