import numpy as np
import single_particle as sp

# get polarizability for an ellipsoid (a>b>c)
a = 2.
b = 0.05
c = 0.02
diel = complex(3.17, 0.025)
wavl = 32.1
k = 2.*np.pi/wavl
alpa, alpb, alpc = sp.ellipsoid_polz(diel, a, b, c)

# incident and scattered field directions
el_i = np.pi/2.
az_i = 0.
el_s = -el_i
az_s = az_i+np.pi

# calculate scattering amplitudes given distribution of orientations
npar = 2000
smat = np.empty([npar,2,2], dtype=complex)
phi_p = np.random.rand(npar)*2.*np.pi
theta_p = 0.
phi_s = 0.

for i in range(npar):
    alp_rot = sp.point_spin(phi_p[i], theta_p, phi_s, alpa, alpb, alpc)
    smat[i,:,:] = sp.scat_matrix(el_i, az_i, el_s, az_s, alp_rot, k)

# get scattering amplitudes and radar variables
vhh2 = np.sum(np.abs(smat[:,0,0])**2.)
vhv2 = np.sum(np.abs(smat[:,0,1])**2.)
vvv2 = np.sum(np.abs(smat[:,1,1])**2.)
c_hh_vv = np.sum(smat[:,0,0]*np.conj(smat[:,1,1]))
c_hv_hh = np.sum(smat[:,0,1]*np.conj(smat[:,0,0]))

zhh = 10.*np.log10(4.*wavl**4./(np.pi**4.*0.93)*vhh2)
zhv = 10.*np.log10(4.*wavl**4./(np.pi**4.*0.93)*vhv2)
zvv = 10.*np.log10(4.*wavl**4./(np.pi**4.*0.93)*vvv2)
zdr = zhh-zvv
ldr = zhv-zhh
rhohv = np.abs(c_hh_vv/(np.sqrt(vhh2*vvv2)))
rhohx = np.abs(c_hv_hh/(np.sqrt(vhh2*vhv2)))
print zhh, zdr, ldr, rhohv, rhohx
