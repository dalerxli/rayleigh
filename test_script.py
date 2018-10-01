import numpy as np
import single_particle as sp

# get polarizability for an ellipsoid (a>b>c)
a = 2.
b = 0.5
c = 0.1
diel = complex(3.17, 0.025)
wavl = 32.1
k = 2.*np.pi/wavl
alpa, alpb, alpc = sp.ellipsoid_polz(diel, a, b, c)

# try rotations and get scattering matrix (phi_p rotation from x-axis)
phi_p = np.pi/4.
theta_p = np.pi/8.
phi_s = 0.
alp_rot = sp.point_spin(phi_p, theta_p, phi_s, alpa, alpb, alpc)

# incident field direction (azimuth from x-axis,
# eh for az_i=0 is along y axis, ev is along z axis)
el_i = 0.
az_i = 0.

# backscatter
el_s = -el_i
az_s = az_i+np.pi
smat = sp.scat_matrix(el_i, az_i, el_s, az_s, alp_rot, k)

# get scattering amplitudes and radar variables
print smat
shh = smat[0,0]
shv = smat[0,1]
svv = smat[1,1]
zdr = 10.*np.log10(np.abs(shh)**2./np.abs(svv)**2.)
ldr = 10.*np.log10(np.abs(shv)**2./np.abs(shh)**2.)
print zdr, ldr
