import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
import gadget as gad
import gadget_subfind as gad_sf
import const as c
import sys, h5py, csv, cgm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.legend_handler import HandlerTuple
import cmocean as cmo
from statistics import median
from matplotlib.ticker import AutoMinorLocator
'''
snap_string = input('Enter the snapshot number of the halo to study \n')
snap: int = snap_string.split()
halo_string = input('Enter the names of the haloes to analyse separated by a space \n')
if halo_string == f"all":
	n = input('How many haloes are present to analyse?\n')
	halo = [f'halo_{i}' for i in range(int(n))]
else:
	halo = halo_string.split()
halo_d_string = input('Enter the names of the directory to analyse the above halo separated by a space \n')
halo_d = halo_d_string.split()
location_string = input('Enter the system that contains the directory that contains the above haloes \n')
location = location_string.split()
print(location)
'''
halo_d = ['level3_MHD_10', 'level3_MHD_1e11', 'level3_MHD_new', 'level3_MHD']
location_string = 'universe/spxfv/Auriga'

haloid = 0

x1, x2 = [], []
cooltemp, halomass = [], []
Temperature, Mass = [], []

#Constants
Zsolar: float		= 0.0127
omegabaryon0: float = 0.048
m_p: float			= 1.67e-24
k_b: float			= 1.38e-16
Mpcincm: float		= c.parsec*1e6
G: float			= 6.67e-8
H_0: float			= 2.17e-18
df = h5py.File('fg2009_ss_hr.h5')

for j in range(len(halo_d)):
	if halo_d[j] == 'level3_MHD_10':
		halo = ['halo_0', 'halo_2', 'halo_6', 'halo_8', 'halo_9', 'halo_11']
		snap = [251]
	if halo_d[j] == 'level3_MHD_1e11':
		halo = ['halo_' + str(n) for n in range(0,12)]
#		halo = ['halo_10', 'halo_11', 'halo_6', 'halo_0', 'halo_9', 'halo_8', 'halo_2', 'halo_7', 'halo_1', 'halo_3', 'halo_5', 'halo_4']
	if halo_d[j] == 'level3_MHD_new':
		halo = ['halo_L8']
		snap = [127]
	if halo_d[j] == 'level3_MHD':
		halo = ['halo_6', 'halo_16', 'halo_23', 'halo_24', 'halo_21', 'halo_27']
		snap = [63]
	for i in range(len(snap)):
		for k in range(len(halo)):
			sf      = gad_sf.load_subfind(int(snap[i]), dir = f'/home/{location_string}/{halo_d[j]}/%s/output/' % halo[k])
			load    = gad.gadget_readsnap(int(snap[i]), snappath = f'/home/{location_string}/{halo_d[j]}/%s/output/' % halo[k], loadonlytype=[0,4], loadonlyhalo = 0, lazy_load=True, subfind = sf)

			stellarBirthRedshift, stellarInitMass	= [], []

			#Rotate haloes such that the stellar angular momentum vecotr is vertical
			load.calc_sf_indizes(sf)
			load.select_halo(sf, do_rotation = True)

			#Define and convert quantities
			load.pos *= 1e3  #Convert position of the cells from Mpc to kpc
			load.vol *= 1e9  #Convert Mpc^3 to kpc^3
			load.mass *= 1e10    #Convert mass from Msol/1e10 to Msol

			load.data['dist']: float = np.sqrt((load.pos**2).sum(axis=1))  #Magnitude of the position vector for each cell giving distance from the centre of the halo
			virialRadius: float = sf.data['frc2'][0]*1e3
			virialMass: float = sf.data['fmc2'][0]*1e10
			cent: float = sf.data['fpos'][haloid,:]
			normRadius: float = load.data['dist']/virialRadius
			halomass.append(virialMass)

			#Information on the chemistry of elements in the Auriga simulations. Including r-process elements.
			#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13		14
			elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb',	'S']
			elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82, 	16]
			elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2, 32.06]
			elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75,	]

			rhocrit = 3. * (load.omega0 * (1+load.redshift)**3. + load.omegalambda) * (load.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * np.pi * c.G)
			rhomean = 3. * (load.omega0 * (1+load.redshift)**3.) * (load.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * np.pi * c.G)
			meanweight = np.sum(load.gmet[:,0:9][:len(load.ne)], axis = 1) / ( np.sum(load.gmet[:,0:9][:len(load.ne)]/elements_mass[0:9], axis = 1) + load.ne*load.gmet[:,0][:len(load.ne)] )

			#SFH of the halo
			whereStars,	= np.where(load.age > 0)
			stellarAge: float	= load.age[whereStars]	#Time when stars formed. load.age > 0 added to exclude wind phase gas cells. This is in terms of the scale factor
			birthRedshift: float = (1/stellarAge) - 1			#stellarAge given in units of the scale factor. Therefore (1/a)-1=z used
			birthMass = np.multiply(load.data['gima'][whereStars], 1e10)	#Mass of stars once born
			stellarBirthRedshift.append((1/stellarAge) - 1)	#
			stellarInitMass.append(np.multiply(load.data['gima'][whereStars], 1e10))

			stellarInitMass: float = np.array(stellarInitMass)

			#Angle for plotting virial radius on projections
			theta: float = np.linspace(0, 2*np.pi, 100)

			#Draw the virial radius of the halo
			x1.append(virialRadius*np.sin(theta))
			x2.append(virialRadius*np.cos(theta))

			Tfac: float = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53 #converts from internal energy to temperature

			#Define physical properties
			load.bfld: float			= np.multiply(load.bfld, np.multiply(c.bfac,1e6))																	#Converts to microG
			load.data['bfld']: float	= abs(np.sqrt((load.bfld**2).sum(axis=1)))																			#Magnitude of bfield vectors, microGaus
			load.data['dense']: float	= load.rho/(np.multiply(c.parsec, 1e6)) ** 3 * np.multiply(c.msol, 1e10)											#Density of cells in g/cm^3
			load.data['T']: float		= np.divide(load.u, Tfac)																							#Temperature in K
			load.data['gz']: float		= abs(np.divide(load.gz, 0.0127))																					#Metallicity normalised to solar metallicity
			load.data['thermP']: float	= np.divide(np.multiply(load.data['dense'], load.data['T']), np.multiply(meanweight, m_p))							#Thermal pressure in Pascals/k_b
			load.data['bfldP']: float	= ((load.data['bfld']/1e6)**2)/(8*np.pi*k_b)																		#Magnetic pressure also in Pascals/k_b
			load.data['totP']: float	= np.add(load.data['thermP'], load.data['bfldP'])																	#Total pressure
			load.data['n_H']: float		= np.divide(np.multiply(load.data['dense'], load.gmet[:,0][:len(load.ne)]), np.multiply(elements_mass[0], c.amu))	#Number density of hydrogen
			load.data['pRat']: float	= np.divide(load.data['thermP'], load.data['bfldP'])																#Pressure ratio
			load.data['coolingRate']: float	= load.gcol[load.gcol<0]
			load.data['heatingRate']: float = load.gcol[load.gcol>0]

			cooltemp.append(median(load.mass[:len(load.ne)][(load.data['T'] >= 1e4) & (load.data['T'] <= 10**4.25) & (normRadius[:len(load.data['T'])] <= 1)]))
			Temperature.append(np.log10(load.data['T']))
			Mass.append(load.mass)

			#------------- COLUMN DENSITY -------------#

			load.data['n_HI']									= ((load.data['dense'][:len(load.ne)] / (elements_mass[0]*c.amu)) * load.gmet[:,0][:len(load.ne)]) * load.data['nh'][:len(load.ne)]
			load.data['n_SiI'], SiI_mass, SiI_mass_frac			= cgm.ionisation('Si', 7, 0, load.redshift, df, elements_mass, load = load)
			load.data['n_SiII'], SiII_mass, SiII_mass_frac		= cgm.ionisation('Si', 7, 1, load.redshift, df, elements_mass, load = load)
			load.data['n_SiIII'], SiIII_mass, SiIII_mass_frac	= cgm.ionisation('Si', 7, 2, load.redshift, df, elements_mass, load = load)
			load.data['n_CIV'], CIV_mass, CIV_mass_frac			= cgm.ionisation('C', 2, 3, load.redshift, df, elements_mass, load = load)
			load.data['n_OVI'], OVI_mass, OVI_mass_frac			= cgm.ionisation('O', 4, 5, load.redshift, df, elements_mass, load = load)
			load.data['n_OVII'], OVII_mass, OVII_mass_frac		= cgm.ionisation('O', 4, 6, load.redshift, df, elements_mass, load = load)
			load.data['n_OVIII'], OVIII_mass, OVIII_mass_frac	= cgm.ionisation('O', 4, 7, load.redshift, df, elements_mass, load = load)

			plt.figure(1)
			plt.hist(np.log10(load.data['T'][(load.sfr == 0) & (normRadius[:len(load.ne)] >= 0.3) & (normRadius[:len(load.ne)] <= 1) & (load.halo[:len(load.ne)] == 0) & (load.subhalo[:len(load.ne)] <= 0)]), weights = load.mass[:len(load.data['T'][(load.sfr == 0) & (normRadius[:len(load.ne)] >= 0.3) & (normRadius[:len(load.ne)] <= 1) & (load.halo[:len(load.ne)] == 0) & (load.subhalo[:len(load.ne)] <= 0)])], density = True, bins = 100, range = [3.5,7])
			plt.grid()
			plt.ylabel(r'PDF')
			plt.xlabel(r'Halo Mass [M$_{\odot}$]')
			plt.savefig(f'/home/universe/c1537815/python/Figures/Paper1Plots/Histo_of_each_halo_temp__{halo_d[j]}_{halo[k]}.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')
			plt.close('all')

plt.figure(2)
plt.scatter(np.log10(halomass), np.log10(cooltemp), c = 'k', facecolors = 'tab:red')
plt.grid()
plt.ylabel(r'Cool temp mass < $10^{4.25}$ [K]')
plt.xlabel(r'Halo Mass [M$_{\odot}$]')
plt.savefig(f'/home/universe/c1537815/python/Figures/Paper1Plots/cool_gas_in_haloes.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')