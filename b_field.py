from gadget import *
from gadget_subfind import *
import h5py
import matplotlib.pyplot as plt
import const as c
import numpy as np
from sys import argv
from statistics import median
from scipy.interpolate import griddata
import sys
import matplotlib as mpl
from Functions import ionisation
import cmocean as cmo
from matplotlib.legend_handler import HandlerTuple

plt.switch_backend('agg')

#snap = [i for i in range(50, 128, 1)]	#z = 0 @ snap = 127, z = 0.5 @ snap = 95, z = 1 @ snap = 77, z = 2 @ snap = 61
snap	= [127]
halo	= ['h12_nobfield', 'h12_standard', 'h9_nobfield', 'h9_standard', 'h8_nobfield', 'h8_standard', 'h11_nobfield', 'h11_standard']
halo_d	= ['level2_cgm_1e10', 'level3_cgm_1e11', 'level4_cgm', 'level4_cgm_1e13']
halo = ['h8_nobfield', 'h8_standard', 'h9_nobfield', 'h9_standard']
haloid	= [0]

kappa_diff, kappa_med, kappa_radii									= [], [], []
kappa_stars, kappa_gas											= [], []
diskRadii = []
gcol = []
v_out_m, v_in_m, met, met_m, dense, dense_m, temp, temp_m, T_dens					= [], [], [], [], [], [], [], [], []
v_out_v, v_in_v, met, met_v, dense, dense_v, temp, temp_v, T_dens					= [], [], [], [], [], [], [], [], []
vel_med, vupper, vlower 										= [], [], []

OVI, CIV, SiII, SiII, HI										= [], [], [], [], []

birth, birthx, birth_sum, birth_x = [], [], [], []

vupper_out_m, vlower_out_m, vupper_in_m, vlower_in_m,  mupper, mlower, dupper, dlower, tupper, tlower	= [], [], [], [], [], [], [], [], [], []
vlow_in_m, vlow_out_m, vup_in_m, vup_out_m, mupper_m, mlower_m, dupper_m, dlower_m, tupper_m, tlower_m	= [], [], [], [], [], [], [], [], [], []
vlow_in_v, vlow_out_v, vup_in_v, vup_out_v, mupper_v, mlower_v, dupper_v, dlower_v, tupper_v, tlower_v	= [], [], [], [], [], [], [], [], [], []

t_med, t_upper, t_lower, t_upper2, t_lower2, t_radii							= [], [], [], [], [], []
tp_med, tp_upper, tp_lower, tp_upper2, tp_lower2, tp_radii						= [], [], [], [], [], []
d_med, d_upper, d_lower, d_upper2, d_lower2, d_radii							= [], [], [], [], [], []
m_med, m_upper, m_lower, m_upper2, m_lower2, m_radii							= [], [], [], [], [], []
pr_med, pr_upper, pr_lower, pr_upper2, pr_lower2, pr_radii						= [], [], [], [], [], []
totpr_med, totpr_upper, totpr_lower, totpr_upper2, totpr_lower2, totpr_radii				= [], [], [], [], [], []
b_med, b_upper, b_lower, b_upper2, b_lower2, b_radii							= [], [], [], [], [], []
v_med, v_upper, v_lower, v_upper2, v_lower2, v_radii							= [], [], [], [], [], []
tde_med, tde_upper, tde_lower, tde_radii								= [], [], [], []
dmin_med, dmin_upper, dmin_lower, dmin_upper2, dmin_lower2, dmin_radii							= [], [], [], [], [], []
dmout_med, dmout_upper, dmout_lower, dmout_upper2, dmout_lower2, dmout_radii							= [], [], [], [], [], []
presrat_med, presrat_lower, presrat_upper = [], [], []

medPresRat, loPresRat, upPresRat = [], [], []

temp_m, temp_upper, temp_lower										= [], [], []
dense_v, dense_upper, dense_lower									= [], [], []
met_m, met_upper, met_lower											= [], [], []
pres_m, pres_upper, pres_lower										= [], [], []
vrad_m, vrad_upper, vrad_lower										= [], [], []

totalStellarMass, virialCellGasMass, totalHaloMass, radiusInVirial, totalRadius, virialCellStellarMass					= [], [], [], [], [], []
radialVelocityHaloMass, radialVelocityRadius, radialVelocity															= [], [], []
pressureRatio, massFlux, bField, totalPressure, thermalPressure, magneticPressure, hydrogenDensity, T, Z, Tvir, dTvir, stellarBirthRedshift, stellarInitMass	= [], [], [], [], [], [], [], [], [], [], [], [], []

colour													= []

n_HI, n_CIV, n_OVI, n_SiII, n_SiII									= [], [], [], [], []
n_HI_radii, n_CIV_radii, n_OVI_radii, n_SiII_radii, n_SiII_radii					= [], [], [], [], []

H_med, H_lower, H_upper, H_radii									= [], [], [], []
C_med, C_lower, C_upper, C_radii									= [], [], [], []
O6_med, O6_lower, O6_upper, O6_radii									= [], [], [], []
Mg_med, Mg_lower, Mg_upper, Mg_radii									= [], [], [], []
C1_med, C1_lower, C1_upper, C1_radii									= [], [], [], []
cFracHI, cFracCIV, cFracOVI, cFracSiII									= [], [], [], []

Hx, Hy, Hgrid, ismgrid										= [], [], [], []

HIprojmed_2, CIVprojmed_2, OVIprojmed_2, SiIIprojmed_2, SiIIprojmed_2					= [], [], [], [], []
HIprojrad_2, CIVprojrad_2, OVIprojrad_2, SiIIprojrad_2, SiIIprojrad_2					= [], [], [], [], []

Mass, dT, dRho, haloes, meanMolecularWeight, meanMolecularWeightUp, meanMolecularWeightLow		= [], [], [], [], [], [], []

#A few cheeky constants
m_p			= 1.67e-24
k_b			= 1.38e-16
Mpcincm		= c.parsec*1e6
G			= 6.67e-8
H_0			= 2.17e-18

# ============ FUNCTIONS ============ #
#Calculate the virial temperature - the average temperature of a gravitationally bound system.
def virialTemp(snap, halo, halo_d):

	#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
	elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
	elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
	elementsMass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
	elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

	m_h = 1.67e-24
	k = 1.38e-16
	G = 6.67e-8
	H_0 = 2.17e-18
	m_halo = subload.data['fmc2'][0]*1.989e43
	meanweight = sum(load.gmet[:,0:9], axis = 1) / ( sum(load.gmet[:,0:9]/elementsMass[0:9], axis = 1) + load.ne*load.gmet[:,0] )
	meanweight = sum(meanweight)/len(meanweight)

	print(meanweight)
	print(m_halo)

	T_vir = ((G**2)*(H_0**2)*(load.omega0)*18*(np.pi**2)/54)**(1/3)*(meanweight*m_h/k)*((m_halo**(2/3))*(1+load.redshift))
	return T_vir

#Interpolation algorithm that matches temperature, density and redshift data from simulations to modelled data. Provides insight into the ion fraction of simulations
def ionisation(element, mass, ion):
	table = np.array(list(df[element][ion]))

	n_H, z, T = [n for n in np.arange(-9, 2.125, 0.125)], 0, np.arange(1, 9.025, 0.025).round(3)	#Initialise length of n_H, z and T arrays

	iontable = table[:,z,:]									#iontable selects all data from table for specific element and ion with z in this case being constant as we are mainly analysing data at z = 0

	logdens, logtemp = np.repeat(np.array(n_H), len(T)), np.tile(np.array(T), len(n_H))	#Create repeating and tiled arrays in opposite orientations

	#Interpolate. As data from ion tables are logged, so much n_H and T
	linetable = 10**(griddata((logdens, logtemp),							#Using arrays from above to save interpolated data to
					iontable.flatten(),						#Flatten ion table data into 1-D array
					(np.log10(load.data['n_H']), np.log10(load.data['T']))))	#T and n_H used as metrics for interpolation

	nIon = abs(((load.data['dense'] / (elementsMass[mass]*c.amu)) * load.gmet[:,mass][:len(load.ne)]) * linetable)	#Number density of specific species of element (CIII, SiVIII etc)
	print(nIon)
	totalIonMass = np.float64(load.data['mass'][:len(load.ne)]*load.gmet[:,mass][:len(load.ne)]*linetable)		#Total mass of the ion in the halo

	totalIonMassFrac = linetable * load.gmet[:,mass][:len(load.ne)]		#Mass fraction of the ion in the halo

	return nIon, totalIonMass, totalIonMassFrac	#nIon is what you need for column density, totalIonMass gives the mass of the specific ion in each cell, totalMassFrac gives the ion fraction in each cell

def med_per(med, per1, per2, ion):

	med.append(median(ion))
	per1.append(median(ion) - np.nanpercentile(ion, 16))
	per2.append(np.nanpercentile(ion, 84) - median(ion))

#Binning function. Bins can be created along any property, mostly used for radial dependence of properties
def avg_weighted(x, y, min, max, s, inc, weight):	#x and y are datasets, min and max is the minimum and maxixum x value to measure from and to, s is the starting position of the data, inc is the total
							#number of increments you want (i.e. 200 would repeat the loop 200 times), and weight is your data to weigh if any
	med, upper, lower, radii = np.zeros(inc), np.zeros(inc), np.zeros(inc), np.zeros(inc)

	nbins = (max-min)/inc

	for i in range(inc):
		pres, = np.where((x >= s) & (x <= s+abs(nbins)))
		if len(pres) > 0:
			med[i]		= weighted_percentile(y[pres], weight[pres], 50)
			upper[i]	= weighted_percentile(y[pres], weight[pres],  84)
			lower[i]	= weighted_percentile(y[pres], weight[pres],  16)
			radii[i]	= s
		s += nbins

	return med, upper, lower, radii

def sumBins(x, y, min, max, s, inc):	#x and y are datasets, min and max is the minimum and maxixum x value to measure from and to, s is the starting position of the data, inc is the total
							#number of increments you want (i.e. 200 would repeat the loop 200 times), and weight is your data to weigh if any
	suM, x_axis = np.zeros(inc), np.zeros(inc)

	nbins = (max-min)/inc

	for i in range(inc):
		pres, = np.where((x >= s) & (x <= s+abs(nbins)))
		if len(pres) > 0:
			suM[i]		= sum(y[pres])
			x_axis[i]	= s
		s += nbins

	return suM, x_axis

#This function converts the 2D grid data from get_Aslice into 1D grid data to easily plot column density data as functions of radius etc.
def col_dens_dist(ion):
	global ionw	#Please ignore

	#Initialise 2-D arrays for the radius and column density (column density is added to a new array which this function spits out)
	s = (len(load.data[str(ion) + 'proj']['x']), len(load.data[str(ion) + 'proj']['y']))

	load.data[str(ion) + 'rad'], load.data[str(ion) + 'ionw']	= np.zeros(s), np.zeros(s)

	#Iterate over x and y coordinates generated by get_Aslice. This creates a magnitude for each pixel
	for n in range(len(load.data[str(ion) + 'proj']['x'])):
		for m in range(len(load.data[str(ion) + 'proj']['y'])):
			load.data[str(ion) + 'rad'][n, m]		= np.sqrt(load.data[str(ion) + 'proj']['x'][n]**2 + load.data[str(ion) + 'proj']['y'][m]**2)

	#Iterate over the length of x and y coordinates again, this time to generate a new array of column density for each pixel in order for the
	#column density values to match the position of the new array
	for n in range(len(load.data[str(ion) + 'proj']['x'])-1):
		for m in range(len(load.data[str(ion) + 'proj']['y'])-1):
			load.data[str(ion) + 'ionw'][n, m]		= load.data[str(ion) + '0'][n,m]

	load.data[str(ion) + 'rad'], load.data[str(ion) + 'ionw']	= load.data[str(ion) + 'rad'].flatten(), load.data[str(ion) + 'ionw'].flatten()

	load.data[str(ion) + 'rad'].tolist()
	load.data[str(ion) + 'rad'] /= virialRadius	#Normalise new radius array to virial radius of halo

	#Remove any nan values from column density array
	nan_array = np.isnan(load.data[str(ion) + 'ionw'])
	not_nan_array = ~nan_array

	load.data[str(ion) + 'ionw']	= load.data[str(ion) + 'ionw'][not_nan_array]
	load.data[str(ion) + 'ionw'].tolist()

	ionw, = np.where((load.data[str(ion) + 'rad'] <= 1))	#Create conditional where whatever the name of the radius variable is only including the virial radius

	return np.log10(load.data[str(ion) + 'ionw'][ionw], where = load.data[str(ion) + 'ionw'][ionw] != 0), load.data[str(ion) + 'rad'][ionw], load.data[str(ion) + 'ionw'][ionw]

#Credit to Andrew Hannington for this. Calculate the weighted percentile for a set of data
def weighted_percentile(data, weights, perc, key="Unspecified Error key..."):
	"""
	Find the weighted Percentile of the data. perc should be given in
	ercentage NOT in decimal!
	Returns a zero value and warning if all Data (or all weights) are NaN
	"""

	# percentage to decimal
	perc				/= 100.0

	# Indices of data array in sorted form
	ind_sorted			= np.argsort(data)

	# Sort the data
	sorted_data			= np.array(data)[ind_sorted]

	# Sort the weights by the sorted data sorting
	sorted_weights			= np.array(weights)[ind_sorted]

	# Remove nan entries
	whereDataIsNotNAN		= np.where(np.isnan(sorted_data) == False)

	sorted_data, sorted_weights	= sorted_data[whereDataIsNotNAN], sorted_weights[whereDataIsNotNAN]

	whereWeightsIsNotNAN		= np.where(np.isnan(sorted_weights) == False)
	sorted_weights			= sorted_weights[whereWeightsIsNotNAN]

	nDataNotNan, nWeightsNotNan	= len(sorted_data), len(sorted_weights)

	if nDataNotNan > 0:
		# Find the cumulative sum of the weights
		cm = np.cumsum(sorted_weights)

		# Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
		whereperc = np.where(cm / float(cm[-1]) >= perc)

		# Reurn the first data value where above is true
		out = sorted_data[whereperc[0][0]]
	else:

		print(key)
		print("[@WeightPercent:] Warning! Data all nan! Returning 0 value!")

		out = np.array([0.0])

	return out

#Calculates the rotational coefficient of a galaxy
def kappa_func():

	disk,		= np.where((normRadius >= 0.0) & (normRadius <= 0.30))

	vel_stars	= load.data['vel'][disk][len(load.ne):]
	pos_stars	= load.pos[disk][len(load.ne):]

	v_xy_stars	= np.sqrt((vel_stars[:,1]**2) + (vel_stars[:,2]**2))
	r_xy_stars	= np.sqrt((pos_stars[:,1]**2) + (pos_stars[:,2]**2))

	j_z_stars	= vel_stars[:,1]*pos_stars[:,2] - pos_stars[:,1]*vel_stars[:,2]

	D_stars		= sum(0.5*load.data['mass'][disk][len(load.ne):]*(j_z_stars/r_xy_stars)**2)
#	D_gas		= sum(0.5*load.data['mass'][disk][:len(load.ne[r_xy_gas != 0])]*(j_z_gas[r_xy_gas != 0]/r_xy_gas[r_xy_gas != 0])**2)

	T_stars		= sum(0.5*load.data['mass'][disk][len(load.ne):]*(np.sqrt(((vel_stars)**2).sum(axis = 1)))**2)
#	T_gas		= sum(0.5*load.data['mass'][disk][:len(load.ne[r_xy_gas != 0])]*(np.sqrt(((vel_gas[r_xy_gas != 0])**2).sum(axis = 1)))**2)

	kappa_rot_stars	= D_stars/T_stars
#	kappa_rot_gas = D_gas/T_gas

	kappa_stars.append(kappa_rot_stars)
#	kappa_gas.append(kappa_rot_gas)

def poly(x_val, n, *p):
	return p[0]*x_val**n + p[1]*x_val**(n-1) + p[2]*x_val**(n-2) + p[3]

def avg(x, y, min, max, s, inc, ion):
	med, upper, lower, radii = [], [], [], []
	nbins = (max - min)/inc

	for i in range(inc):
		pres, = np.where((x >= s) & (x <= s+abs(nbins)) & (load.data[str(ion) + 'rad'] <= 1))
		if len(pres) > 0:
			med.append(median(y[pres]))
			upper.append(np.nanpercentile(y[pres], 84))
			lower.append(np.nanpercentile(y[pres], 16))
			radii.append(s)

		s += nbins
	return med, upper, lower, radii

for i in range(len(halo_d)):

	if halo_d[i] == 'level3_cgm_1e11':
		halo = ['h10_nobfield', 'h10_standard', 'h4_nobfield', 'h4_standard', 'h11_nobfield', 'h11_standard', 'h5_nobfield', 'h5_standard']
		halo = ['h10_nobfield', 'h10_standard', 'h5_nobfield', 'h5_standard']	#1e11+1e11.5 haloes - set 1
		halo2 = ['h11_nobfield', 'h11_standard', 'h4_nobfield', 'h4_standard']	#1e11+1e11.5 haloes - set 2
	if halo_d[i] == 'level4_cgm':
		halo = ['h12_nobfield', 'h12_standard', 'h6_nobfield', 'h6_standard', 'h5_standard_nobfield', 'h5_standard']
		halo = ['h5_standard_nobfield', 'h5_standard']	#1e12 halo - set 1
		halo2 = ['h6_nobfield', 'h6_standard']			#1e12 halo - set 2
	if halo_d[i] == 'level4_cgm_1e13':
		halo = ['h3_nobfield', 'h3_standard', 'h4_nobfield', 'h4_standard', 'h8_nobfield', 'h8_standard', 'h7_nobfield', 'h7_standard']
		halo = ['h3_nobfield', 'h3_standard', 'h8_nobfield', 'h8_standard']	#1e13 halo - set 1
		halo = ['h8_nobfield', 'h8_standard']
		halo2 = ['h7_nobfield', 'h7_standard']	#1e13 halo - set 2
#		halo = ['h3_nobfield', 'h3_standard']	#1e12.5 halo - set 1
#		halo = ['h4_nobfield', 'h4_standard']	#1e12.5 halo - set 2
	for j in range(len(halo)):
		for k in range(len(haloid)):
			#load in the data from their directory
			if halo_d[i] != 'level4_cgm_1e13':
				subload = load_subfind(snap[0], dir = '/home/cosmos/spxfv/Auriga/' + halo_d[i] + '/%s/output/' % halo[j])
				load    = gadget_readsnap(snap[0], snappath = '/home/cosmos/spxfv/Auriga/' + halo_d[i] + '/%s/output/' % halo[j], loadonlytype=[0,4], loadonlyhalo=haloid[k], lazy_load=True, subfind = subload)
			else:
				subload = load_subfind(snap[0], dir = '/home/tango/spxfv/surge/' + halo_d[i] + '/%s/output/' % halo[j])
				load    = gadget_readsnap(snap[0], snappath = '/home/tango/spxfv/surge/' + halo_d[i] + '/%s/output/' % halo[j], loadonlytype=[0,4], loadonlyhalo=haloid[k], lazy_load=True, subfind = subload)

			#Reorientate galaxy to look edge on
			load.calc_sf_indizes(subload)
			load.select_halo(sf = subload, do_rotation = True, haloid = haloid[k])

			print(load.redshift)

			#Investigating the birth of stars
			stellarAge	= load.age[load.age > 0]	#Time when stars formed. load.age > 0 added to exclude wind phase gas cells. This is in terms of the scale factor
			whereStars	= np.where(load.age > 0)	#Where stars are formed excluding wind particles.
			birthRedshift = (1/stellarAge) - 1
			birthMass = load.data['gima'][whereStars]*1e10
			stellarBirthRedshift.append((1/stellarAge) - 1)
			stellarInitMass.append(load.data['gima'][whereStars]*1e10)

			#Definitions of virial radius, halo mass, halo velocity in m/s and stellar mass
			virialRadius	= subload.data['frc2'][haloid[k]]*1e3	#Outer edge of the circumgalactic medium
			haloMass	= subload.data['fmc2'][haloid[k]]*1e10	#Mass of the dark matter halo
			haloVelocity	= subload.data['fvel'][haloid[k]]*1e3	#Total velocity of the halo for normalisation with gas cells.
			stellarMass	= subload.data['fmty'][0,4]*1e10	#Stellar mass of the galaxy

			#print(halo_d[i], halo[j], np.log10(haloMass))

			load.pos 	*= 1e3	#Converts position of cells into kpc
			load.mass	*= 1e10	#Converts cell mass into solar masses. Why the fuck its in M_sol/1e10 I'll never understand
			load.vol	*= 1e9	#Convert volume to some sort of normal units

			#basic parameters and unit corrections
			load.data['dist']	= np.sqrt((load.pos**2).sum(axis=1))	#magnitude of the 3D array load.data['pos']
			normRadius		= load.data['dist']/virialRadius	#Normalise position values to virial radius
			m_halo			= subload.data['fmc2'][haloid[k]]*1.989e33	#Halo mass in g

			#Read hdf5 file for column density interpolation
			df = h5py.File('fg2009_ss_hr.h5')

			#This where statement will restrict properties to only include GAS cells within the virial radius, exclude the ISM and also exclude any satellite haloes that may have formed
			w, = np.where((load.sfr[:len(load.ne)] == 0) & (normRadius[:len(load.ne)] <= 1) & (normRadius[:len(load.ne)] >= 0) & (load.halo[:len(load.ne)] == 0) & (load.subhalo[:len(load.ne)] == 0) | (load.subhalo[:len(load.ne)] == -1))
			w2, = np.where((load.sfr[:len(load.ne)] == 0) & (normRadius[:len(load.ne)] <= 1) & (normRadius[:len(load.ne)] >= 0))

			#Miscellaneous appends to lists. Usually constant or key properties
			radiusInVirial.append(normRadius[w])
			totalRadius.append(normRadius)
			totalHaloMass.append(haloMass)
			colour.append(round(np.log10(haloMass), 1))
			virialCellGasMass.append(load.mass[w])
			totalStellarMass.append(stellarMass)
			virialCellStellarMass.append(np.log10(load.mass[len(load.ne):][whereStars]))
			print(len(virialCellStellarMass[0]))
			print(len(stellarInitMass[0]))

			noism, = np.where(load.sfr[:len(load.ne)] == 0)

			#Table of elements. Calculations made with this table are used for metallicity or column density
			#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
			elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
			elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
			elementsMass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
			elementsSolar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

			haloes.append(halo[j])
			print(haloes)

			meanweight			= sum(load.gmet[:,0:9][:len(load.ne)], axis = 1) / ( sum(load.gmet[:,0:9][:len(load.ne)]/elementsMass[0:9], axis = 1) + load.ne*load.gmet[:,0][:len(load.ne)] )

			#Calculations of physical properties
			Tfac				= 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53

			load.bfld			*= c.bfac * 1e6
			load.data['bfld']	= abs(np.sqrt((load.bfld**2).sum(axis=1)))										#Magnitude of the magnetic field
			load.data['dense']	= load.rho/(c.parsec * 1e6) ** 3 * c.msol * 1e10									#Density of cells in g/cm^3
			load.data['T']		= load.u/Tfac														#Temperature in K
			load.data['gcol']	= load.gcol														#Cooling function. -ve are cooling and +ve are heating
			load.data['gz']		= abs(load.gz/0.0127)													#Metallicity normalised to solar metallicity
			load.data['thermP']	= load.data['dense']*load.data['T']/(meanweight*m_p)									#Thermal pressure in Pascals/k_b
			load.data['bfldP']	= ((load.data['bfld']/1e6)**2)/(8*np.pi*k_b)										#Magnetic pressure also in Pascals/k_b
			load.data['totP']	= load.data['thermP'] + load.data['bfldP']										#Total pressure
			load.data['n_H']	= ((load.data['dense'] / (elementsMass[0]*c.amu)) * load.gmet[:,0][:len(load.ne)])					#Number density of hydrogen
			load.data['Tvir']	= ((G**2)*(H_0**2)*(load.omega0*18*np.pi**2)/54)**(1/3)*(meanweight*m_p/k_b)*((m_halo**(2/3))*(1+load.redshift))	#Virial temperature of the halo
			load.data['T/Tvir']	= load.data['T']/load.data['Tvir']											#Ratio of temperature of all cells to virial temperature
			load.data['pRat']	= load.data['thermP'][w]/load.data['bfldP'][w]								#Pressure ratio

			print(load.data['gcol'][load.data['gcol']< 0])

			gasdens					= load.rho / (c.parsec*1e6)**3. * c.msol * 1e10 									# g cm^-3
			gasX					= load.gmet[:,0][:len(load.ne)] 											# hydrogen mass fraction

			#Calculate radial velocity for all cells
			mag						= np.sqrt(((load.pos[:,0])**2) + ((load.pos[:,1])**2) + ((load.pos[:,2])**2))			#Magnitude of the position of each cell from the centre of the halo.
			load.data['velocity']	= (load.vel[:,0]*load.pos[:,0]+load.vel[:,1]*load.pos[:,1]+load.vel[:,2]*load.pos[:,2])/mag	#Radial velcoity, km/s assuming no change to units above
			dm						= load.data['velocity'] * load.mass/(3.17e-8*(virialRadius*3.086e16))		#Mass flux positive is outflowing matter and negative is accretion. 3.17e-8 converts km/s to km/yr

			#Parameters for visualisation
			boxsize			= 2*virialRadius	#Determine the size of the box as 2x larger than intended
			boxlos			= 2*virialRadius	#Line-of-sight for the box
			pixres			= 2.0 			#Resolution of pixels in the visualisation. Higher value = lower resolution image. Use high resolution for column density calculation.
			pixreslos		= 0.1 			#Number of slices along the line of sight

			imgcent	= [0,0,0]
			axes	= [[1,0], [2,0], [2,1]] 		#Orientation of axes for get_Aslice. [2,1] is face on, [1,0] and [2,0] are edge on. Each value can be switched
			axes	= [1,0]

			radialVelocityHaloMass.append(np.log10(load.data['mass'][w2]))
			convert = pixreslos*Mpcincm/1e3
			
			load.data['n_HI']				= ((load.data['dense'][:len(load.ne)] / (elementsMass[0]*c.amu)) * load.gmet[:,0][:len(load.ne)]) * load.data['nh'][:len(load.ne)]
			load.data['n_CIV'], CIV_mass, CIV_mass_frac	= ionisation('C', 2, 3)
			load.data['n_OVI'], OVI_mass, OVI_mass_frac	= ionisation('O', 4, 5)
			load.data['n_SiII'], SiII_mass, SiII_mass_frac	= ionisation('Si', 7, 1)
			
			print(len(load.data['n_HI']))
			#Create projections of various ion column density. Used to compare to observational column densities
			load.data['HIproj']				= load.get_Aslice("n_HI", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = axes, proj = True, numthreads = 8)
			load.data['CIVproj']				= load.get_Aslice("n_CIV", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = axes, proj = True, numthreads = 8)
			load.data['OVIproj']				= load.get_Aslice("n_OVI", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = axes, proj = True, numthreads = 8)
			load.data['SiIIproj']				= load.get_Aslice("n_SiII", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = axes, proj = True, numthreads = 8)
			
			load.data['HI0']				= load.data['HIproj']['grid']*convert
			load.data['CIV0']				= load.data['CIVproj']['grid']*convert
			load.data['OVI0']				= load.data['OVIproj']['grid']*convert
			load.data['SiII0']				= load.data['SiIIproj']['grid']*convert

			load.data['HI']					= load.data['HIproj']['grid'].flatten()*convert
			load.data['CIV']				= load.data['CIVproj']['grid'].flatten()*convert
			load.data['OVI']				= load.data['OVIproj']['grid'].flatten()*convert
			load.data['SiII']				= load.data['SiIIproj']['grid'].flatten()*convert

			load.data['HIionw'], load.data['HIrad'], _		= col_dens_dist('HI')
			load.data['CIVionw'], load.data['CIVrad'], _		= col_dens_dist('CIV')
			load.data['OVIionw'], load.data['OVIrad'], _		= col_dens_dist('OVI')
			load.data['SiIIionw'], load.data['SiIIrad'] , _		= col_dens_dist('SiII')

			HI_med, HI_upper, HI_lower, HI_radii			= avg(load.data['HIrad'], load.data['HIionw'], min(load.data['HIrad']), max(load.data['HIrad']), min(load.data['HIrad']), 50, 'HI')
			CIV_med, CIV_upper, CIV_lower, CIV_radii		= avg(load.data['CIVrad'], load.data['CIVionw'], min(load.data['CIVrad']), max(load.data['CIVrad']), min(load.data['CIVrad']), 50, 'CIV')
			OVI_med, OVI_upper, OVI_lower, OVI_radii		= avg(load.data['OVIrad'], load.data['OVIionw'], min(load.data['OVIrad']), max(load.data['OVIrad']), min(load.data['OVIrad']), 50, 'OVI')
			SiII_med, SiII_upper, SiII_lower, SiII_radii		= avg(load.data['SiIIrad'], load.data['SiIIionw'], min(load.data['SiIIrad']), max(load.data['SiIIrad']), min(load.data['SiIIrad']), 50, 'SiII')

			H_med.append(HI_med), H_upper.append(HI_upper), H_lower.append(HI_lower), H_radii.append(HI_radii)
			C_med.append(CIV_med), C_upper.append(CIV_upper), C_lower.append(CIV_lower), C_radii.append(CIV_radii)
			O6_med.append(OVI_med), O6_upper.append(OVI_upper), O6_lower.append(OVI_lower), O6_radii.append(OVI_radii)
			Mg_med.append(SiII_med), Mg_upper.append(SiII_upper), Mg_lower.append(SiII_lower), Mg_radii.append(SiII_radii)
			
			print(load.data['HIionw'])
			print(load.data['HIionw'][load.data['HIionw'] >= 18])
			cFracHI.append(len(load.data['HIionw'][load.data['HIionw'] >= 14.15])/len(load.data['HIionw']))
			cFracCIV.append(len(load.data['CIVionw'][load.data['CIVionw'] >= 14.69])/len(load.data['CIVionw']))
			cFracOVI.append(len(load.data['OVIionw'][load.data['OVIionw'] >= 15.41])/len(load.data['OVIionw']))
			cFracSiII.append(len(load.data['SiIIionw'][load.data['SiIIionw'] >= 11.58])/len(load.data['SiIIionw']))
			
			radialVelocity.append(load.data['velocity'][w2])
			T.append(load.data['T'][w])
			gcol.append(load.data['gcol'][w][load.data['gcol'][w] < 0])
			Z.append(load.data['gz'][w])
			hydrogenDensity.append(load.data['n_H'][w])
			thermalPressure.append(load.data['thermP'][w])
			magneticPressure.append(load.data['bfldP'][w])
			totalPressure.append(load.data['totP'][w])
			pressureRatio.append(load.data['pRat'])
			bField.append(load.data['bfld'][w])
			radialVelocityRadius.append(normRadius[w2])
			massFlux.append(sum(dm))
			Mass.append(load.mass[w])
			Tvir.append(load.data['Tvir'])
			dTvir.append(load.data['T']/load.data['Tvir'])
			
			outflows, = np.where((dm[:len(load.ne)] >= 0) & (load.sfr[:len(load.ne)] == 0) & (normRadius[:len(load.ne)] < 1) & (normRadius[:len(load.ne)] >= 0) & (load.halo[:len(load.ne)] == 0) & (load.subhalo[:len(load.ne)] == 0) | (load.subhalo[:len(load.ne)] == -1))
			inflows, = np.where((dm[:len(load.ne)] <= 0) & (load.sfr[:len(load.ne)] == 0) & (normRadius[:len(load.ne)] < 1) & (normRadius[:len(load.ne)] >= 0) & (load.halo[:len(load.ne)] == 0) & (load.subhalo[:len(load.ne)] == 0) | (load.subhalo[:len(load.ne)] == -1))
			
			te_med, te_upper, te_lower, te_radii			= avg_weighted(normRadius[w], load.data['T'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			
			td_med, td_upper, td_lower, td_radii			= avg_weighted(load.data['n_H'][w], load.data['T'][w], min(load.data['n_H'][w]), max(load.data['n_H'][w]), min(load.data['n_H'][w]), 100, load.mass[w])
			me_med, me_upper, me_lower, me_radii			= avg_weighted(normRadius[w], load.data['gz'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			de_med, de_upper, de_lower, de_radii			= avg_weighted(normRadius[w], load.data['n_H'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			pre_med, pre_upper, pre_lower, pre_radii		= avg_weighted(normRadius[w], load.data['thermP'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			totpre_med, totpre_upper, totpre_lower, totpre_radii	= avg_weighted(normRadius[w], load.data['totP'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			be_med, be_upper, be_lower, be_radii			= avg_weighted(normRadius[w], load.data['bfld'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			vrad_med, vrad_upper, vrad_lower, vrad_radii		= avg_weighted(normRadius[w], load.data['velocity'][w], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])
			dmein_med, dmein_upper, dmein_lower, dmein_radii		= avg_weighted(normRadius[inflows], dm[inflows], min(normRadius[inflows]), max(normRadius[inflows]), min(normRadius[inflows]), 50, load.mass[inflows])
			dmeout_med, dmeout_upper, dmeout_lower, dmeout_radii		= avg_weighted(normRadius[outflows], dm[outflows], min(normRadius[outflows]), max(normRadius[outflows]), min(normRadius[outflows]), 50, load.mass[outflows])
			birth_sum, birth_x	= sumBins(birthRedshift, birthMass, min(birthRedshift), max(birthRedshift), min(birthRedshift), 200)
			medPresRat, upPresRat, loPresRat, radPresRat			= avg_weighted(normRadius[w], load.data['pRat'], min(normRadius[w]), max(normRadius[w]), min(normRadius[w]), 50, load.mass[w])

			t_med.append(te_med), t_upper.append(te_upper), t_lower.append(te_lower), t_radii.append(te_radii)
			tde_med.append(td_med), tde_upper.append(td_upper), tde_lower.append(td_lower), tde_radii.append(td_radii)
			d_med.append(de_med), d_upper.append(de_upper), d_lower.append(de_lower), d_radii.append(de_radii)
			m_med.append(me_med), m_upper.append(me_upper), m_lower.append(me_lower), m_radii.append(me_radii)
			pr_med.append(pre_med), pr_upper.append(pre_upper), pr_lower.append(pre_lower), pr_radii.append(pre_radii)
			totpr_med.append(totpre_med), totpr_upper.append(totpre_upper), totpr_lower.append(totpre_lower), totpr_radii.append(totpre_radii)
			v_med.append(vrad_med), v_upper.append(vrad_upper), v_lower.append(vrad_lower), v_radii.append(vrad_radii)
			dmin_med.append(dmein_med), dmin_upper.append(dmein_upper), dmin_lower.append(dmein_lower), dmin_radii.append(dmein_radii)
			dmout_med.append(dmeout_med), dmout_upper.append(dmeout_upper), dmout_lower.append(dmeout_lower), dmout_radii.append(dmeout_radii)
			b_med.append(be_med), b_upper.append(be_upper), b_lower.append(be_lower), b_radii.append(be_radii)
			birth.append(birth_sum), birthx.append(birth_x)
			presrat_med.append(weighted_percentile(load.data['pRat'][::2], load.mass[w], 50)), presrat_lower.append(weighted_percentile(load.data['pRat'][::2], load.mass[w], 16)), presrat_upper.append(weighted_percentile(load.data['pRat'][::2], load.mass[w], 84))
			
			n_HI.append(load.data['HIionw']), n_HI_radii.append(load.data['HIrad'])
			n_CIV.append(load.data['CIVionw']), n_CIV_radii.append(load.data['CIVrad'])
			n_OVI.append(load.data['OVIionw']), n_OVI_radii.append(load.data['OVIrad'])
			n_SiII.append(load.data['SiIIionw']), n_SiII_radii.append(load.data['SiIIrad'])
			

print(presrat_upper, presrat_lower)
title = [r'T [K]', r'n$_{H}$ [cm$^{-3}$]', r'[Z/Z$_{\odot}$]', r'P$_{\mathrm{Total}}$ [Pa/k]', r'B [r$\mu{G}$]']
title2 = [r'log$_{10}$M$_{200c}$ = 10', r'log$_{10}$M$_{200c}$ = 11', r'log$_{10}$M$_{200c}$ = 11.5', r'log$_{10}$M$_{200c}$ = 12', r'log$_{10}$M$_{200c}$ = 13']
#Physical properties as a function of radius
plt.figure(1)
fontsize = 10*int(len(haloes))
fig, axs = plt.subplots(nrows=5, ncols=5, sharex = 'col', sharey = 'row', figsize = (10,10))

for it3 in range(5):
	axs[0,it3].plot(t_radii[2*it3+2], np.log10(t_med[2*it3+2]), color = 'b', linestyle = '-', label = 'B = 0')
	axs[0,it3].plot(t_radii[2*it3 + 3], np.log10(t_med[2*it3 + 3]), color = 'r', linestyle = '--', label = 'B > 0')
	axs[0,it3].fill_between(t_radii[2*it3+2], np.log10(t_lower[2*it3+2]), np.log10(t_upper[2*it3+2]), color = 'b', alpha = 0.3)
	axs[0,it3].fill_between(t_radii[2*it3+3], np.log10(t_lower[2*it3+3]), np.log10(t_upper[2*it3+3]), color = 'r', alpha = 0.3)

	axs[1,it3].plot(d_radii[2*it3+2], np.log10(d_med[2*it3+2]), color = 'b', linestyle = '-', label = 'B = 0')
	axs[1,it3].plot(d_radii[2*it3 + 3], np.log10(d_med[2*it3 + 3]), color = 'r', linestyle = '--', label = 'B > 0')
	axs[1,it3].fill_between(d_radii[2*it3+2], np.log10(d_lower[2*it3+2]), np.log10(d_upper[2*it3+2]), color = 'b', alpha = 0.3)
	axs[1,it3].fill_between(d_radii[2*it3+3], np.log10(d_lower[2*it3+3]), np.log10(d_upper[2*it3+3]), color = 'r', alpha = 0.3)

	axs[2,it3].plot(m_radii[2*it3+2], np.log10(m_med[2*it3+2]), color = 'b', linestyle = '-', label = 'B = 0')
	axs[2,it3].plot(m_radii[2*it3 + 3], np.log10(m_med[2*it3 + 3]), color = 'r', linestyle = '--', label = 'B > 0')
	axs[2,it3].fill_between(m_radii[2*it3+2], np.log10(m_lower[2*it3+2]), np.log10(m_upper[2*it3+2]), color = 'b', alpha = 0.3)
	axs[2,it3].fill_between(m_radii[2*it3+3], np.log10(m_lower[2*it3+3]), np.log10(m_upper[2*it3+3]), color = 'r', alpha = 0.3)

	axs[3,it3].plot(totpr_radii[2*it3+2], np.log10(totpr_med[2*it3+2]), color = 'b', linestyle = '-', label = 'B = 0')
	axs[3,it3].plot(totpr_radii[2*it3 + 3], np.log10(totpr_med[2*it3 + 3]), color = 'r', linestyle = '--', label = 'B > 0')
	axs[3,it3].fill_between(totpr_radii[2*it3+2], np.log10(totpr_lower[2*it3+2]), np.log10(totpr_upper[2*it3+2]), color = 'b', alpha = 0.3)
	axs[3,it3].fill_between(totpr_radii[2*it3+3], np.log10(totpr_lower[2*it3+3]), np.log10(totpr_upper[2*it3+3]), color = 'r', alpha = 0.3)

	axs[4,it3].plot(b_radii[2*it3 + 1], np.log10(b_med[2*it3 + 1]), color = 'r', linestyle = '--', label = 'B > 0')
	axs[4,it3].fill_between(b_radii[2*it3+3], np.log10(b_lower[2*it3+3]), np.log10(b_upper[2*it3+3]), color = 'r', alpha = 0.3)

	axs[0,it3].grid()
	axs[1,it3].grid()
	axs[2,it3].grid()
	axs[3,it3].grid()
	axs[4,it3].grid()

	axs[0,it3].tick_params(bottom = 'false')
	axs[1,it3].tick_params(bottom = 'false')
	axs[2,it3].tick_params(bottom = 'false')
	axs[3,it3].tick_params(bottom = 'false')
	axs[4,it3].tick_params(bottom = 'false')

	axs[-1,it3].set_xlabel(r'R/R$_{200c}$')
#	axs[0,it3].set_ylabel(title[it3])

for it22 in range(int(len(haloes)/2)-2):
	axs[0,it22].set_ylim(4.0, 7.0)
	axs[1,it22].set_ylim(-6, -1)
	axs[2,it22].set_ylim(-4, 1)
	axs[3,it22].set_ylim(0.0, 5.0)
	axs[4,it22].set_ylim(-3, 1)
	axs[0,it22].set_xlim(0,1)
	axs[1,it22].set_xlim(0,1)
	axs[3,it22].set_xlim(0,1)
	axs[2,it22].set_xlim(0,1)
	axs[3,it22].set_xticks([0.25,0.5,0.75])

axs[0,0].set_yticks([5.0, 6.0, 7.0])
axs[1,0].set_yticks([-5,-4,-3,-2])
axs[2,0].set_yticks([-3,-2,-1,0])
axs[3,0].set_yticks([1,2,3,4])
axs[4,0].set_yticks([-3,-2,-1,0])

axs[3,0].set_xticks([0.0,0.25,0.5,0.75])
axs[3,-1].set_xticks([0.25,0.5,0.75,1.0])

axs[0,0].set_ylabel(title[0])
axs[1,0].set_ylabel(title[1])
axs[2,0].set_ylabel(title[2])
axs[3,0].set_ylabel(title[3])
axs[4,0].set_ylabel(title[4])

#for it23 in range(int(len(haloes)/2)):
#	axs[it23,0].set_title(title2[it23])

for ax in axs.flat:
	ax.label_outer()

plt.subplots_adjust(hspace = 0, wspace = 0)
plt.savefig('./Figures/Paper2Plots/bfld_collated_properties_z=' + str(load.redshift) + '_set1.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')
plt.close('all')
'''
#Magnetic field as a function of radius
plt.figure(2)
fig, axs = plt.subplots(nrows=1, ncols=int(len(haloes)/2), sharex = 'col', sharey = 'row', figsize = (16,4))

axs[0].plot(b_radii[1], np.log10(b_med[1]), color = 'b', linestyle = '-')
axs[0].fill_between(b_radii[1], np.log10(b_lower[1]), np.log10(b_upper[1]), color = 'b', alpha = 0.35)
axs[1].plot(b_radii[3], np.log10(b_med[3]), color = 'b', linestyle = '-')
axs[1].fill_between(b_radii[3], np.log10(b_lower[3]), np.log10(b_upper[3]), color = 'b', alpha = 0.35)
axs[2].plot(b_radii[5], np.log10(b_med[5]), color = 'b', linestyle = '-')
axs[2].fill_between(b_radii[5], np.log10(b_lower[5]), np.log10(b_upper[5]), color = 'b', alpha = 0.35)
axs[3].plot(b_radii[7], np.log10(b_med[7]), color = 'b', linestyle = '-')
axs[3].fill_between(b_radii[7], np.log10(b_lower[7]), np.log10(b_upper[7]), color = 'b', alpha = 0.35)
axs[4].plot(b_radii[9], np.log10(b_med[9]), color = 'b', linestyle = '-')
axs[4].fill_between(b_radii[9], np.log10(b_lower[9]), np.log10(b_upper[9]), color = 'b', alpha = 0.35)

axs[0].set_ylabel(r'B [$\mu$G]')
axs[0].set_xlabel(r'R/R$_{200c}$')
axs[1].set_xlabel(r'R/R$_{200c}$')
axs[2].set_xlabel(r'R/R$_{200c}$')
axs[3].set_xlabel(r'R/R$_{200c}$')
axs[4].set_xlabel(r'R/R$_{200c}$')

axs[0].set_xlabel([0.0,0.25,0.5,0.75,1.0])
axs[1].set_xlabel([0.0,0.25,0.5,0.75,1.0])
axs[2].set_xlabel([0.0,0.25,0.5,0.75,1.0])
axs[3].set_xlabel([0.0,0.25,0.5,0.75,1.0])
axs[4].set_xlabel([0.0,0.25,0.5,0.75,1.0])

plt.legend()
axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()
axs[4].grid()
plt.subplots_adjust(wspace = 0.1)
plt.savefig('./Figures/Paper2Plots/Magnetic_Field_Strength_z=' + str(load.redshift) + '_set1.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')
plt.close('all')
'''
#np.array(haloMass)
#colour = [int(((3*(haloMass[i]-haloMass[0]))/haloMass[-1])*100) for i in range(len(haloMass))]

plt.figure(2)
#fontsize = 10*int(len(haloes))
fig, axs = plt.subplots(2, 2, figsize = (4,4), sharey = True, sharex = True)
axs[0,0].scatter(totalStellarMass[::2], cFracHI[::2], s = 12, c = 'r', marker = 's', label = r'HI$\geq{14.15}$') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[0,1].scatter(totalStellarMass[::2], cFracCIV[::2], s = 12, c = 'r', marker = 's', label = r'CIV$\geq{14.69}$') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[1,0].scatter(totalStellarMass[::2], cFracOVI[::2], s = 12, c = 'r', marker = 's', label = r'OVI$\geq{15.41}$') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[1,1].scatter(totalStellarMass[::2], cFracSiII[::2], s = 12, c = 'r', marker = 's', label = r'SiII$\geq{11.58}$') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)

axs[0,0].scatter(totalStellarMass[1::2], cFracHI[1::2], s = 12, c = 'b', marker = 'x') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[0,1].scatter(totalStellarMass[1::2], cFracCIV[1::2], s = 12, c = 'b', marker = 'x') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[1,0].scatter(totalStellarMass[1::2], cFracOVI[1::2], s = 12, c = 'b', marker = 'x') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[1,1].scatter(totalStellarMass[1::2], cFracSiII[1::2], s = 12, c = 'b', marker = 'x') #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)

axs[0,0].set_ylabel(r'Covering Fraction')
axs[1,0].set_ylabel(r'Covering Fraction')

axs[0,0].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('./Figures/Paper2Plots/SF_History_z=' + str(load.redshift) + '_set1.pdf', dpi = 300, transparent = True)
plt.close('all')

plt.figure(3)
#fontsize = 10*int(len(haloes))
fig, axs = plt.subplots(nrows = 5, ncols=1, figsize = (6,18), sharex = True)

noBin = axs[0].plot(dmin_radii[2], np.log10(abs(dmin_med[2])), c = 'b', label = r'$-\dot{M}_{B=0}$')
noBout = axs[0].plot(dmout_radii[2], np.log10(dmout_med[2]), ls = '--', c = 'b', label = r'$+\dot{M}_{B=0}$')
Bin = axs[0].plot(dmin_radii[3], np.log10(abs(dmin_med[3])), c = 'r', label = r'$-\dot{M}_{B>0}$')
Bout = axs[0].plot(dmout_radii[3], np.log10(dmout_med[3]), ls = '--', c = 'r', label = r'$+\dot{M}_{B>0}$')
axs[1].plot(dmin_radii[4], np.log10(abs(dmin_med[4])), c = 'b')
axs[1].plot(dmout_radii[4], np.log10(dmout_med[4]), ls = '--', c = 'b')
axs[1].plot(dmin_radii[5], np.log10(abs(dmin_med[5])), c = 'r')
axs[1].plot(dmout_radii[5], np.log10(dmout_med[5]), ls = '--', c = 'r')
axs[2].plot(dmin_radii[6], np.log10(abs(dmin_med[6])), c = 'b')
axs[2].plot(dmout_radii[6], np.log10(dmout_med[6]), ls = '--', c = 'b')
axs[2].plot(dmin_radii[7], np.log10(abs(dmin_med[7])), c = 'r')
axs[2].plot(dmout_radii[7], np.log10(dmout_med[7]), ls = '--', c = 'r')
axs[3].plot(dmin_radii[8], np.log10(abs(dmin_med[8])), c = 'b')
axs[3].plot(dmout_radii[8], np.log10(dmout_med[8]), ls = '--', c = 'b')
axs[3].plot(dmin_radii[9], np.log10(abs(dmin_med[9])), c = 'r')
axs[3].plot(dmout_radii[9], np.log10(dmout_med[9]), ls = '--', c = 'r')
axs[4].plot(dmin_radii[10], np.log10(abs(dmin_med[10])), c = 'b')
axs[4].plot(dmout_radii[10], np.log10(dmout_med[10]), ls = '--', c = 'b')
axs[4].plot(dmin_radii[11], np.log10(abs(dmin_med[11])), c = 'r')
axs[4].plot(dmout_radii[11], np.log10(dmout_med[11]), ls = '--', c = 'r')

axs[4].xlabel(r'R/R$_{200c}$')
axs[0].set_ylabel(r'dM/dt [$\mathrm{M}_{\odot}/\mathrm{yr}$]')
axs[1].set_ylabel(r'dM/dt [$\mathrm{M}_{\odot}/\mathrm{yr}$]')
axs[2].set_ylabel(r'dM/dt [$\mathrm{M}_{\odot}/\mathrm{yr}$]')
axs[3].set_ylabel(r'dM/dt [$\mathrm{M}_{\odot}/\mathrm{yr}$]')
axs[4].set_ylabel(r'dM/dt [$\mathrm{M}_{\odot}/\mathrm{yr}$]')

axs[0].set_xlim(0,1)
axs[1].set_xlim(0,1)
axs[2].set_xlim(0,1)
axs[3].set_xlim(0,1)
axs[4].set_xlim(0,1)

fig.legend([(Bin), (Bout), (noBin), (noBout)], [r'$-\dot{M}_{B=0}$', r'$+\dot{M}_{B=0}$', r'$-\dot{M}_{B>0}$', r'$+\dot{M}_{B>0}$'], scatterpoints=1, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='center', bbox_to_anchor=(0.5, 0.02), ncol = 2)

plt.subplots_adjust(hspace = 0)
plt.savefig('./Figures/Paper2Plots/Mass_Flux_z=' + str(load.redshift) + '_set1.pdf', dpi = 300, transparent = True)
plt.close('all')

plt.figure(4)
#fontsize = 10*int(len(haloes))
fig, axs = plt.subplots(int(len(haloes)/2), 2, figsize = (4,12), sharey = True, sharex = True)
axs[0,0].scatter(birthx[0], np.log10(birth[0]), s = 12) #, c = virialCellStellarMass[0], cmap = plt.cm.rainbow)
axs[0,1].scatter(birthx[1], np.log10(birth[1]), s = 12) #, c = virialCellStellarMass[1], cmap = plt.cm.rainbow)
axs[1,0].scatter(birthx[2], np.log10(birth[2]), s = 12) #, c = virialCellStellarMass[2], cmap = plt.cm.rainbow)
axs[1,1].scatter(birthx[3], np.log10(birth[3]), s = 12) #, c = virialCellStellarMass[3], cmap = plt.cm.rainbow)
axs[2,0].scatter(birthx[4], np.log10(birth[4]), s = 12) #, c = virialCellStellarMass[4], cmap = plt.cm.rainbow)
axs[2,1].scatter(birthx[5], np.log10(birth[5]), s = 12) #, c = virialCellStellarMass[5], cmap = plt.cm.rainbow)
axs[3,0].scatter(birthx[6], np.log10(birth[6]), s = 12) #, c = virialCellStellarMass[6], cmap = plt.cm.rainbow)
axs[3,1].scatter(birthx[7], np.log10(birth[7]), s = 12) #, c = virialCellStellarMass[7], cmap = plt.cm.rainbow)
axs[4,0].scatter(birthx[8], np.log10(birth[8]), s = 12) #, c = virialCellStellarMass[8], cmap = plt.cm.rainbow)
axs[4,1].scatter(birthx[9], np.log10(birth[9]), s = 12) #, c = virialCellStellarMass[9], cmap = plt.cm.rainbow)

axs[0,0].set_ylabel(r'Initial Mass [$M_{\odot}$]')
axs[1,0].set_ylabel(r'Initial Mass [$M_{\odot}$]')
axs[2,0].set_ylabel(r'Initial Mass [$M_{\odot}$]')
axs[3,0].set_ylabel(r'Initial Mass [$M_{\odot}$]')
axs[4,0].set_ylabel(r'Initial Mass [$M_{\odot}$]')

axs[0,0].set_ylim(3, 10)
axs[0,0].set_xlim(-1, 20)

plt.tight_layout()
plt.savefig('./Figures/Paper2Plots/SF_History_z=' + str(load.redshift) + '_set1.jpg', dpi = 300, transparent = True)
plt.close('all')
'''
plt.figure(6)
fig, axs = plt.subplots(ncols=2, nrows=1, figsize = (4,4), sharey = True, sharex = True)

axs[0].hist2d(np.log10(bField[6]), np.log10(T[6]), bins = 250, weights = virialCellGasMass[6], norm = mpl.colors.LogNorm(), cmap = plt.get_cmap('Spectral_r'))
axs[1].hist2d(np.log10(bField[7]), np.log10(T[7]), bins = 250, weights = virialCellGasMass[7], norm = mpl.colors.LogNorm(), cmap = plt.get_cmap('Spectral_r'))

axs[0].ylabel('logT')
axs[0].xlabel('logB')
axs[1].xlabel('logB')

axs[0].ylim(4.0, 7.0)
axs[1].colorbar()

plt.tight_layout()
plt.savefig('./Figures/Paper2Plots/bfld_T_2dHist.pdf', dpi = 300)
plt.close('all')
'''
plt.figure(7)
fig, axs = plt.subplots(4, 5, figsize = (24,15))
#plt.rcParams.update({'font.size': 24})

for i in range(5):

	axs[0,i].fill_between(H_radii[2*i+2], H_upper[2*i+2], H_lower[2*i+2], alpha = 0.3, color = 'b')
	axs[1,i].fill_between(O6_radii[2*i+2], O6_upper[2*i+2], O6_lower[2*i+2], alpha = 0.3, color = 'b')
	axs[2,i].fill_between(C_radii[2*i+2], C_upper[2*i+2], C_lower[2*i+2], alpha = 0.3, color = 'b')
	axs[3,i].fill_between(Mg_radii[2*i+2], Mg_upper[2*i+2], Mg_lower[2*i+2], alpha = 0.3, color = 'b')
	axs[0,i].fill_between(H_radii[2*i+3], H_upper[2*i+3], H_lower[2*i+3], alpha = 0.3, color = 'r')
	axs[1,i].fill_between(O6_radii[2*i+3], O6_upper[2*i+3], O6_lower[2*i+3], alpha = 0.3, color = 'r')
	axs[2,i].fill_between(C_radii[2*i+3], C_upper[2*i+3], C_lower[2*i+3], alpha = 0.3, color = 'r')
	axs[3,i].fill_between(Mg_radii[2*i+3], Mg_upper[2*i+3], Mg_lower[2*i+3], alpha = 0.3, color = 'r')

	axs[0,i].plot(H_radii[2*i+2], H_med[2*i+2], color = 'b', label = 'B = 0')
	axs[1,i].plot(O6_radii[2*i+2], O6_med[2*i+2], color = 'b', label = 'B = 0')
	axs[2,i].plot(C_radii[2*i+2], C_med[2*i+2], color = 'b', label = 'B = 0')
	axs[3,i].plot(Mg_radii[2*i+2], Mg_med[2*i+2], color = 'b', label = 'B = 0')
	axs[0,i].plot(H_radii[2*i+3], H_med[2*i+3], color = 'r', label = 'B > 0')
	axs[1,i].plot(O6_radii[2*i+3], O6_med[2*i+3], color = 'r', label = 'B > 0')
	axs[2,i].plot(C_radii[2*i+3], C_med[2*i+3], color = 'r', label = 'B > 0')
	axs[3,i].plot(Mg_radii[2*i+3], Mg_med[2*i+3], color = 'r', label = 'B > 0')

axs[0,0].set_ylabel(r'N$_{HI}$ [cm$^{-2}$]')
axs[1,0].set_ylabel(r'N$_{OVI}$ [cm$^{-2}$]')
axs[2,0].set_ylabel(r'N$_{CIV}$ [cm$^{-2}$]')
axs[3,0].set_ylabel(r'N$_{SiII}$ [cm$^{-2}$]')

for i in range(int(len(haloes)/2)):
	axs[0,i].set_ylim(13,22)
	axs[1,i].set_ylim(12,18)
	axs[2,i].set_ylim(12,18)
	axs[3,i].set_ylim(12,18)

for ax in axs.flat:
	ax.set_xlabel(r'R/R$_{200}$')
	ax.label_outer()
	ax.grid()

plt.subplots_adjust(hspace = 0.15, wspace = 0.15)

plt.savefig('./Figures/Paper2Plots/bfld_col_dens_radii_z=' + str(load.redshift) + '_set1.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')
plt.close('all')

plt.figure(8)

plt.plot(T[6][load.data['gcol']< 0], gcol[6])

plt.savefig('./Figures/Paper2Plots/cooling_function_test.pdf', dpi = 300, transparent = True, bbox_inches = 'tight')
plt.close('all')
