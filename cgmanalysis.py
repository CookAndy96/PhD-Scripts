import numpy as np
from gadget import *
from gadget_subfind import *
import matplotlib.pyplot as plt
import const as c
from sys import argv
import scipy
from scipy.interpolate import griddata

def virialTemp(snap, halo, halo_d):
#	subload = load_subfind(int(snap), dir = '/home/universe/spxfv/Auriga/' + halo_d + '/%s/output/' % halo)
#	load = gadget_readsnap(int(snap), snappath = '/home/universe/spxfv/Auriga/' + halo_d + '/%s/output/' % halo, loadonlytype=[0], loadonlyhalo=0, lazy_load=True, subfind = subload)

	#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
	elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
	elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
	elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
	elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

	m_h = 1.67e-24
	k = 1.38e-16
	G = 6.67e-8
	H_0 = 2.17e-18
	m_halo = subload.data['fmc2'][0]*1.989e43
	meanweight = sum(load.gmet[:,0:9], axis = 1) / ( sum(load.gmet[:,0:9]/elements_mass[0:9], axis = 1) + load.ne*load.gmet[:,0] )
	meanweight = sum(meanweight)/len(meanweight)

	print(meanweight)
	print(m_halo)

	T_vir = ((G**2)*(H_0**2)*(load.omega0)*18*(np.pi**2)/54)**(1/3)*(meanweight*m_h/k)*((m_halo**(2/3))*(1+load.redshift))
	return T_vir

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

"""The following function interpolates hydrogen number density, redshift and temperature from Arepo simulations over ion tables generated with cloudy.
This produces the estimated ion number density, ion mass and ion mass fraction from these simulations. The function can be used with other simulation
codes but will likely need some tuning."""
def ionisation(element, mass, ion, z, df, elements_mass, load = load):
	table = np.array(list(df[element][ion]))
	redShift = np.array([0.0, 0.12202, 0.25893, 0.41254, 0.58489, 0.77828, 0.99526, 1.2387, 1.5119, 1.8184, 2.1623, 2.5481, 2.9811, 3.4668, 4.0119, 4.6234, 5.3096, 6.0795, 6.9433, 7.9125, 9.0, 10.22])
	rdshft, rdidx = find_nearest(redShift, z)
	print('Nearest Redshift = ', rdshft)
	#Initialise length of n_H, z and T arrays
	n_H, z, T = np.arange(-9, 2.125, 0.125).round(3), rdidx, np.arange(1, 9.025, 0.025).round(3)

	#Create repeating and tiled arrays in opposite orientations
	logdens, logtemp = np.repeat(np.array(n_H), len(T)), np.tile(np.array(T), len(n_H))

	#iontable selects all data from table for specific element and ion with z in this case being constant as we are mainly analysing data at z = 0
	iontable = table[:,z,:]

	#Interpolate. As data from ion tables are logged, so much n_H and T
	#Using arrays from above to save interpolated data to flatten ion table data into 1-D array T and n_H used as metrics for interpolation
	linetable = np.power(10, scipy.interpolate.griddata((logdens, logtemp), iontable.flatten(), (np.log10(load.data['n_H']), np.log10(load.data['T']))))
	metFrac = np.multiply(load.gmet[:,mass][:len(load.ne)], linetable)
	#Number density of specific species of element (CIII, SiVIII etc)
	nIon = abs(np.multiply((np.divide(load.data['dense'], np.multiply(elements_mass[mass], c.amu))),metFrac))

	#Total mass of the ion in the halo
	totalIonMass = np.float64(np.multiply(load.data['mass'][:len(load.ne)],metFrac))

	#Mass fraction of the ion in the halo
	totalIonMassFrac = np.multiply(linetable, load.gmet[:,mass][:len(load.ne)])

	return nIon, totalIonMass, totalIonMassFrac

def weighted_percentile(data, weights, perc, key="Unspecified Error key..."):
	"""
	Find the weighted Percentile of the data. dec should be given in decimal!
	Returns a zero value and warning if all Data (or all weights) are NaN
	"""
	perc = np.divide(perc,100)
	# Indices of data array in sorted form
	ind_sorted			= np.argsort(data)

	# Sort the data and weights
	sorted_data			= np.array(data)[ind_sorted]
	sorted_weights		= np.array(weights)[ind_sorted]

	# Remove nan entries
	whereDataIsNotNAN		= np.where(np.isnan(sorted_data) == False)
	whereWeightsIsNotNAN	= np.where(np.isnan(sorted_weights) == False)

	sorted_data, sorted_weights	= sorted_data[whereDataIsNotNAN], sorted_weights[whereDataIsNotNAN]
	sorted_weights			= sorted_weights[whereWeightsIsNotNAN]

	nDataNotNan, nWeightsNotNan	= len(sorted_data), len(sorted_weights)

	if nDataNotNan > 0:
		# Find the cumulative sum of the weights
		cm = np.cumsum(sorted_weights)

		# Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
		whereperc = np.where(np.divide(cm,float(cm[-1])) >= perc)

		# Reurn the first data value where above is true
		out = sorted_data[whereperc[0][0]]
	else:

		print(key)
		print("[@WeightPercent:] Warning! Data all nan! Returning 0 value!")

		out = np.array([0.0])

	return out

#Binning function. Bins can be created along any property, mostly used for radial dependence of properties
def avgWeighted(x, y, min, max, s, inc, weight):       #x and y are datasets, min and max is the minimum and maxixum x value to measure from and to, s is the starting position of the data, inc is the total
                                                        #number of increments you want (i.e. 200 would repeat the loop 200 times), and weight is your data to weigh if any
	med, upper, lower, radii = np.zeros(inc-1), np.zeros(inc-1), np.zeros(inc-1), np.zeros(inc-1)
	nbins = np.divide(1, inc)
	
	for i in range(inc-1):
		pres, = np.where((x > s) & (x <= s+abs(nbins)))
		if len(pres) > 0:
			med[i]          = weighted_percentile(y[pres], weight[pres], 50)
			upper[i]        = weighted_percentile(y[pres], weight[pres],  84)
			lower[i]        = weighted_percentile(y[pres], weight[pres],  16)
			radii[i]        = s

		s = np.add(s,nbins)

	return med, upper, lower, radii

#Right, let me try and explain what this fucking stoopid piece of code does.
#This function produces radial magnitudes for the PIXELS of column density generated with get_Aslice
def colDensDist(ion):
	global ionw     #Please ignore

	#Initialise 2-D arrays for the radius and column density (column density is added to a new array which this function spits out)
	s = (len(load.data[str(ion) + 'proj']['x']), len(load.data[str(ion) + 'proj']['y']))

	load.data[str(ion) + 'rad'], load.data[str(ion) + 'ionw']       = np.zeros(s), np.zeros(s)

	#Iterate over x and y coordinates generated by get_Aslice. This creates a magnitude for each pixel
	for n in range(len(ion['x'])):
		for m in range(len(ion['y'])):
			load.data[str(ion) + 'rad'][n, m] = np.sqrt(ion['x'][n]**2 + ion['y'][m]**2)

	#Iterate over the length of x and y coordinates again, this time to generate a new array of column density for each pixel in order for the
	#column density values to match the position of the new array
	for n in range(len(ion['x'])-1):
		for m in range(len(ion['y'])-1):
			load.data[str(ion) + 'ionw'][n, m] = load.data[str(ion)][6*n + m]

	load.data[str(ion) + 'rad'], load.data[str(ion) + 'ionw']       = load.data[str(ion) + 'rad'].flatten(), load.data[str(ion) + 'ionw'].flatten()

	load.data[str(ion) + 'rad'].tolist()
	load.data[str(ion) + 'rad'] /= virialRadius  #Normalise new radius array to virial radius of halo

	#Remove any nan values from column density array
	nan_array = np.isnan(load.data[str(ion) + 'ionw'])
	not_nan_array = ~nan_array

	load.data[str(ion) + 'ionw']    = load.data[str(ion) + 'ionw'][not_nan_array]
	load.data[str(ion) + 'ionw'].tolist()

	ionw, = np.where((load.data[str(ion) + 'rad'] <= 1))       #Create conditional where whatever the name of the radius variable is only including the virial radius

	return np.log10(load.data[str(ion) + 'ionw'][ionw], where = load.data[str(ion) + 'ionw'][ionw] != 0), load.data[str(ion) + 'rad'][ionw], load.data[str(ion) + 'ionw'][ionw]

def find_indices(listToCheck, itemToFind):

	listToCheck.tolist()
	indices = []

	for idx, value in enumerate(listToCheck):
		if value == itemToFind:
			indices.append(idx)

	return indices

def depletionTime():
	return subload.data['smty'][0,0]*1e10/load.sfr

def CoolingTime():
	k_b			= 1.38e-16
	kpcincm		= c.parsec*1e3

	tcool		= load.u * 1e10 * load.data['dense'] / (abs(load.gcol[load.gcol<0]) * load.data['n_H']**2)

	csound		= sqrt(5./3. * k_b * load.data['T'] / ((load.nh * 0.6 + (1-load.nh) * 1.2) * m_p))

	lcool		= tcool * csound / kpcincm

	return lcool, tcool

def poly(x_val, n, *p):
	return p[0]*x_val**n + p[1]*x_val**(n-1) + p[2]*x_val**(n-2) + p[3]

def dynamicalTime(virialRadius):
	tDyn = virialRadius/load.data['vel']
	return tDyn

def Projection(N, x, y, grid, col, row, multiplot, label, name, cmap):
	plt.figure(N)
	fig, axs = plt.subplots(nrows=row, ncols=col, figsize = (col,row))

	if multiplot == True and row <= col:
		for row in range(row):
			for col in range(col):
				axs[row,col].pcolormesh(x[row*col], y[row*col], np.transpose(grid[row*col])/int(boxlos/pixreslos), cmap = cmap, rasterized = True, norm = matplotlib.colors.LogNorm(vmin=min, vmax=max))
				axs[row,col].plot(x1[row*col], x2[row*col], 'k-')
				axs[row,col].set_xlabel('x')
				axs[row,col].set_ylabel('y')

			pcm = axs[row,col].pcolormesh(x[row*col], y[row*col], np.transpose(grid[row*col])/int(boxlos/pixreslos), cmap = cmap, rasterized = True, norm = matplotlib.colors.LogNorm(vmin=min, vmax=max))
			axins1 = inset_axes(axs[row,col], loc = 'lower center', width = '80%', height = '5%', borderpad = -3) #bbox_to_anchor=(0.2, 1, 1, 0), bbox_transform=axs[5,0].transAxes)
			cbar1 = fig.colorbar(pcm, orientation = "horizontal", cax = axins1, pad = 0)
			cbar1.set_label(label=str(label))

	elif multiplot == True and row > col:
		for col in range(col):
			for row in range(row):
				axs[row,col].pcolormesh(x[row*col], y[row*col], np.transpose(grid[row*col])/int(boxlos/pixreslos), cmap = cmap, rasterized = True, norm = matplotlib.colors.LogNorm(vmin=min, vmax=max))
				axs[row,col].plot(x1[row*col], x2[row*col], 'k-')
				axs[row,col].set_xlabel('x')
				axs[row,col].set_ylabel('y')

			pcm = axs[row,col].pcolormesh(x[row*col], y[row*col], np.transpose(grid[row*col])/int(boxlos/pixreslos), cmap = cmap, rasterized = True, norm = matplotlib.colors.LogNorm(vmin=min, vmax=max))
			axins1 = inset_axes(axs[row,col], loc = 'lower center', width = '80%', height = '5%', borderpad = -3) #bbox_to_anchor=(0.2, 1, 1, 0), bbox_transform=axs[5,0].transAxes)
			cbar1 = fig.colorbar(pcm, orientation = "horizontal", cax = axins1, pad = 0)
			cbar1.set_label(label=str(label))

	else:
		plt.pcolormesh(x[row*col], y[row*col], np.transpose(grid[row*col])/int(boxlos/pixreslos), cmap = cmaprho, rasterized = True, norm = matplotlib.colors.LogNorm(vmin=min, vmax=max))
		plt.plot(x1[row*col], x2[row*col], 'k-')

	for ax in axs.flat:
		ax.set_aspect(1.0)
		ax.label_outer()
		ax.set_yticks([])

	plt.subplots_adjust(hspace = 0, wspace = 0)
	plt.savefig(f'/home/user/c1537815/figures/images_for_paper_2/{name}.jpg', dpi = 300, transparent = True, bbox_inches = 'tight')
	plt.close('all')

def keepLOS():
	print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
	snap.calc_sf_indizes(snap_subfind)
	if rotation_matrix is None:
		print(f"[@{int(snapNumber)}]: New rotation of snapshots")
		rotation_matrix = snap.select_halo(snap_subfind, do_rotation=True)
		rotationsavepath = savePathBaseFigureData + f"rotation_matrix_{int(snapNumber)}.h5"
		# ... save the rotation_matrix here if you want to be able to use the same matrix for future use on other snapshots of the same halo.

		# save ...

		# If we don't want the same rotation matrix for all snapshots, set rotation_matrix back to None
		if (HYPARAMS["constantRotationMatrix"] == False):
			rotation_matrix = None
	else:
		print(f"[@{int(snapNumber)}]: Existing rotation of snapshots")
		snap.select_halo(snap_subfind, do_rotation=False)
		snap.rotateto(
			rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
		)

def avg(x, y, min, max):
	med, upper, lower, radii = [], [], [], []
	nbins = np.divide(np.subtract(max, min), 100)

	for i in np.arange(min, max, 0.01):
		pres, = np.where((x >= i) & (x <= np.add(i, abs(nbins))))
		if len(pres) > 0:
			med.append(median(y[pres]))
			upper.append(np.nanpercentile(y[pres], 84))
			lower.append(np.nanpercentile(y[pres], 16))
			radii.append(i)

	return med, upper, lower, radii

def cum_avg(x, y, min, max):
	med, radii = [], []
	nbins = np.divide(np.subtract(max, min), 0.01)
	print(max, min)
	for i in np.arange(min, max, 0.01):
		pres, = np.where((x >= i) & (x <= np.add(i, abs(nbins))))
		if len(pres) > 0:
			print(i)
			med.append(np.sum(y[pres]))
			radii.append(i)

	return med, radii

def cum_sum(x, y, min, max):
	med, radii = [], []
	nbins = np.divide(np.subtract(max, min), 100)
	y2 = 0
	for i in np.arange(min, max, 0.01):
		pres, = np.where((x >= i) & (x <= np.add(i, abs(0.01))))
		if len(pres) > 0:
			med.append(sum(y[pres]) + y2)
			radii.append(i)
			y2 = sum(y[pres])

	return med, radii

def sumBins(x, y, min, max, s, inc):	#x and y are datasets, min and max is the minimum and maxixum x value to measure from and to, s is the starting position of the data, inc is the total
							#number of increments you want (i.e. 200 would repeat the loop 200 times), and weight is your data to weigh if any
	suM, x_axis = np.zeros(inc), np.zeros(inc)

	nbins = (max-min)/inc

	for i in range(inc):
		pres, = np.where((x >= s) & (x <= np.add(s, abs(nbins))))
		if len(pres) > 0:
			suM[i]		= np.sum(y[pres])
			x_axis[i]	= s
		s += nbins

	return suM, x_axis

def isContaminated(snap, halo_d, halo, haloid):
	subload = load_subfind(int(snap), dir = '/home/universe/spxfv/Auriga/' + halo_d + '/%s/output/' % halo)
	load = gadget_readsnap(int(snap), snappath = '/home/universe/spxfv/Auriga/' + halo_d + '/%s/output/' % halo, loadonlytype=[2], lazy_load=True, subfind = subload)

	load.pos *= 1e3
	load.pos -= np.array(subload.data['fpos'][haloid,:]*1e3)
	load.data['dist'] = np.sqrt((load.pos**2).sum(axis=1))

	return min(load.data['dist']) >= subload.data['frc2'][haloid]*1e3