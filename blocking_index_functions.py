import numpy as np
import iris
from datetime import datetime, timedelta


def require_persistence(ibi, period=5):
	'''
	Calculates the persistence of an instantaneously blocking index cube
	Inputs:
	- ibi = iris cube of instantaneous blocking (no persistence applied)
	- period = persistence criterion to apply (days) 
	Output:
	- iris cube of a blocking index with the persistence criterion applied
	'''
	bi = ibi.copy()
	bi_data = bi.data
	time = bi.coord('time')
	Ntimes = time.points.shape[0]
	dtime = time.points[1] - time.points[0]
	ntsteps_in_period = int(24*period/dtime)
	
	data = np.empty_like(bi.data)
	for nn in range(Ntimes - ntsteps_in_period + 1):
		nibi = bi[nn:ntsteps_in_period+nn,:,:].collapsed('time', iris.analysis.SUM)
		data[ntsteps_in_period + nn - 1,:,:] = np.where(nibi.data==ntsteps_in_period,1,0)

	pbi = iris.cube.Cube(data, long_name='Boolean blocking index',units='1',
						 dim_coords_and_dims=[(time,0),(bi.coord('latitude'),1),(bi.coord('longitude'),2)])
	
	return pbi


def backdate_persistence(pbi, period=5, daily_data=True):
	'''
	Applies the persistence criterion backwards in time too
	Inputs:
	- pbi = iris cube of blocking index with persistence applied
	- period = persistence criterion (days)
	- daily_data = whether the blocking index is daily (default is True)
	Output:
	- iris cube of blcoking index with persistence applied in both directions (the final blocking index)
	'''
	bi = pbi.copy()
	bi_data = bi.data
	time = bi.coord('time')
	Ntimes = time.points.shape[0]
	dtime = time.points[1] - time.points[0] # known (assumed) to be in hours

	data = bi.data.copy()

	if daily_data:
		ntsteps_in_period = period
	else:
		ntsteps_in_period = int(24*period/dtime)

	for nn in range(Ntimes - ntsteps_in_period + 1):
		for tt in range(ntsteps_in_period - 1):
			data[nn + tt, :, :] = np.logical_or(data[nn + tt, :, :], data[nn + ntsteps_in_period - 1, :, :])
			
	bpbi = bi.copy(data=data)

	return bpbi
	
	
def calc_Zstar(Z500):
	'''
	Calculates instantaneous anomalies from the zonal mean, Z-star
	Input:
	- Z500 = Z500 cube of any time resolution
	Output:
	- Zstar cube
	'''
	zmean = Z500.collapsed('longitude',iris.analysis.MEAN) # (Z-bar)
	Zstar = Z500 - zmean
	Zstar.rename('Z500 zonal anomaly')
	Zstar.units = 'metres'

	return Zstar
	
	
def calc_Zstarbar(Zstar):
	'''
	Takes the mean of a cube's values every month across all years
	For example, there will be a mean for January, February, etc.
	In other words, this is the climatological monthly value
	'''

	# add auxiliary coords to the cube
	coord_names = [n.name() for n in Zstar.coords()]
	if not 'day_of_year' in coord_names:
		icc.add_day_of_year(Zstar,'time','day_of_year')
	if not 'month' in coord_names:
		icc.add_month_number(Zstar,'time','month')
	if not 'month_name' in coord_names:
		icc.add_month(Zstar,'time','month_name')
	if not 'year' in coord_names:
		icc.add_year(Zstar,'time','year')

	# loop through Jan - Dec
	months = np.arange(1,13,1)
	cubelist = []
	for m in months:
		# subset cube to this month only
		m_cons = iris.Constraint(month = lambda cell: cell.point == m)
		m_cube = Zstar.extract(m_cons)
		# take mean
		m_mean = m_cube.collapsed('time', iris.analysis.MEAN)
		# add to list
		cubelist.append(m_mean)

	# concatenate cubes into one
	all_means = iris.cube.CubeList(cubelist).merge_cube()

	return all_means	


def calc_Zstarprime(Zstar, Zstarbar):
	'''
	Calculates the Z-star-prime anomalies (Zstarprime = Zstar - Zstarbar)
	Inputs:
	- Zstar
	- Zstarbar (climatology, monthly)
	Output:
	- cube of Z-star-prime
	'''
	# add auxiliary coordinates, if not there already
	for cube in [Zstar, Zstarbar]:
		coord_names = [n.name() for n in cube.coords()]
		if not 'day_of_year' in coord_names:
			icc.add_day_of_year(cube,'time','day_of_year')
		if not 'month' in coord_names:
			icc.add_month_number(cube,'time','month')
		if not 'month_name' in coord_names:
			icc.add_month(cube,'time','month_name')
		if not 'year' in coord_names:
			icc.add_year(cube,'time','year')
			
	years = np.arange(cube.coord('year').points[0], cube.coord('year').points[-1]+1, 1)
	months = np.arange(cube.coord('month').points[0], cube.coord('month').points[-1]+1, 1)
	
	# perform calculation one calendar month, one year at a time
	Zstarprime_list = []
	for y in years:
		y_cons = iris.Constraint(time = lambda cell: cell.point.year == y)
		y_Zstar = Zstar.extract(y_cons)
		if y_Zstar is None: # (checking if the dataset we provide has any data for this year)
			continue
		
		else:
			d = np.empty_like(y_Zstar.data)
			i = 0
			for m,mth in enumerate(month_numbers):
				m_cons = iris.Constraint(time = lambda cell: cell.point.month == mth)
				m_Zstar = y_Zstar.extract(m_cons)
				if m_Zstar is None: # (checking if the dataset we provide has any data for this month and year)
					continue
				else:
					m_Zstarbar = Zstarbar.extract(m_cons)
					m_data = m_Zstar.data
					m_Zstarprime_data = np.empty_like(m_data)
					j = len(m_Zstar.coord('time').points)
					for t in range(j):
						if j != 1:
							m_Zstarprime_data[t,:,:] =  m_data[t,:,:] - m_Zstarbar.data[:,:]
						else:
							m_Zstarprime_data[:,:] =  m_data[:,:] - m_Zstarbar.data[:,:]
					d[i:i+j,:,:] = m_Zstarprime_data
					i += j

			y_Zstarprime = y_Zstar.copy()
			y_Zstarprime.data = d
			Zstarprime_list.append(y_Zstarprime)    

	# concatenate into one cube
	Zstarprime = iris.cube.CubeList(Zstarprime_list).concatenate_cube()

	return Zstarprime


def calc_Zstarprime_BI(Zstarprime, thresh=100, pers=5, apply_pers=True):
	'''
	Calculates the Z-star-prime blocking index
	Inputs:
	- Zstarprime cube (of any time resolution)
	- thresh = minimum Zstarprime magnitude (m) to denote blocking
	- pers = minimum persistence criterion (days) to result in blocking
	- save_inst = whether or not to save the blocking index before persistence is applied
	Output:
	- Zstarprime daily blocking index
	'''
	cube = Zstarprime.copy()
	cube_data = cube.data
   
	# set up array to put BI in to
	time = Zstarprime.coord('time')
	lon = Zstarprime.coord('longitude')
	lat = Zstarprime.coord('latitude')
	inst_block = Zstarprime.data
	
	# find instantaneously blocked areas (where anomaly is above threshold on that particular day)
	# this will only work if "thresh" is positive (which should always be the case for blocking anticyclones anyway)
	inst_block[inst_block < thresh] = 0
	inst_block[inst_block >= thresh] = 1
	# put into a cube
	inst_bi = iris.cube.Cube(inst_block, long_name=f'Z-star-prime blocking index (thresh of {thresh}, no persistence)',
							 units='1',dim_coords_and_dims=[(time,0),(lat,1),(lon,2)])

	if apply_pers:    
		# apply persistence criterion
		copy_inst = inst_bi.copy()
		pers_bi = require_persistence(copy_inst, period=pers)
		copy_bi = pers_bi.copy()
		# backdate persistence too
		pers_bd_bi = backdate_persistence(copy_bi, period=pers, daily_data=False)
		pers_bd_bi.rename(f'Z-star-prime blocking index (thresh of {thresh}, {pers}-day persistence)')
	
		return pers_bd_bi

	else:
		return inst_bi
