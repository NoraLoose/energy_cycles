if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--filter_fac", default=32, type=int, help="Filter factor")
	parser.add_argument("--end_time", default=2900, type=int, help="End time of model output file to be filtered")
	parser.add_argument("--extended_diags", default="True", type=str, help="if you want extended diagnostics, set to 'True'")
	
	args = parser.parse_args()
	
	filter_fac = args.filter_fac
	end_time = args.end_time
	extended_diags = args.extended_diags == "True"

	#!/usr/bin/env python
	# coding: utf-8
	
	# # Step 2a: Compute Lorenz energy cycle
	
	# * This notebook assumes that `filter_data.ipynb` has been run with `lorenz = True`
	# * All terms computed in this notebook are depth-integrated, and are a function of **(time, y, x)**
	# 
	# Note: To get non-depth-integrated terms, remove all `.sum(dim='zi')` and `.sum(dim='zl')` statements.
	
	# In[1]:
	
	
	# In[2]:
	
	
	import numpy as np
	import matplotlib.pyplot as plt
	import xarray as xr
	import dask
	from dask.diagnostics import ProgressBar
	
	
	# In[3]:
	
	
	run = 'nw2_0.03125deg_N15_baseline_hmix20'
	
	
	# ## Get a view of Neverworld2 data
	
	# In[4]:
	
	
	# static file with grid information
	path = '/glade/p/univ/unyu0004/gmarques/NeverWorld2/baselines/'
	st = xr.open_dataset('%s/%s/static.nc' % (path, run), decode_times=False)
	
	
	# In[5]:
	
	
	path = '/glade/scratch/noraloose/filtered_data'
	chunks = {'time': 1, 'zl':1}
	nr_days = 100
	
	filename_av_f = '%s/%s/averages_%08d_filtered_fac%i' %(path, run, end_time-nr_days+2, filter_fac) 
	filename_sn_f = '%s/%s/snapshots_%08d_filtered_fac%i' %(path, run, end_time-nr_days+5, filter_fac) 
	
	av_f = xr.open_zarr(filename_av_f, decode_times=False)
	sn_f = xr.open_zarr(filename_sn_f, decode_times=False)
	
	
	# ## Prepare NW2 grid information
	
	# In[6]:
	
	
	from xgcm import Grid
	
	Nx = np.size(st.xh)
	Ny = np.size(st.yh)
	
	# symmetric
	coords = {
	    'X': {'center': 'xh', 'outer': 'xq'},
	    'Y': {'center': 'yh', 'outer': 'yq'},
	    'Z': {'center': 'zl', 'outer': 'zi'} 
	}
	metrics = {
	    ('X',):['dxCu','dxCv','dxT','dxBu'],
	    ('Y',):['dyCu','dyCv','dyT','dyBu'],
	    ('X', 'Y'): ['area_t', 'area_u', 'area_v']
	}
	st['zl'] = av_f['zl']
	st['zi'] = av_f['zi']
	
	grid = Grid(st, coords=coords, periodic=['X'])
	
	st['dxT'] = grid.interp(st.dxCu,'X')
	st['dyT'] = grid.interp(st.dyCv,'Y', boundary='fill')
	st['dxBu'] = grid.interp(st.dxCv,'X')
	st['dyBu'] = grid.interp(st.dyCu,'Y',boundary='fill')
	
	grid = Grid(st, coords=coords, periodic=['X'], metrics=metrics)
	grid
	
	
	# ## New dataset for Lorenz cycle
	
	# In[7]:
	
	
	ds = xr.Dataset() # new xarray dataset for terms in Lorenz cycle 
	
	ds.attrs['filter_shape'] = 'Gaussian' 
	ds.attrs['filter_factor'] = filter_fac
	
	for dim in ['time','zl','yh','xh']:
	    ds[dim] = av_f[dim]
	
	
	# # Energy reservoirs & their tendencies
	
	# ### MPE & EPE
	# 
	# \begin{align}
	#  \text{PE} = \frac{1}{2}\sum_{n=0}^{N-1} g_n' \eta_n^2
	#  \qquad
	#  \text{MPE} = \frac{1}{2}\sum_{n=0}^{N-1} g_n' \bar{\eta}_n^2
	#  \qquad
	#  \text{EPE} = \overline{\text{PE}} - \text{MPE}
	# \end{align}
	# with $g_k' = g (\rho_{k+1} - \rho_k) / \rho_o$
	
	# In[8]:
	
	
	rho_ref = 1000  # refernce density in NeverWorld2
	# reduced gravity
	gprime = 10 * grid.diff(av_f.zl,'Z',boundary='fill') / rho_ref
	gprime[15] = np.nan
	
	
	# In[9]:
	
	
	ds['MPE'] = (0.5 * gprime * av_f['e']**2).sum(dim='zi')
	ds['EPE'] = (0.5 * gprime * av_f['e2']).sum(dim='zi') - ds['MPE']
	ds['MPE'].attrs = {'units' : 'm3 s-2', 'long_name': 'Mean Potential Energy'}
	ds['EPE'].attrs = {'units' : 'm3 s-2', 'long_name': 'Eddy Potential Energy'}
	
	
	# ### MPE & EPE tendencies
	# \begin{align} 
	#     \partial_t(\text{MPE}) = \sum_{n=0}^{N-1} g_n' \bar{\eta}_n \partial_t\bar{\eta}_n
	#      \qquad
	#      \partial_t(\text{EPE}) = \frac{1}{2}\sum_{n=0}^{N-1} g_n' \overline{\partial_t(\eta^2_n}) - \partial_t(\text{MPE}) 
	# \end{align}
	
	# In[10]:
	
	
	ds['dMPEdt'] = (gprime * av_f['e'] * av_f['de_dt']).sum(dim='zi')
	ds['dMPEdt'].attrs = {'units' : 'm3 s-3', 'long_name': 'Mean Potential Energy tendency'}
	ds['dEPEdt'] = (0.5 * gprime * av_f['de2_dt']).sum(dim='zi') - ds['dMPEdt']
	ds['dEPEdt'].attrs = {'units' : 'm3 s-3', 'long_name': 'Eddy Potential Energy tendency'}
	
	
	# ### MKE & EKE
	# 
	# \begin{align}
	#  \text{KE} = \frac{1}{2}\sum_{n=1}^{N} h_n (u_n^2 + v_n^2)
	#  \qquad
	#  \text{MKE} = \frac{1}{2}\sum_{n=1}^{N} \bar{h}_n (\bar{u}_n^2 + \bar{v}_n^2)
	#  \qquad
	#  \text{EKE} = \overline{\text{KE}} - \text{MKE}
	# \end{align}
	
	# In[11]:
	
	
	MKE = 0.5 * av_f['h'] * (
	    grid.interp((av_f['u']**2).fillna(value=0), 'X')  # use simple mid-point average consistent with how KE is discretized in online model
	    + grid.interp((av_f['v']**2).fillna(value=0), 'Y')  # use simple mid-point average consistent with how KE is discretized in online model
	)
	ds['MKE'] = MKE.sum(dim='zl')
	
	#av_f['hKE'] is filtered KE = 0.5 * h * (u^2 + v^2)
	ds['EKE'] = av_f['hKE'].sum(dim='zl') - ds['MKE']  
	
	ds['MKE'].attrs = {'units' : 'm3 s-2', 'long_name': 'Mean Kinetic Energy (non-TWA)'}
	ds['EKE'].attrs = {'units' : 'm3 s-2', 'long_name': 'Eddy Kinetic Energy (non-TWA)'}
	
	
	# ### MKE & EKE tendencies
	# \begin{align} 
	#     \partial_t(\text{MKE}) =  \text{5-day-average}\left(\partial_t \text{MKE}\right) = \frac{1}{\tau_1-\tau_0} \int_{\tau_0}^{\tau_1} \partial_t (\text{MKE})\, dt = \frac{\text{MKE}(\tau_1) - \text{MKE}(\tau_0)}{\tau_1-\tau_0},
	# \end{align}
	# where the 5-day time interval is denoted by $[\tau_0, \tau_1]$. We can thus get the MKE tendencies from MKE snapshots. Similarly, we can compute
	# 
	# \begin{align} 
	#     \partial_t(\text{EKE}) =  \frac{\overline{\text{KE}}(\tau_1) - \overline{\text{KE}}(\tau_0)}{\tau_1-\tau_0} - \partial_t(\text{MKE})
	# \end{align}
	# from snapshots of `hKE` and the MKE snapshots.
	
	# In[12]:
	
	
	if np.all(av_f.average_DT == av_f.average_DT[0]):
	    deltat = av_f.average_DT[0] * 24 * 60 * 60
	else: 
	    raise AssertionError('averaging intervals vary')
	
	
	# In[13]:
	
	
	# MKE tendency
	MKE = 0.5 * sn_f['h'] * (
	    grid.interp((sn_f['u']**2).fillna(value=0), 'X')  # use simple mid-point average consistent with how KE is discretized in online model
	    + grid.interp((sn_f['v']**2).fillna(value=0), 'Y')  # use simple mid-point average consistent with how KE is discretized in online model
	)
	
	if np.array_equal(av_f.time_bnds[:,1], sn_f.time):
	    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
	        dMKEdt = (MKE - MKE.shift(time=1)) / deltat
	        dMKEdt['time'] = av_f['h'].time
	else: 
	    raise AssertionError('av and sn datasets not compatitble')
	
	dMKEdt = dMKEdt.where(av_f.time > av_f.time[0])  
	dMKEdt = dMKEdt.chunk({'time': 1})
	ds['dMKEdt'] = dMKEdt.sum(dim='zl')
	
	ds['dMKEdt'].attrs = {'units' : 'm3 s-3', 'long_name': 'Mean Kinetic Energy tendency (non-TWA)'}
	
	# EKE tendency
	hKE = sn_f['hKE']
	if np.array_equal(av_f.time_bnds[:,1], sn_f.time):
	    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
	        hKEdt = (hKE - hKE.shift(time=1)) / deltat
	        hKEdt['time'] = av_f['h'].time
	else: 
	    raise AssertionError('av and sn datasets not compatitble')
	 
	hKEdt = hKEdt.where(av_f.time > av_f.time[0])  
	hKEdt = hKEdt.chunk({'time': 1})
	ds['dEKEdt'] = hKEdt.sum(dim='zl') - ds['dMKEdt']
	ds['dEKEdt'].attrs = {'units' : 'm3 s-3', 'long_name': 'Eddy Kinetic Energy tendency (non-TWA)'}
	
	
	# ## Energy conversion terms (cf. Figure 3a in Loose et al., 2022)
	
	# ### EKE production $\Sigma^L$
	# 
	# \begin{align}
	#    \Sigma^L = \underbrace{- \sum_{n=1}^N\overline{h_n (u_n \partial_x M_n + v_n \partial_y M_n)}}_{\overline{\text{PE_to_KE+KE_BT}}} + \sum_{n=1}^N\bar{h}_n(\bar{u}_n \underbrace{\overline{\partial_x M_n}}_{-\overline{\text{PFu+u_BT_accel_visc_rem}}} + \bar{v}_n \underbrace{\overline{\partial_y M_n}}_{-\overline{\text{PFv+v_BT_accel_visc_rem}}} )
	# \end{align}
	
	# In[14]:
	
	
	EKE_production = av_f['PE_to_KE+KE_BT'] - av_f['h'] / st['area_t']  * (
	    grid.interp((av_f['u'] * av_f['PFu+u_BT_accel_visc_rem'] * st['area_u']).fillna(value=0), 'X')
	    + grid.interp((av_f['v'] * av_f['PFv+v_BT_accel_visc_rem'] * st['area_v']).fillna(value=0), 'Y')
	)
	ds['EKE_production'] = EKE_production.sum(dim='zl')
	ds['EKE_production'].attrs = {'units' : 'm3 s-3', 'long_name': 'EKE production (non-TWA)'}
	
	
	# ### Baroclinic conversion $\Gamma^L$
	# 
	# With 
	# $$\mathcal{E} = \sum_{n=1}^N \left(\nabla\cdot(\overline{h_n\mathbf{u}_n}) \underbrace{- \overline{\nabla\cdot(h_n\mathbf{u}_n)}}_{\overline{\partial_t h_n}}\right) \bar{M}_n,
	# \qquad
	#     M_n = \sum_{k=0}^{n-1} g_k' \eta_k
	# $$ 
	# we have
	# 
	# \begin{align}
	#  \Gamma^L & = -\left(\sum_{n=1}^N(\overline{h_n u_n}\cdot \partial_x \bar{M}_n + \overline{h_n v_n}\cdot \partial_y \bar{M}_n ) -\sum_{n=1}^N\bar{h}_n(\bar{u}_n \overline{\partial_x M_n} + \bar{v}_n \overline{\partial_y M_n})\right) - \mathcal{E}\\
	#  & =\Sigma^L
	#  \underbrace{-  \sum_{n=1}^N \left(\overline{M_n \left(\partial_x (h_n u_n) + \partial_y (h_n v_n)\right)} -\bar{M}_n\left(\overline{\partial_x (h_n u_n) + \partial_y (h_n v_n}\right)\right)}_\text{EPE tendency, see eqn (A5)}
	#  + \underbrace{ \sum_{n=1}^N \left(\overline{\partial_x(h_n u_n M_n) + \partial_y(h_n v_n M_n)} 
	#  - \left(\partial_x(\overline{h_n u_n} \bar{M}_n) + \partial_y (\overline{h_n v_n} \bar{M}_n)\right)\right)}_{-\mathcal{D},\text{ see eqn (A13)}}
	#  \end{align}
	# 
	# where the last identity follows from equations (A12) and (A13) in Loose et al. (2022). 
	# 
	# Thus, we have two options to compute $\Gamma^L$: via the first line, or via the second line. The two options would give the same result in a non-discretized world, but lead to small differences in our discretized world (in time and space). For Figure 8 in Loose et al. (2022), we computed $\Gamma^L$ via the second option, and this is the default below (`BC_conversion`). 
	# 
	# If you want to diagnose $\Gamma^L$ via the first option (`BC_conversion_alt`), set `extended_diags=True` at the top of this notebook.
	
	# In[15]:
	
	
	MP = grid.cumsum(gprime * av_f['e'],'Z')  # Montgomery potential
	av_f['MP'] = MP.transpose('time', 'zl', 'yh', 'xh')  # reorder coordinates
	av_f['MP'].attrs = {'units' : 'm2 s-2', 'long_name': 'Montgomery potential'}
	
	
	# In[16]:
	
	
	# compute eddy pressure flux divergence, div = - D
	uhM_mean = av_f['uh'] * grid.interp(av_f['MP'].fillna(value=0),'X', metric_weighted=['X','Y'])
	uflux = grid.diff(uhM_mean.fillna(value=0),'X')
	vhM_mean = av_f['vh'] * grid.interp(av_f['MP'].fillna(value=0),'Y', metric_weighted=['X','Y'], boundary='fill')
	vflux = grid.diff(vhM_mean.fillna(value=0),'Y')
	div_mean = (uflux + vflux).where(st.wet) / st.area_t  # finite volume discretization
	
	# av_f[uhM_div] = filtered(div(uhM))
	div = av_f['uhM_div'] - div_mean
	div = div.chunk({'yh':Ny, 'xh':Nx})
	
	ds['BC_conversion'] = (ds['EKE_production'] + div.sum(dim='zl') + ds['dEPEdt']).chunk({'yh':Ny, 'xh':Nx})
	ds['BC_conversion'].attrs = {'units' : 'm3 s-3', 'long_name': 'baroclinic conversion (non-TWA)'}
	
	if extended_diags:
	    # extra term E
	    uflux = grid.diff(av_f['uh'].fillna(value=0), 'X')
	    vflux = grid.diff(av_f['vh'].fillna(value=0), 'Y')
	    div = (uflux + vflux).where(st.wet) / st.area_t  # finite volume discretization
	    extra_term = - av_f['MP'] * (div + av_f['dhdt']) 
	
	    conversion = (
	        - grid.interp((av_f['uh'] / st['dyCu']) * (grid.derivative(av_f['MP'], 'X')), 'X')
	        - grid.interp((av_f['vh'] / st['dxCv']) * (grid.derivative(av_f['MP'], 'Y', boundary='fill')), 'Y')
	        - av_f['h'] / st['area_t'] * (
	            grid.interp((av_f['u'] * (av_f['PFu+u_BT_accel_visc_rem']) * st['area_u']).fillna(value=0),'X')
	            + grid.interp((av_f['v'] * (av_f['PFv+v_BT_accel_visc_rem']) * st['area_v']).fillna(value=0),'Y') 
	        )
	    )
	
	    ds['BC_conversion_alt'] = ((conversion - extra_term).sum(dim='zl')).chunk({'yh':Ny, 'xh':Nx})
	    ds['BC_conversion_alt'].attrs = {'units' : 'm3 s-3', 'long_name': 'baroclinic conversion (non-TWA), alternative computation'}
	
	
	# ### MKE --> MPE
	# \begin{align}
	#    (\text{MKE} \to \text{MPE}) & = \underbrace{\sum_{n=1}^N\bar{h}_n \left(\bar{u}_n \overline{\partial_x M_n} + \bar{v}_n \overline{\partial_y M_n}\right)}_\text{see Figure 3a}\\
	#     & = \Gamma^A + \partial_t \text{MPE} +  \sum_n \nabla\cdot\left( \overline{h_n\mathbf{u}_n} \bar{M}_n\right),
	# \end{align}
	# 
	# where
	# $$
	# \Gamma^A =
	# -\left(\sum_{n=1}^N\overline{h_n\mathbf{u}_n}\cdot\nabla \bar{M}_n -\sum_{n=1}^N\bar{h}_n\bar{\mathbf{u}}_n\cdot\overline{\nabla M_n}\right) - \mathcal{E}
	# $$
	# and
	# $$
	#     \partial_t\text{MPE} = - \sum_n \nabla\cdot\left( \overline{h_n\mathbf{u}_n} \bar{M}_n\right) + \sum_n \overline{h_n \mathbf{u}_n} \cdot \nabla \bar{M}_n + \mathcal{E}
	# $$
	# 
	# Again, we have two options to compute $(\text{MKE} \to \text{MPE})$ via the first line, or via the second line. For Figure 8 in Loose et al. (2022), we chose the second option, and this is the default below (`MKE_to_MPE`). 
	# 
	# If you want to diagnose $(\text{MKE} \to \text{MPE})$ via the first option (`MKE_to_MPE_alt`), set `extended_diags=True` at the top of this notebook.
	
	# In[17]:
	
	
	uflux = grid.diff((av_f['uh'] * grid.interp(av_f['MP'].fillna(value=0), 'X', metric_weighted=['X','Y'])).fillna(value=0),'X')
	vflux = grid.diff((av_f['vh'] * grid.interp(av_f['MP'].fillna(value=0), 'Y', metric_weighted=['X','Y'], boundary='fill')).fillna(value=0),'Y')
	div = (uflux + vflux).where(st.wet) / st.area_t  # finite volume discretization
	ds['MKE_to_MPE'] = (ds['BC_conversion'] + ds['dMPEdt'] + div.sum(dim='zl')).chunk({'yh':Ny, 'xh':Nx})
	ds['MKE_to_MPE'].attrs = {'units' : 'm3 s-3', 'long_name': 'MKE to MPE conversion (non-TWA)'}
	
	if extended_diags:
	    MKE_to_MPE_alt =  - av_f['h'] / st['area_t']  * (
	        grid.interp((av_f['u'] * (av_f['PFu+u_BT_accel_visc_rem']) * st['area_u']).fillna(value=0),'X')
	        + grid.interp((av_f['v'] * (av_f['PFv+v_BT_accel_visc_rem']) * st['area_v']).fillna(value=0),'Y')
	    )
	    ds['MKE_to_MPE_alt'] = MKE_to_MPE_alt.sum(dim='zl')
	    ds['MKE_to_MPE_alt'].attrs = {'units' : 'm3 s-3', 'long_name': 'MKE to MPE conversion (non-TWA), alternative computation'}
	
	
	# ### EKE transport
	
	# \begin{align}
	# \mathcal{T}^L & =
	# \sum_{n=1}^N \left[
	# \underbrace{\overline{\nabla\cdot\left(
	# \mathbf{u}_n\frac{h_n|\mathbf{u}_n|^2}{2}\right)}}_\overline{-\text{KE_adv}}
	# -\nabla\cdot\left(
	# \bar{\mathbf{u}}_n\underbrace{\frac{\bar{h}_n|\bar{\mathbf{u}}_n|^2}{2}}_\text{MKE}
	# \right)\right],
	# \end{align}
	
	# In[21]:
	
	
	MKE = 0.5 * av_f['h'] * (
	    grid.interp((av_f['u']**2).fillna(value=0), 'X')  # use simple mid-point average consistent with how KE is discretized in online model
	    + grid.interp((av_f['v']**2).fillna(value=0), 'Y')  # use simple mid-point average consistent with how KE is discretized in online model
	)
	
	MKE_transport =  1 / st.area_t * (
	        grid.diff((grid.interp(MKE.fillna(value=0),'X') * av_f['u'] * st.dyCu).fillna(value=0),'X')
	        + grid.diff((grid.interp(MKE.fillna(value=0),'Y',boundary='fill') * av_f['v'] * st.dxCv).fillna(value=0),'Y')
	)
	MKE_transport = MKE_transport.chunk({'yh':Ny, 'xh':Nx})
	
	ds['MKE_transport'] = MKE_transport.sum(dim='zl')
	ds['MKE_transport'].attrs = {'units' : 'm3 s-3', 'long_name': 'MKE transport (non-TWA)'}
	
	EKE_transport = - av_f['KE_adv'] - MKE_transport
	ds['EKE_transport'] = EKE_transport.sum(dim='zl')
	ds['EKE_transport'].attrs = {'units' : 'm3 s-3', 'long_name': 'EKE transport (non-TWA)'}
	
	
	# ### Work done by eddy momentum fluxes
	
	# \begin{align}
	# \Pi^L & =  \sum_{n=1}^N \bar{h}_n\bar{\mathbf{u}}_n \cdot \overline{\mathbf{f}\times\mathbf{u}_n}
	# + \sum_{n=1}^N \bar{h}_n \bar{\mathbf{u}}_n \cdot \left(\overline{\mathbf{u}_n\cdot \nabla\mathbf{u}_n}- \bar{\mathbf{u}}_n\cdot \nabla\bar{\mathbf{u}}_n\right) + \sum_{n=1}^N \frac{|\bar{\mathbf{u}}_n|^2}{2}\left(\overline{\nabla\cdot(h_n\mathbf{u}_n)} - \nabla\cdot(\bar{h}_n\bar{\mathbf{u}}_n)\right)\\ 
	# &= \sum_{n=1}^N \bar{h}_n \bar{\mathbf{u}}_n \cdot \left(\overline{\underbrace{\mathbf{f}\times\mathbf{u}_n + \mathbf{u}_n\cdot \nabla\mathbf{u}_n}_{-\text{CA[uv]_visc_rem}}}\right)
	#  + \sum_{n=1}^N \frac{|\bar{\mathbf{u}}_n|^2}{2}\left(\overline{\underbrace{\nabla\cdot(h_n\mathbf{u}_n)}_{\text{-dhdt}}}\right) - \text{MKE transport}
	# \end{align}
	
	# The first line is equation (A9) in Loose et al. (2022), and the second line uses
	# $$
	# \text{MKE transport} =\nabla\cdot\left(
	# \bar{\mathbf{u}}_n\frac{\bar{h}_n|\bar{\mathbf{u}}_n|^2}{2}
	# \right) = \frac{|\mathbf{u}_n|^2}{2} \nabla\cdot(\bar{h}_n\bar{\mathbf{u}}_n) + \bar{h}_n\bar{\mathbf{u}}_n\cdot (\bar{\mathbf{u}}_n\cdot \nabla)\bar{\mathbf{u}}_n
	# $$
	# We will use the second line to compute $\Pi^L$ below.
	
	# In[19]:
	
	
	MKE = 0.5 * av_f['h'] * (
	    grid.interp((av_f['u']**2).fillna(value=0), 'X')  # use simple mid-point average consistent with how KE is discretized in online model
	    + grid.interp((av_f['v']**2).fillna(value=0), 'Y')  # use simple mid-point average consistent with how KE is discretized in online model
	)
	MKE_transport =  1 / st.area_t * (
	        grid.diff((grid.interp(MKE.fillna(value=0),'X') * av_f['u'] * st.dyCu).fillna(value=0),'X')
	        + grid.diff((grid.interp(MKE.fillna(value=0),'Y',boundary='fill') * av_f['v'] * st.dxCv).fillna(value=0),'Y')
	)
	MKE_transport = MKE_transport.chunk({'yh':Ny, 'xh':Nx})
	
	ke_u = - av_f['u'] * (av_f['CAu_visc_rem']) * st['area_u']
	ke_v = - av_f['v'] * (av_f['CAv_visc_rem']) * st['area_v']
	
	work_eddy_momentum_fluxes = (
	    av_f['h'] / st['area_t'] * (grid.interp(ke_u.fillna(value=0), 'X') + grid.interp(ke_v.fillna(value=0), 'Y'))
	    - MKE / av_f['h'] * av_f['dhdt']
	    - MKE_transport
	)
	work_eddy_momentum_fluxes = work_eddy_momentum_fluxes.chunk({'yh':Ny, 'xh':Nx})
	
	ds['work_eddy_momentum_fluxes'] = work_eddy_momentum_fluxes.sum(dim='zl')
	ds['work_eddy_momentum_fluxes'].attrs = {
	    'units' : 'm3 s-3', 
	    'long_name': 'Energy exchange between MKE and EKE through eddy momentum fluxes (non-TWA); positive means EKE reservoir receives energy'
	}
	
	
	# ### Wind work on MKE & EKE reservoir
	# 
	# Wind work on MKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{F^{u,\text{wind}}_n} + \bar{v}_n \overline{F^{v,\text{wind}}_n}), 
	# \end{align}
	# 
	# Wind work on EKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \overline{\underbrace{h_n (u_n F^{u,\text{wind}}_n + v_n F^{v,\text{wind}}_n)}_\text{KE_stress}} 
	#     - \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{F^{u,\text{wind}}_n} + \bar{v}_n \overline{F^{v,\text{wind}}_n}), 
	# \end{align}
	
	# In[20]:
	
	
	MKE_wind_stress = av_f['h'] * (
	    grid.interp((av_f['u'] * (av_f['du_dt_str_visc_rem'])).fillna(value=0), 'X', metric_weighted=['X','Y']) 
	    + grid.interp((av_f['v'] * (av_f['dv_dt_str_visc_rem'])).fillna(value=0), 'Y', metric_weighted=['X','Y']) 
	)
	ds['MKE_wind_stress'] = MKE_wind_stress.sum(dim='zl')
	ds['MKE_wind_stress'].attrs = {'units' : 'm3 s-3', 'long_name': 'Wind work on MKE reservoir (non-TWA)'}
	
	ds['EKE_wind_stress'] = av_f['KE_stress'].sum(dim='zl') - ds['MKE_wind_stress']
	ds['EKE_wind_stress'].attrs = {'units' : 'm3 s-3', 'long_name': 'Wind work on EKE reservoir (non-TWA)'}
	
	
	# ### Bottom drag and vertical friction on MKE & EKE reservoir
	# 
	# Work of vertical stresses on MKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{F^{u}_n} + \bar{v}_n \overline{F^{v}_n}), 
	# \end{align}
	# 
	# Work of vertical stresses on EKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \overline{\underbrace{h_n (u_n F^{u}_n + v_n F^{v}_n)}_\text{KE_visc}} 
	#     - \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{\underbrace{F^{u}_n}_\text{du_dt_visc_rem}} + \bar{v}_n \overline{\underbrace{F^{v}_n}_\text{dv_dt_visc_rem}}), 
	# \end{align}
	# 
	# where $\mathbf{F}_n = (F^u_n, F^v_n)$ are the vertical stresses acting on layer $n$.
	# 
	# The vertical stresses can be further decomposed into contributions by wind stress, bottom drag, and vertical friction:
	# \begin{align*}
	# \overline{\mathbf{F}_n} & = \overline{\mathbf{F}^\text{wind}_n} + \overline{\mathbf{F}^\text{drag}_n} + \overline{\mathbf{F}^\text{visc}_n}, \\
	# \overline{h_n \mathbf{u}_n \cdot \mathbf{F}_n} & = \overline{h_n \mathbf{u}_n \cdot \mathbf{F}^\text{wind}_n} + \overline{h_n \mathbf{u}_n \cdot \mathbf{F}^\text{drag}_n} + \overline{h_n \mathbf{u}_n \cdot \mathbf{F}^\text{visc}_n}.
	# \end{align*}
	# 
	# For the wind contributions, we have diagnostics from the model (see previous section). We disentangle the remainders $\overline{\mathbf{F}_n} - \overline{\mathbf{F}^\text{wind}_n}$ and $\overline{h_n \mathbf{u}_n \cdot \mathbf{F}_n}- \overline{h_n \mathbf{u}_n \cdot \mathbf{F}^\text{wind}_n }$ into contributions from bottom drag and vertical friction offline, by classifying them as a contribution by bottom drag if the lower interface of layer $n$ is within the bottom boundary layer, and as a contribution by vertical friction otherwise. This is handled by applying `av_f.bottom_mask`.
	
	# In[21]:
	
	
	# MKE_vertical_stresses and EKE_vertical_stresses include contributions from bottom drag, vertical viscosity, and wind stress
	MKE_vertical_stresses = av_f['h'] * (
	    grid.interp((av_f['u'] * (av_f['du_dt_visc_rem'])).fillna(value=0), 'X', metric_weighted=['X','Y']) 
	    + grid.interp((av_f['v'] * (av_f['dv_dt_visc_rem'])).fillna(value=0), 'Y', metric_weighted=['X','Y']) 
	)
	EKE_vertical_stresses = av_f['KE_visc'] - MKE_vertical_stresses
	
	# tease out bottom drag and vertical viscosity contribution
	MKE_bottom_drag = (
	        MKE_vertical_stresses - MKE_wind_stress  # subtract 3D wind stress contribution from vertical stresses
	    ) * av_f['bottom_mask']  # extract the bit in the bottom boundary layer
	ds['MKE_bottom_drag'] = MKE_bottom_drag.sum(dim='zl')
	ds['MKE_bottom_drag'].attrs = {'units' : 'm3 s-3', 'long_name': 'Bottom drag work on MKE reservoir (non-TWA)'}
	
	ds['MKE_vertical_viscosity'] = MKE_vertical_stresses.sum(dim='zl') - ds['MKE_wind_stress'] - ds['MKE_bottom_drag']
	ds['MKE_vertical_viscosity'].attrs = {'units' : 'm3 s-3', 'long_name': 'Vertical friction work on MKE reservoir (non-TWA)'}
	
	EKE_wind_stress = av_f['KE_stress'] - MKE_wind_stress
	EKE_bottom_drag = (
	        EKE_vertical_stresses - EKE_wind_stress  # subtract 3D wind stress contribution from vertical stresses
	    ) * av_f['bottom_mask']  # extract the bit in the bottom boundary layer
	ds['EKE_bottom_drag'] = EKE_bottom_drag.sum(dim='zl')
	ds['EKE_bottom_drag'].attrs = {'units' : 'm3 s-3', 'long_name': 'Bottom drag work on EKE reservoir (non-TWA)'}
	
	ds['EKE_vertical_viscosity'] = EKE_vertical_stresses.sum(dim='zl') - ds['EKE_wind_stress'] - ds['EKE_bottom_drag']
	ds['EKE_vertical_viscosity'].attrs = {'units' : 'm3 s-3', 'long_name': 'Vertical friction work on EKE reservoir (non-TWA)'}
	
	
	# ### Work of horizontal friction on MKE & EKE reservoir
	# 
	# Work of horizontal friction on MKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{F^{u,h}_n} + \bar{v}_n \overline{F^{v,h}_n}), 
	# \end{align}
	# 
	# Work of horizontal friction on EKE reservoir:
	# \begin{align}
	#     \sum_{n=1}^N \overline{\underbrace{h_n (u_n F^{u,h}_n + v_n F^{v,h}_n)}_\text{KE_horvisc}} 
	#     - \sum_{n=1}^N \bar{h}_n (\bar{u}_n \overline{F^{u,h}_n} + \bar{v}_n \overline{F^{v,h}_n}), 
	# \end{align}
	
	# In[22]:
	
	
	MKE_horizontal_viscosity = av_f['h'] * (
	    grid.interp((av_f['u'] * (av_f['diffu_visc_rem'])).fillna(value=0), 'X', metric_weighted=['X','Y']) 
	    + grid.interp((av_f['v'] * (av_f['diffv_visc_rem'])).fillna(value=0), 'Y', metric_weighted=['X','Y']) 
	)
	ds['MKE_horizontal_viscosity'] = MKE_horizontal_viscosity.sum(dim='zl')
	ds['MKE_horizontal_viscosity'].attrs = {'units' : 'm3 s-3', 'long_name': 'Horizontal friction on MKE reservoir (non-TWA)'}
	
	ds['EKE_horizontal_viscosity'] = av_f['KE_horvisc'].sum(dim='zl') - ds['MKE_horizontal_viscosity']
	ds['EKE_horizontal_viscosity'].attrs = {'units' : 'm3 s-3', 'long_name': 'Horizontal friction on EKE reservoir (non-TWA)'}
	
	
	# ## Save to netcdf
	
	# In[23]:
	
	
	ds
	
	
	# In[24]:
	
	
	scratchpath = '/glade/scratch/noraloose/filtered_data'
	filename = '%s/%s/lorenz_cycle_%08d_fac%i' %(scratchpath, run, end_time-nr_days+2, filter_fac) 
	filename
	
	
	# In[31]:
	
	
	ds.to_zarr(filename)
	
	
	# In[ ]:
	
	
	
	
