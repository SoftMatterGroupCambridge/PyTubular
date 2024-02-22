#!/usr/bin/env python

import numpy as np
import sympy
import time

import PyTubular.sympy_definitions.eigenvalues as eigenvalues
import PyTubular.sympy_definitions.eigenfunctions as eigenfunctions
import PyTubular.sympy_definitions.exit_rate as exit_rate
import PyTubular.sympy_definitions.normalized_density as normalized_density


'''
Bibliography

[1] Sojourn probabilities in tubes and pathwise irreversibility for 
	Ito processes. Julian Kappler, Michael E. Cates, Ronojoy Adhikari
	https://arxiv.org/abs/2009.04250

''';



class symbolic:
	'''
	This class contains both the perturbative results as well as
	numerical methods for solving the Fokker-Planck equation inside the tube

	General workflow for perturbative expressions:
	- Define symbolic drift, diffusivity, and path.
	- Generate lambda functions for D_k, E_k, dD_k/dt, dE_k/dt
	- Evaluate D_k, E_k, dD_k/dt, dE_k/dt for given times
	- Use evaluated D_k, E_k, dD_k/dt, dE_k/dt to evaluate perturbative expressions
	'''
	def __init__(self,parameters={}):
		'''Initialization method of the class

		Sets simulation parameters and defines symbolic length-, time-,
		and diffusivity scales

		Args:
			parameters (dict): A dictionary with the parameters defined by the
							   user. This is optional, all parameters can be
							   also set after initialization of the class.

		Returns:
			None
		'''
		#
		# basic units of length, time, and diffusivity
		self.L = sympy.symbols('L',real=True,positive=True)
		self.T = sympy.symbols('T',real=True,positive=True)
		self.D0 = self.L**2 / self.T
		self.tD = self.T
		#
		# other symbolic variables
		self.t = sympy.symbols('t',real=True,nonnegative=True)
		self.x = sympy.symbols('x',real=True)
		self.xDL = sympy.symbols(r'\tilde{x}',real=True)
		self.n = sympy.symbols('n',real=True,integer=True,positive=True)
		self.epsilon = sympy.symbols(r'\epsilon',real=True,positive=True)
		self.d_log_eps = sympy.symbols('\dot{\epsilon}/\epsilon',real=True)
		self.ti = 0 # initial time
		self.tf_set = False # final time not set
		#
		self.N_max = 6
		self.Dk = sympy.symbols('D_0:%d'%self.N_max)
		self.d_Dk = sympy.symbols('\dot{D}_0:%d'%self.N_max)
		self.Ek = sympy.symbols('E_0:%d'%self.N_max)
		self.d_Ek = sympy.symbols('\dot{E}_0:%d'%self.N_max)
		#
		# temporal parameters
		self.Nt = 101 # temporal datapoints
		self.Nt_set = False # for numerical simulation. If self.Nt is not
		# set by hand then a stability criterion for the timestep  will be
		# used to determine self.Nt
		#
		# Further parameters for numerical simulation
		self.saving_stride = 10 # output frequency in multiples of simulation
		                       # timestep
		self.set_Nx() # set spatial discretization
		self.P0_set = False # initial condition not set (default is sin in tube)
		#
		self.verbose = True
		#
		self.E_functions_constructed = False
		self.D_functions_constructed = False
		#
		self.factorials = []
		for i in range(self.N_max):
		    self.factorials.append(np.math.factorial(i))
		#
		# get eigenfunctions and set local variables
		self.eigenvalues_dict = {}
		for i,term in (eigenvalues.eigenvalues_dict).items():
			self.eigenvalues_dict[i] = self.__substitute_local_variables(
							expression=term,
							remote=eigenvalues
							)

		# get eigenfunctions and set local variables
		self.eigenfunctions_dict = {}
		for i,term in (eigenfunctions.eigenfunctions_dict).items():
			self.eigenfunctions_dict[i] = self.__substitute_local_variables(
							expression=term,
							remote=eigenfunctions
							)

		self.exit_rate_dict = {}
		for i,term in (exit_rate.exit_rate_dict).items():
			self.exit_rate_dict[i] = self.__substitute_local_variables(
							expression=term,
							remote=exit_rate
							)

		self.normalized_density_dict = {}
		for i,term in (normalized_density.normalized_density_dict).items():
			self.normalized_density_dict[i] = self.__substitute_local_variables(
							expression=term,
							remote=normalized_density
							)

	def set_parameters(self,parameters):
		'''Sets parameters of an instance of the class

		With this method the parameters of an existing instance of the class
		can be set or changed.

		Args:
			parameters (dict): A dictionary with the parameters defined by the
							   user.

		Returns:
			None
		'''
		#
		try:
			self.set_a(a=parameters['a'])
		except KeyError:
			pass
		#
		try:
			self.set_D(D=parameters['D'])
		except KeyError:
			pass
		#
		try:
			self.set_path(path=parameters['path'])
		except KeyError:
			pass
		#
		try:
			self.set_R(R=parameters['R'])
		except KeyError:
			pass
		#
		try:
			self.set_Nt(Nt=parameters['Nt'])
		except KeyError:
			pass
		#
		try:
			self.set_ti(ti=parameters['ti'])
		except KeyError:
			pass
		#
		try:
			self.set_tf(tf=parameters['tf'])
		except KeyError:
			pass
		#
		try:
			self.set_P0(P0=parameters['P0'])
		except KeyError:
			pass

	def set_P0(self,P0):
		'''Set initial distribution for numerical simulation

		This method sets the initial distribution for the simulation of the
		Fokker-Planck equation.

		Args:
			P0 (sympy expression): Initial distribution P0 as function of
								   self.x

		Returns:
			None
		'''
		self.P0 = P0
		self.P0_set = True

	def set_ti(self,ti):
		'''Set initial time for analytical calculation and numerical simulation

		This method sets the initial time for both the analytical and numerical
		functionalities of the class. Upon initialization, the default value
		for the initial time is 0.

		Args:
			ti (sympy expression): Initial time ti

		Returns:
			None
		'''
		self.ti = ti

	def set_tf(self,tf):
		'''Set final time for analytical calculation and numerical simulation

		This method sets the final time for both the analytical and numerical
		functionalities of the class

		Args:
			tf (sympy expression): Final time tf

		Returns:
			None
		'''
		self.tf = tf
		self.tf_set = True

	def set_Nt(self,Nt):
		'''Set number of timesteps in time interval [ti, tf]

		This method sets the number of time steps in the interval [ti, tf]
		for both evaluating analytical and numerical functionalities of the
		class. Upon initialization, the default value for Nt is 101.

		If no value for Nt is set by the user, then for numerical simulations
		a number of time steps will be estimated by a numerical stability
		criterion (see method XXXXX)

		Args:
			Nt (int): number of time steps

		Returns:
			None
		'''
		self.Nt = Nt
		self.Nt_set = True

	def set_Nx(self,Nx=101):
		'''Set number of spatial discretization points inside the tube

		This method sets the number of spatial discretization points in the
		dimensionless tube coordinates, i.e. in the interval [-1,1],
		for both evaluating analytical and numerical functionalities of the
		class.

		Upon initialization of the class, the default value for Nx is 101.

		Note that the boundary values x = -1 and x = 1 are not included in
		the count Nx. The number Nx refers to the discretiztion points that
		are strictly inside the interval [-1,1].

		Args:
			Nx (int): number of spatial discretization points

		Returns:
			None
		'''
		self.Nx = Nx
		self.dxDL = 2/(self.Nx + 1) # dimensionless
		self.one_by_dxDL_squared = 1/self.dxDL**2
		self.xDL_array =  np.arange(self.Nx+2,dtype=float)*self.dxDL - 1

	def set_a(self,a):
		'''Set drift

		This method sets the drift profile for both analytical and numerical
		functionalities of the the class.

		Args:
			a (sympy expression): Drift profile as function of self.x, self.t,
								  and in units self.T/self.T.
								  Example:
								    a = self.T/self.T * \
								  		sympy.sin(self.x/self.L - self.t/self.T)

		Returns:
			None
		'''
		self.a = a # in units of self.L/self.T
		self.E_functions_constructed = False
		#
		expr = sympy.expand( a / (self.L/self.T) )
		expr = expr.subs(self.x,self.x*self.L)
		expr = expr.subs(self.t,self.t*self.tD)
		expr = sympy.expand( expr )
		self.a_lambda = sympy.lambdify((self.x,self.t),expr)

	def set_D(self,D):
		'''Set diffusivity

		This method sets the diffusivity profile for both analytical and
		numerical functionalities of the the class.

		To be consistent with the assumptions of the Fokker-Planck equation,
		the diffusivity should always be strictly positive.

		Args:
			D (sympy expression): Diffusivity profile as function of self.x,
								  self.t, and in units
								  self.D0 = self.L**2 / self.T
								  Example:
								    D = self.D0*(1+0.1*sympy.sin(self.x/self.L))

		Returns:
			None
		'''
		self.D = D
		#
		self.D_functions_constructed = False
		#
		expr = D/self.D0
		expr = expr.subs(self.x,self.x*self.L)
		expr = expr.subs(self.t,self.t*self.tD)
		expr = sympy.expand( expr )
		#
		self.D_lambda = np.vectorize( sympy.lambdify((self.x,self.t),expr) )

	def set_path(self,path):
		'''Set reference path

		This method sets the reference path for both analytical and numerical
		functionalities of the the class.

		Args:
			path (sympy expression): path as function of self.t and in units
									 self.L.
									 Example:
									 	path = self.L * ( 1 + self.t/self.T )

		Returns:
			None
		'''
		self.path = path
		self.dot_path = sympy.diff(path,self.t,1).doit()
		#
		if self.dot_path == 0: # if path is constant
			self.path_lambda = np.vectorize( sympy.expand(sympy.lambdify(
												self.t, 
												path/self.L ) ) )
		else:
			self.path_lambda = np.vectorize( 
								sympy.lambdify( \
										self.t, \
										sympy.expand(
												path.subs(self.t,
														self.t*self.tD)/self.L
												) \
											) \
										)
		if sympy.diff( self.dot_path ,self.t,1).doit() == 0: # dot_path constant
			expr = sympy.expand( self.dot_path*self.tD/self.L )
		else:
			expr = sympy.expand( (self.dot_path).subs(
									self.t,self.t*self.tD)*self.tD/self.L )
		self.dot_path_lambda = np.vectorize( sympy.lambdify(self.t,expr ) )
		#
		self.E_functions_constructed = False
		self.D_functions_constructed = False

	def set_R(self,R):
		'''Set tube radius

		This method sets the tube radius for both analytical and numerical
		functionalities of the the class.

		Since in PyTubular we consider finite-radius tubes around a reference
		path, the function R should always be positive within the temporal
		domain [ti,tf].

		Args:
			R (sympy expression): Tube radius as function of self.t and in units
								  self.L.
								  Example:
									 R = self.L * (1+sympy.exp(-self.t/self.T))

		Returns:
			None
		'''
		self.R = R
		self.epsilon = R/self.L
		expr = sympy.expand( (self.epsilon).subs(self.t,self.t*self.tD) )
		self.epsilon_lambda = np.vectorize( sympy.lambdify(self.t,expr) )
		#
		expr = sympy.expand( sympy.diff(self.epsilon,self.t,1).doit() * self.tD )
		expr = sympy.expand( expr.subs(self.t, self.t*self.tD) )
		self.d_epsilon_lambda = np.vectorize( sympy.lambdify(self.t,expr) )
		#
		self.set_d_log_R(d_log_R=sympy.diff(R,self.t,1)/R)

	def set_d_log_R(self,d_log_R):
		'''Set time derivative of the logarithm of the tube radius

		This method sets time derivative
		```
		\partial_t \ln( R(t) / L ) \equiv \frac{ \dot{R}(t) }{R(t)}.
		```

		This method is always called when a tube radius is set via self.set_R.

		Args:
			d_log_R (sympy expression): Time derivative of the logarithm of the
			 							tube radius radius as function of self.t
										and in units of self.T (or self.tD)
								  		Example:
									 		d_log_R = 1/self.tD

		Returns:
			None
		'''
		#
		self.d_log_R = d_log_R
		self.d_log_eps = self.tD * d_log_R
		expr = sympy.expand( (self.d_log_eps).subs(self.t,self.t*self.tD) )
		self.d_log_eps_lambda = np.vectorize( sympy.lambdify(self.t,expr ) )


	'''
	ANALYTICAL PART OF MODULE
	''';
	@property
	def analytical(self):
		class SubSpaceClass:
			def get_eigenvalues(self_,powers_separate=False,order=3):
				return self.get_eigenvalues(powers_separate=powers_separate,
											order=order)
			@property
			def eigenvalues(self_): return self.get_eigenvalues()
			#
			def get_eigenfunctions(self_,powers_separate=False,order=3):
				return self.get_eigenfunctions(powers_separate=powers_separate,
											order=order)
			@property
			def eigenfunctions(self_): return self.get_eigenfunctions()
			#
			def get_normalized_density(self_,powers_separate=False,order=3):
				return self.get_normalized_density(powers_separate=powers_separate,
											order=order)
			@property
			def normalized_density(self_): return self.get_normalized_density()
			#
			def get_exit_rate(self_,powers_separate=False,order=3):
				return self.get_exit_rate(powers_separate=powers_separate,
											order=order)
			@property
			def exit_rate(self_): return self.get_exit_rate()
			#
			def eval_exit_rate(self_,t,order=5,
									powers_separate=False,full_output=False):
				return self.eval_exit_rate(t=t,order=order,
										powers_separate=powers_separate,
										full_output=full_output)
			#
			def eval_normalized_density(self_,t=None,Nx=201,
									order=5,xDL=True,
									powers_separate=False,full_output=False):
				return self.eval_normalized_density(t=t,Nx=Nx,
										order=order,xDL=xDL,
										powers_separate=powers_separate,
										full_output=full_output)
		return SubSpaceClass()



	def __substitute_local_variables(self,expression,remote):
		'''Substitute local symbolic variables into imported expressions

		All perturbative analytical results are stored in separate python
		files (see XXX), and are imported in the beginning of this file
		(in lines XXX).
		Upon importing, each analytical expression comes with its own symbolic
		variables for space, time, etc.

		This function takes an expression and substitutes the remote symbolic
		variables with the local symbolic variables used in this class.

		This function is class private as normally there is no reason for the
		user to call it.

		Args:
			expression (sympy expression): sympy expression that uses symbolic
										   variables from the remote namespace
			remote (sympy expression): remote namespace

		Returns:
			sympy expression: the input expression, but with symbolic variables
							  from the self namespace (i.e. from this class)
		'''
		#
		if expression == 0:
			return expression
		#
		try:
			expression = expression.subs(remote.x,self.xDL)
		except AttributeError:
			pass
		#
		try:
			expression = expression.subs(remote.n,self.n)
		except AttributeError:
			pass
		#
		try:
			expression = expression.subs(remote.epsilon,self.epsilon)
		except AttributeError:
			pass
		#
		try:
			expression = expression.subs(remote.d_log_eps,self.d_log_eps)
		except AttributeError:
			pass
		#
		for j in range(self.N_max):
			expression = expression.subs(remote.Dk[j],self.Dk[j])
			expression = expression.subs(remote.Ek[j],self.Ek[j])
		try:
			for j in range(self.N_max):
				expression = expression.subs(remote.d_Dk[j],self.d_Dk[j])
				expression = expression.subs(remote.d_Ek[j],self.d_Ek[j])
		except AttributeError:
			pass
		#
		return expression

	def __pad_array(self,array):
		'''

		to do


		'''
		# The arrays Dk, Ek need to be of length self.N_max. If they are
		# longer, we trim and throw a warning. If they are shorter, we
		# pad with zeros and throw a warning
		if len(array) > self.N_max:
			array_ = array[:self.N_max]
		elif len(array) < self.N_max:
			array_ = np.zeros([self.N_max,*np.shape(array)[1:]],dtype=float)
			array_[:len(array)] = array
		else:
			array_ = array
		return array_

	def __get_power_series(self,dictionary,
								order=3,
								powers_separate=False,
								starting_at_power_negative_two=False):
		'''Get power series from a dictionary of coefficients

		This function takes an dictionary of coefficients and returns the
		corresponding power series to a given order.

		Args:
			dictionary (dict): Dictionary with the coefficints of the power
			   series. This function assumees that the keys in the dictionary
			   are an enumeration, e.g. dictionary[0] = lowest order coefficient
			order (int): Order to which the power series should be returned.
				 This integer is inclusive, i.e. terms up to including order
				 dictionary[order] will be returned.
			powers_separate (bool): If False, then a power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned. Thus, in that case:
					- if order > len(dictionary), the
					  input dictionary is returned
					- if order <= len(dictionary), the
						  truncated input dictionary is returned
			starting_at_power_negative_two (bool): If True, then the lowest
				power is self.epsilon**(-2), with prefactor dictionary[0]
				If False, then the lowest power is self.epsilon**(0), with
				prefactor dictionary[0]

		Returns:
			if powers_separate == True:
				dictionary: dictionary with the power series coefficients up
				 	to the desired order
			else:
				sympy expression: power series in self.epsilon up to the
					desired order

		'''
		if powers_separate:
			output = {}
			for i,item in dictionary.items():
				if i <= order:
					output[i] = item
		else:
			output = 0
			for i,item in dictionary.items():
				if i <= order:
					output += self.epsilon**i * item
			if starting_at_power_negative_two:
				output /= self.epsilon**2
		return output

	def get_eigenvalues(self,powers_separate=False,order=3):
		'''Get power series with perturbative Fokker-Planck eigenvalues

		This function returns the symbolic perturbative expansion
		```
			\lambda_n = \sum_{k=0}^{N} \tilde{\epsilon}^k \lambda_n^{(k)}
		```
		up to a given order N <= 5.

		The eigenvalues correspond to a domain [-1,1] with absorbing boundary
		conditions, the expression is perturbative around the free-diffusion
		eigenvalues (i.e. the case where the deterministic drift in the
		Fokker-Planck equation vanishes).

		The eigenvalues are derived in App. C of Ref. [1]

		Args:
			order (int): Order N up to including which the power series is
				be returned. For this module the perturbative eigenvalue has
				been evaluated to order N = 5; if the order is set to a number
				larger than 5, the perturbative expression to order N = 5 is
				returned.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of sympy expressions containing the power
				 	series coefficients up to the desired order
			else:
				sympy expression: power series in self.epsilon up to the
					desired order
		'''
		return self.__get_power_series(dictionary=self.eigenvalues_dict,
									powers_separate=powers_separate,
									order=order,
									starting_at_power_negative_two=True)

	def get_eigenfunctions(self,powers_separate=False,order=3):
		'''Get power series with perturbative Fokker-Planck eigenfunctions

		This function returns the symbolic perturbative expansion
		```
			\rho_n = \sum_{k=0}^{N} \tilde{\epsilon}^k \rho_n^{(k)}
		```
		up to a given order N <= 5.

		The eigenfunctions correspond to a domain [-1,1] with absorbing boundary
		conditions, the expression is perturbative around the free-diffusion
		eigenvalues (i.e. the case where the deterministic drift in the
		Fokker-Planck equation vanishes).

		The eigenfunctions are derived in App. C of Ref. [1]

		Args:
			order (int): Order N up to including which the power series is
				be returned. For this module the perturbative eigenfunctions
				have been evaluated to order N = 5; if the order is set to a
				number larger than 5, the perturbative expression to order N = 5
				is returned.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of sympy expressions containing the power
				 	series coefficients up to the desired order
			else:
				sympy expression: power series in self.epsilon up to the
					desired order
		'''
		return self.__get_power_series(dictionary=self.eigenfunctions_dict,
									powers_separate=powers_separate,
									order=order)

	def get_exit_rate(self,powers_separate=False,order=3):
		'''Get power-series expansion of the tubular exit rate

		This function returns the symbolic perturbative exit rate
		```
			\alpha = \sum_{k=0}^{N+2} c_k \epsilon^{-2+k}
		```
		up to a given order N <= 3.

		The exit rate is derived in App. C of Ref. [1]

		Args:
			order (int): Order N up to including which the power series is
				be returned. For this module the perturbative exit rate
				has been evaluated to order N = 3; if the order is set to a
				number larger than 3, the exit rate to order N = 3 is returned.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of sympy expressions containing the power
				 	series coefficients up to the desired order
			else:
				sympy expression: power series in self.epsilon up to the
					desired order
		'''
		output = self.__get_power_series(dictionary=self.exit_rate_dict,
									powers_separate=powers_separate,
									order=order,
									starting_at_power_negative_two=True)
		return output

	def get_normalized_density(self,powers_separate=False,order=3):
		'''Get normalized probability density inside the tube

		This function returns the symbolic perturbative probability density
		inside the tube,
		```
			P_{\epsilon}^{n,\varphi}(x,t) = \sum_{k=0}^{N} \epsilon^{k} \
				\times [ N_{s}^{(k)} sin( pi/2 * (x+1) )
					+ N_{c}^{(k)} cos( pi/2 * (x+1) ) ]
		```
		up to a given order N <= 5. Here (x,t) are the dimensionless spatial
		and temporal coordinates inside the tube, as defined in Eq. (A1) of
		Ref. [1].

		Args:
			order (int): Order N up to including which the power series is
				be returned. For this module the perturbative normalized density
				has been evaluated to order N = 5; if the order is set to a
				number larger than 5, the density to order N = 5 is returned.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of sympy expressions containing the power
				 	series coefficients up to the desired order
			else:
				sympy expression: power series in self.epsilon up to the
					desired order
		'''
		return self.__get_power_series(dictionary=self.normalized_density_dict,
									powers_separate=powers_separate,
									order=order)

	def eval_power_series(self,dictionary,
							Dk,Ek,
							epsilon,
							d_Dk=None,
							d_Ek=None,
							order=3,
							powers_separate=False,
							**kwargs):
		'''Evaluate a power series on given tuples D_k, E_k, d_Dk, d_Ek

		Args:
			dictionary (dict): A dictionary of lambda functions such that
				dictionary[i] represents the prefactor for the power \epsilon^i
			Dk (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (D_1, ..., D_{self.N_max}) on which
				the power series is evaluated.
			Ek (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (E_1, ..., E_{self.N_max}) on which
				the power series is evaluated.
			d_Dk (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (dD_1/dt, ..., dD_{self.N_max}/dt)
				on which the power series is evaluated. If d_Dk == None, then
				all dDk/dt = 0.
			d_Ek (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (dE_1/dt, ..., dE_{self.N_max}/dt)
				on which the power series is evaluated. If d_Ek == None, then
				all dEk/dt = 0.
			order (int): Order N up to including which the power series is
				be returned.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.
			**kwargs: keyword arguments for the lambda functions dictionary[i].
				In particular, one keyword argument can be a numpy array of the
				dimensionless spatial coordinate x (e.g. if the dictionary
				represents the power series for the normalized probability
				density inside the tube)

		Note that the N_t in Dk, Ek, d_Dk, d_Ek typically represent the number
		of timesteps in a time series, so that e.g. Dk[k,i] = D_k at timestep i.
		For all Dk, Ek, d_Dk, d_Ek, the N_t need to be the identical.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				 power series coefficients up to the desired order, i.e.
				 dictionary[i] = (power series coefficients at order \epsilon^i)
				 with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of length Nt
		'''
		#
		Ek = self.__pad_array(Ek)
		Dk = self.__pad_array(Dk)
		if d_Dk is not None:
			d_Dk = self.__pad_array(d_Dk)
		if d_Ek is not None:
			d_Ek = self.__pad_array(d_Ek)
		#
		# evaluate items
		powers_evaluated = {}
		for i,item in dictionary.items():
			if i <= order:
				if (d_Dk is not None) or (d_Ek is not None):
					powers_evaluated[i] = item(Dk,d_Dk,Ek,d_Ek,**kwargs)
				else:
					powers_evaluated[i] = item(Dk,Ek,**kwargs)
		#
		if powers_separate:
			return powers_evaluated
		else:
			have_output_array = False
			for i,item in powers_evaluated.items():
				if not have_output_array:
					output = epsilon**i * item
					have_output_array = True
				else:
					output = output + epsilon**i * item
			return output

	def __preprocess_input_for_eval(self,Dk,Ek,
									x = None,
									d_Dk=None,
									d_Ek=None,
									epsilon=1.,
									d_log_eps=None):
		'''Preprocess input numpy arrays for evaluation of power series

		This function serves two purposes:
		- If only the arrays for D_k, E_k have been provided, this function adds
		  vanishing temporal derivatives dD_k/dt, dE_k/dt.
		- If arrays for D_k, E_k (and possibly dD_k/dt, dE_k/dt) as well as an
		  array x with dimensionless spatial coordinates inside the tube have
		  been provided, this function ensures that the lambda functions for
		  power series expansions can be callled with the arrays
		  (D_k, E_k, d_Dk, d_Ek, x, epsilon, d_log_eps) in a vectorized fashion.

		Args:
			Dk (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (D_1, ..., D_{self.N_max}) on which
				the power series is evaluated.
			Ek (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (E_1, ..., E_{self.N_max}) on which
				the power series is evaluated.
			d_Dk (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (dD_1/dt, ..., dD_{self.N_max}/dt)
				on which the power series is evaluated. If d_Dk == None, then
				an array is created such that dD_k/dt = 0.
			d_Ek (np.array): A numpy array of size (self.N_max, N_t), where
				N_t is the number of tuples (dE_1/dt, ..., dE_{self.N_max}/dt)
				on which the power series is evaluated. If d_Ek == None, then
				an array is created such that dE_k/dt = 0.
			x (np.array): A numpy array containing dimensionless spatial
				coordinate values, i.e. floats in the domain [-1,1].
			epsilon (float or np.array): Dimensionless tube radius
			d_log_eps (float or np.array): Dimensionless temporal derivative of
				log(epsilon)
		Returns:
			dictionary with the preprocessed floats/arrays for epsilon,
			d_log_eps, Dk, Ek, d_Dk, d_Ek, x
		'''
		#
		if d_Dk is None:
			d_Dk = np.zeros_like(Dk)
		#
		if d_Ek is None:
			d_Ek = np.zeros_like(Ek)
		#
		if not np.isscalar(Dk[0]):
			if x is not None:
				x = x[np.newaxis,:]
				Dk = Dk[:,:,np.newaxis]
				d_Dk = d_Dk[:,:,np.newaxis]
				Ek = Ek[:,:,np.newaxis]
				d_Ek = d_Ek[:,:,np.newaxis]
				if not np.isscalar(epsilon):
					epsilon = epsilon[:,np.newaxis]
					if d_log_eps is not None:
						if not np.isscalar(d_log_eps):
							d_log_eps = d_log_eps[:,np.newaxis]

		output_dictionary = {'epsilon':epsilon,
						'Dk':Dk,'Ek':Ek,
						'd_Dk':d_Dk,'d_Ek':d_Ek,
						'x':x,
						'd_log_eps':d_log_eps}
		return output_dictionary

	def eval_eigenvalues_DE(self,Dk,Ek,
							n=1,
							epsilon=1.,
							order=3,
							powers_separate=False,
							):
		'''Evaluate perturbative eigenvalue on given D_k, E_k, n, epsilon

		Args:
			Dk (numpy.array): Array of size (self.N_max, N_t), where N_t is the
			 	number of tuples (D_1, ..., D_{self.N_max}) on which the power
				series is evaluated.
			Ek (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (E_1, ..., E_{self.N_max}) on which the power
				series is evaluated.
			n (int): Integer that specifies which eigenvalue is evaluated. For
				example, n = 1 represents the eigenvalue with the smallest
				absolute value. Default value is n = 1.
			epsilon (float or numpy.array): dimensionless tube radius. A float
				represent a time-independent tube radius. If an array is given,
				it needs to have the same length as the Dk[i], Ek[i].
			order (int): Order N up to including which the perturbative
				eigenvalue is returned. Default value is N = 3.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Note that the N_t in Dk, Ek, epsilon typically represent the number
		of timesteps in a time series, so that e.g. Dk[k,i] = D_k at timestep i.
		For all Dk, Ek, epsilon the N_t need to be the identical.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order
				\epsilon^{i-2})
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of length Nt
		'''
		result = self.eval_power_series(
							dictionary=eigenvalues.eigenvalues_lambda_dict,
							Dk=Dk,
							Ek=Ek,
							powers_separate=powers_separate,
							order=order,
							n=n,
							epsilon=epsilon,
							)
		if powers_separate:
			return result
		else:
			return result/epsilon**2
		return self.eval_power_series(
							dictionary=eigenvalues.eigenvalues_lambda_dict,
							Dk=Dk,
							Ek=Ek,
							powers_separate=powers_separate,
							order=order,
							n=n,
							epsilon=epsilon,
							)

	def eval_eigenfunctions_DE(self,Dk,Ek,
							epsilon=1.,
							x=np.linspace(-1,1,num=201,endpoint=True),
							n=1,
							powers_separate=False,
							order=3):
		'''Evaluate perturbative eigenfunctions on given D_k, E_k, n, epsilon, x

		Args:
			Dk (numpy.array): Array of size (self.N_max, N_t), where N_t is the
			 	number of tuples (D_1, ..., D_{self.N_max}) on which the power
				series is evaluated.
			Ek (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (E_1, ..., E_{self.N_max}) on which the power
				series is evaluated.
			n (int): Integer that specifies which eigenvalue is evaluated. For
				example, n = 1 represents the eigenvalue with the smallest
				absolute value. Default value is n = 1.
			epsilon (float or numpy.array): dimensionless tube radius. A float
				represent a time-independent tube radius. If an array is given,
				it needs to have the same length as the Dk[i], Ek[i].
			x (numpy.array, optional): Array with dimensionless positions in
				[-1,1]. Defaults to x= numpy.linspace(-1,1,num=201,endpoint=True)
			order (int): Order N up to including which the perturbative
				eigenvalue is returned. Default value is N = 3.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Note that the N_t in Dk, Ek, epsilon typically represent the number
		of timesteps in a time series, so that e.g. Dk[k,i] = D_k at timestep i.
		For all Dk, Ek, epsilon the N_t need to be the identical.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order
				\epsilon^{i-2})
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 2D numpy array of size ( Nt, len(x) )
		'''
		#
		preprocessed_input = self.__preprocess_input_for_eval(Dk=Dk,Ek=Ek,
												epsilon=epsilon,x=x)
		#
		return self.eval_power_series(
							dictionary=eigenfunctions.eigenfunctions_lambda_dict,
							Dk=preprocessed_input['Dk'],
							Ek=preprocessed_input['Ek'],
							powers_separate=powers_separate,
							order=order,
							n=n,
							x=preprocessed_input['x'],
							epsilon=preprocessed_input['epsilon'],
							)

	def eval_normalized_density_DE(self,Dk,
							Ek,
							d_Dk=None,
							d_Ek=None,
							epsilon=1.,
							xDL=np.linspace(-1,1,num=201,endpoint=True),
							powers_separate=False,
							order=3):
		'''Evaluate perturbative normalized density on given D_k, E_k, etc.

		Args:
			Dk (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (D_1, ..., D_{self.N_max}) on which the power
				series is evaluated.
			Ek (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (E_1, ..., E_{self.N_max}) on which the power
				series is evaluated.
			n (int): Integer that specifies which eigenvalue is evaluated. For
				example, n = 1 represents the eigenvalue with the smallest
				absolute value. Default value is n = 1.
			epsilon (float or numpy.array): dimensionless tube radius. A float
				represent a time-independent tube radius. If an array is given,
				it needs to have the same length as the Dk[i], Ek[i].
			xDL (numpy.array, optional): Array with dimensionless positions in
				[-1,1]. Defaults to xDL = numpy.linspace(-1,1,num=201,endpoint=True)
			order (int): Order N up to including which the perturbative
				eigenvalue is returned. Default value is N = 3.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Note that the N_t in Dk, Ek, epsilon typically represent the number
		of timesteps in a time series, so that e.g. Dk[k,i] = D_k at timestep i.
		For all Dk, Ek, epsilon the N_t need to be the identical.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order
				\epsilon^{i-2})
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 2D numpy array of size ( Nt, len(xDL) )
		'''
		#
		preprocessed_input = self.__preprocess_input_for_eval(Dk=Dk,Ek=Ek,
												d_Dk=d_Dk,d_Ek=d_Ek,
												epsilon=epsilon,x=xDL)
		#
		return self.eval_power_series(
							dictionary=normalized_density.normalized_density_lambda_dict,
							Dk=preprocessed_input['Dk'],
							d_Dk=preprocessed_input['d_Dk'],
							Ek=preprocessed_input['Ek'],
							d_Ek=preprocessed_input['d_Ek'],
							powers_separate=powers_separate,
							order=order,
							x=preprocessed_input['x'],
							epsilon=preprocessed_input['epsilon'],
							)

	def eval_exit_rate_DE(self,Dk,Ek,
							d_Dk=None,
							d_Ek=None,
							epsilon=1.,
							d_log_eps=0.,
							order=3,
							powers_separate=False
							):
		'''Evaluate perturbative exit rate on given D_k, E_k, etc.

		Args:
			Dk (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (D_1, ..., D_{self.N_max}) on which the exit
				series is evaluated.
			Ek (numpy.array): Array of size (self.N_max, N_t), where N_t is the
				number of tuples (E_1, ..., E_{self.N_max}) on which the exit
				series is evaluated.
			d_Dk (numpy.array, optional): Array of size (self.N_max, N_t), where
			 	N_t is the number of tuples (dD_1/dt, ..., dD_{self.N_max}/dt)
				on which the exit rate is evaluated. Defaults to (0, ..., 0)
			d_Ek (numpy.array, optional): Array of size (self.N_max, N_t), where
				N_t is the number of tuples (dE_1/dt, ..., dE_{self.N_max}/dt)
				on which the exit rate is evaluated. Defaults to (0, ..., 0)
			epsilon (float or numpy.array, optional): dimensionless tube radius.
				A float represent a time-independent tube radius. If an array is
				given, it needs to have the same length as the Dk[i], Ek[i].
				Defaults to 1.
			d_log_eps (float or numpy.array, optional): dimensionless time
				derivative of ln(epsilon). Defaults to 0.
			order (int): Order N up to including which the perturbative
				eigenvalue is returned. Default value is N = 3.
			powers_separate (bool): If False, then the power series in
				self.epsilon is returned.
				If True, then a dictionary with the power series coefficients
				up to the desired order are returned.

		Note that the N_t in Dk, Ek, d_Dk, d_Ek, epsilon, d_log_eps typically
		represent the number of timesteps in a time series, so that e.g.
		Dk[k,i] = D_k at timestep i. For all Dk, Ek, epsilon the N_t need to be
		identical.

		Returns:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order
				\epsilon^{i-2})
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 2D numpy array of size ( Nt, len(x) )
		'''
		#
		preprocessed_input = self.__preprocess_input_for_eval(Dk=Dk,Ek=Ek,
												d_Dk=d_Dk,d_Ek=d_Ek,
												epsilon=epsilon,
												d_log_eps=d_log_eps)
		#
		output = self.eval_power_series(
							dictionary=exit_rate.exit_rate_lambda_dict,
							Dk=preprocessed_input['Dk'],
							Ek=preprocessed_input['Ek'],
							d_Dk=preprocessed_input['d_Dk'],
							d_Ek=preprocessed_input['d_Ek'],
							powers_separate=powers_separate,
							order=order,
							epsilon=preprocessed_input['epsilon'],
							d_log_eps=preprocessed_input['d_log_eps'],
							)
		#
		if not powers_separate:
			output /= epsilon**2
		return output

	def construct_D_functions(self):
		'''Construct lambda functions for D_k from symbolic diffusivity profile

		This method defines lists of lambda functions self.Dk_symbolic,
		self.d_Dk_symbolic, such that
		self.Dk_symbolic[i] = D_k(x=path(t),t)
		self.d_Dk_symbolic[i] = dD_k(x=path(t),t)/dt

		Args: None
		Returns: None
		'''
		self.Dk_symbolic = []
		self.d_Dk_symbolic = []
		#
		for i in range(self.N_max):
			if i == 0:
				current_derivative = self.D
			else:
				current_derivative = sympy.diff(current_derivative,
												self.x,1)
			#
			self.Dk_symbolic.append( \
							  self.L**i / self.D0 \
							* sympy.Rational(1,self.factorials[i]) \
							* sympy.expand(current_derivative.subs(self.x,self.path)) \
										)
			self.d_Dk_symbolic.append( \
						self.tD * sympy.diff(self.Dk_symbolic[-1], self.t, 1 ) \
							)
		#print(self.Dk_symbolic)
		#print(self.d_Dk_symbolic)
		self.Dk_lambda = []
		for i, current_function in enumerate(self.Dk_symbolic):
			self.Dk_lambda.append(
				sympy.lambdify( self.t,
								current_function.subs(self.t,self.t*self.T) )
									)
		self.d_Dk_lambda = []
		for i, current_function in enumerate(self.d_Dk_symbolic):
			self.d_Dk_lambda.append(
				sympy.lambdify( self.t,
								current_function.subs(self.t,self.t*self.T) )
									)
		self.D_functions_constructed = True

	def construct_E_functions(self):
		'''Construct lambda functions for E_k from symbolic drift profile

		This method defines lists of lambda functions self.Ek_symbolic,
		self.d_Ek_symbolic, such that
		self.Ek_symbolic[i] = E_k(x=path(t),t)
		self.d_Ek_symbolic[i] = dE_k(x=path(t),t)/dt

		Args: None
		Returns: None
		'''
		self.Ek_symbolic = []
		self.d_Ek_symbolic = []
		#
		for i in range(self.N_max):
			if i == 0:
				self.Ek_symbolic.append( sympy.Float( 0 ) )
				self.d_Ek_symbolic.append( sympy.Float( 0 ) )
				continue
			#
			if i == 1:
				current_derivative = self.a
			else:
				current_derivative = sympy.diff(current_derivative,
												self.x,1)
			#
			self.Ek_symbolic.append( \
							  -self.L**i / self.D0 \
							* sympy.Rational(1,self.factorials[i]) \
							* current_derivative.subs(self.x,self.path) \
										)
			if i == 1:
				self.Ek_symbolic[-1] += self.tD/self.L * self.dot_path
			elif i == 2:
				self.Ek_symbolic[-1] += sympy.Rational(1,2)*self.tD \
															* self.d_log_R
			#
			self.d_Ek_symbolic.append( \
						self.tD * sympy.diff(self.Ek_symbolic[-1], self.t, 1 ) \
							)
		#print(self.Ek_symbolic)
		#print(self.d_Ek_symbolic)
		self.Ek_lambda = []
		for i, current_function in enumerate(self.Ek_symbolic):
			self.Ek_lambda.append(
				sympy.lambdify( self.t,
								current_function.subs(self.t,self.t*self.T) )
									)
		self.d_Ek_lambda = []
		for i, current_function in enumerate(self.d_Ek_symbolic):
			self.d_Ek_lambda.append(
				sympy.lambdify( self.t,
								current_function.subs(self.t,self.t*self.T) )
									)
		self.E_functions_constructed = True

	def __eval_Dk(self,t):
		'''Evaluates lambda functions for D_k for given array with times t

		This method defines numpy.arrays self.Dk_evaluated, self.d_Dk_evaluated
		of shape (self.N_Max, Nt), where Nt = len(t), such that
		self.Dk_evalauted[i,j] = D_i(t=t[j])
		self.d_Dk_evalauted[i,j] = dD_i/dt|_{t=t[j]}

		Args:
			t (numpy.array): Array with time values
		Returns: None
		'''
		# t = array of times
		#
		Nt = len(t)
		#
		self.Dk_evaluated = np.zeros([self.N_max,Nt],dtype=float)
		self.d_Dk_evaluated = np.zeros([self.N_max,Nt],dtype=float)
		#
		for i,current_function in enumerate(self.Dk_lambda):
			self.Dk_evaluated[i] = current_function(t)
		for i,current_function in enumerate(self.d_Dk_lambda):
			self.d_Dk_evaluated[i] = current_function(t)

	def __eval_Ek(self,t):
		'''Evaluates lambda functions for E_k for given array with times t

		This method defines numpy.arrays self.Ek_evaluated, self.d_Ek_evaluated
		of shape (self.N_Max, Nt), where Nt = len(t), such that
		self.Ek_evalauted[i,j] = E_i(t=t[j])
		self.d_Ek_evalauted[i,j] = dE_i/dt|_{t=t[j]}

		Args:
			t (numpy.array): Array with time values
		Returns: None
		'''
		# t = array of times
		#
		Nt = len(t)
		#
		self.Ek_evaluated = np.zeros([self.N_max,Nt],dtype=float)
		self.d_Ek_evaluated = np.zeros([self.N_max,Nt],dtype=float)
		#
		for i,current_function in enumerate(self.Ek_lambda):
			self.Ek_evaluated[i] = current_function(t)
		for i,current_function in enumerate(self.d_Ek_lambda):
			self.d_Ek_evaluated[i] = current_function(t)


	def __preprocess_for_eval_of_power_series(self,t=None):
		'''Preprocess lambda functions for E_k for given array with times t

		This method defines numpy.arrays self.Ek_evaluated, self.d_Ek_evaluated
		of shape (self.N_Max, Nt), where Nt = len(t), such that
		self.Ek_evalauted[i,j] = E_i(t=t[j])
		self.d_Ek_evalauted[i,j] = dE_i/dt|_{t=t[j]}

		Args:
			t (numpy.array): Array with time values
		Returns: None
		'''
		#
		if t is not None:
			if np.isscalar(t):
				t = np.array([t])
		else:
			t = np.linspace(np.float(self.ti/self.tD),
							np.float(self.tf/self.tD),
							num=self.Nt,
							endpoint=True)
		#
		self.epsilon_evaluated = self.epsilon_lambda(t)
		self.d_epsilon_evaluated = self.d_log_eps_lambda(t)
		#
		if not self.D_functions_constructed:
			self.construct_D_functions()
		if not self.E_functions_constructed:
			self.construct_E_functions()
		#
		self.__eval_Ek(t)
		self.__eval_Dk(t)
		#
		return t


	def eval_exit_rate(self,t,
							order=5,
							powers_separate=False,
							full_output=True):
		'''Evaluates perturbative exit rate for given array of times t

		Args:
			t (numpy.array): Array of times.
			order (int, optional): Order N up to including which the perturbative
				eigenvalue is returned. Defaults to N = 5.
			powers_separate (bool, optional): If False, then the power series in
				self.epsilon is returned. If True, then a dictionary with the
				power series coefficients up to the desired order are returned.
				Defaults to True
			full_output (bool, optional): If True, a dictionary with times,
				epsilon, exit_rate is return. If False, only exit_rate is
				returned. Defaults to True.

		Returns:
			if full_output == True:
				dictionary with numpy.arrays with times, epsilon, and exit_rate
			else:
				perturbative exit_rate

			Note that the type of exit_rate depends on powers_separate:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order
				\epsilon^{i-2})
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of size Nt
		'''
		#
		t = self.__preprocess_for_eval_of_power_series(t=t)
		#
		exit_rate = self.eval_exit_rate_DE(Dk=self.Dk_evaluated,
								Ek=self.Ek_evaluated,
								d_Dk=self.d_Dk_evaluated,
								d_Ek=self.d_Ek_evaluated,
								epsilon=self.epsilon_evaluated,
								d_log_eps=self.d_epsilon_evaluated,
								powers_separate=powers_separate,
								order=order)
		#
		if full_output:
			output_dictionary = {'t':t,
								'epsilon':self.epsilon_evaluated,
								'exit_rate':exit_rate}
			return output_dictionary
		else:
			return exit_rate

	def eval_eigenvalues(self,t=None,
							n=1,
							order=5,
							powers_separate=False,
							full_output=True):
		'''Evaluates perturbative eigenvalues for given array of times t

		Args:
			t (numpy.array): Array of times.
			n (int, optional): Integer that specifies which eigenvalue is
				evaluated. For example, n = 1 represents the eigenvalue with the
				smallest absolute value. Defaults to 1.
			order (int, optional): Order N up to including which the perturbative
				eigenvalue is returned. Defaults to N = 5.
			powers_separate (bool, optional): If False, then the power series in
				self.epsilon is returned. If True, then a dictionary with the
				power series coefficients up to the desired order are returned.
				Defaults to True
			full_output (bool, optional): If True, a dictionary with times,
				epsilon, exit_rate is return. If False, only exit_rate is
				returned. Defaults to True.

		Returns:
			if full_output == True:
				dictionary with times, epsilon, and perturbative eigenvalues
			else:
				perturbative eigenvalues

			Note that the type of exit_rate depends on powers_separate:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order \epsilon^i)
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of size Nt
		'''
		#
		t = self.__preprocess_for_eval_of_power_series(t=t)
		#
		eigenvalues = self.eval_eigenvalues_DE(Dk=self.Dk_evaluated,
								Ek=self.Ek_evaluated,
								epsilon=self.epsilon_evaluated,
								powers_separate=powers_separate,
								order=order,
								n=n)
		#
		if full_output:
			output_dictionary = {'t':t,
								'epsilon':self.epsilon_evaluated,
								'eigenvalues':eigenvalues}
			return output_dictionary
		else:
			return eigenvalues

	def eval_eigenfunctions(self,t,
							n=1,
							Nx=None,
							order=5,
							xDL=True,
							powers_separate=False,
							full_output=True):
		'''Evaluates perturbative eigenfunctions for given array of times t

		Args:
			t (numpy.array): Array of times.
			n (int, optional): Integer that specifies which eigenvalue is
				evaluated. For example, n = 1 represents the eigenvalue with the
				smallest absolute value. Defaults to 1.
			Nx (int, optional): number of spatial datapoints for evaluation of
				the eigenfunction. Defaults to self.Nx, which defaults to 101.
			order (int, optional): Order N up to including which the perturbative
				eigenvalue is returned. Defaults to N = 5.
			xDL (bool, optional): If True, then spatial datapoints are returned
				in dimensionless units. If False, then spatial datapoints are
				returned in physical units. Defaults to True.
			powers_separate (bool, optional): If False, then the power series in
				self.epsilon is returned. If True, then a dictionary with the
				power series coefficients up to the desired order are returned.
				Defaults to True
			full_output (bool, optional): If True, a dictionary with times,
				epsilon, exit_rate is return. If False, only exit_rate is
				returned. Defaults to True.

		Returns:
			if full_output == True:
				dictionary with t, epsilon, x, path, perturbative eigenfunctions
			else:
				perturbative eigenfunctions

			The type of the eigenfunctions depends on powers_separate:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order \epsilon^i)
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of size Nt
		'''
		#
		t = self.__preprocess_for_eval_of_power_series(t=t)
		#
		if Nx is None:
			Nx = self.Nx
		#
		xDL_ = np.linspace(-1,1,num=Nx,endpoint=True)
		#
		eigenfunctions_evaluated = self.eval_eigenfunctions_DE(Dk=self.Dk_evaluated,
								Ek=self.Ek_evaluated,
								x=xDL_,
								epsilon=self.epsilon_evaluated,
								powers_separate=powers_separate,
								order=order,
								n=n)
		#
		path_eval = self.path_lambda(t)
		if not xDL:
			x_out = xDL_*epsilon_evaluated + path_eval
		else:
			x_out = xDL_
		#
		if full_output:
			output_dictionary = {'t':t,
								'x':x_out,
								'epsilon':self.epsilon_evaluated,
								'path_eval':path_eval,
								'eigenfunctions':eigenfunctions_evaluated}
			return output_dictionary
		else:
			return eigenfunctions_evaluated

	def eval_normalized_density(self,t=None,
							Nx=None,
							order=5,
							xDL=True,
							powers_separate=False,
							full_output=True):
		'''Evaluates perturbative normalized density for given array of times t

		Args:
			t (numpy.array): Array of times.
			Nx (int, optional): number of spatial datapoints for evaluation of
				the eigenfunction. Defaults to self.Nx, which defaults to 101.
			order (int, optional): Order N up to including which the perturbative
				eigenvalue is returned. Defaults to N = 5.
			xDL (bool, optional): If True, then spatial datapoints are returned
				in dimensionless units. If False, then spatial datapoints are
				returned in physical units. Defaults to True.
			powers_separate (bool, optional): If False, then the power series in
				self.epsilon is returned. If True, then a dictionary with the
				power series coefficients up to the desired order are returned.
				Defaults to True
			full_output (bool, optional): If True, a dictionary with times,
				epsilon, exit_rate is return. If False, only exit_rate is
				returned. Defaults to True.

		Returns:
			if full_output == True:
				dictionary with t, epsilon, x, path, normalized density
			else:
				normalized density

			The type of the normalized density depends on powers_separate:
			if powers_separate == True:
				dictionary: dictionary of numpy arrays containing the evaluated
				power series coefficients up to the desired order, i.e.
				dictionary[i] = (power series coefficients at order \epsilon^i)
				with len(dictionary[i]) = Nt.
			else:
				numpy array: power series in self.epsilon up to the desired
				order, i.e. 1D numpy array of size Nt
		'''
		#
		t = self.__preprocess_for_eval_of_power_series(t=t)
		#
		if Nx is None:
			Nx = self.Nx
		#
		xDL_ = np.linspace(-1,1,num=Nx,endpoint=True)
		#
		normalized_density = self.eval_normalized_density_DE(Dk=self.Dk_evaluated,
								Ek=self.Ek_evaluated,
								d_Dk=self.d_Dk_evaluated,
								d_Ek=self.d_Ek_evaluated,
								xDL=xDL_,
								epsilon=self.epsilon_evaluated,
								powers_separate=powers_separate,
								order=order)
		#
		path_eval = self.path_lambda(t)
		if not xDL:
			x_out = xDL_*epsilon_evaluated + path_eval
		else:
			x_out = xDL_
		#
		if full_output:
			output_dictionary = {'t':t,
								'x':x_out,
								'epsilon':self.epsilon_evaluated,
								'path_eval':path_eval,
								'normalized_density':normalized_density}
			return output_dictionary
		else:
			return normalized_density



	'''
	NUMERICAL SIMULATIONS
	''';
	@property
	def numerical(self):
		class SubSpaceClass:
			def simulate(self_,P0=None):
				return self.simulate(P0=P0)
			@property
			def run(self_): return self.simulate()
			#
			def get_results(self_):
				return self.simulation_return_results()
			@property
			def results(self_): return self.simulation_return_results()
			#
		return SubSpaceClass()


	def __simulation_print_time_remaining(self,step,end='\r'):
		'''Print remaining time for running numerical simulation

		Args:
			step (int): current step of simulation
			end (str, optional): end of print function. Defaults to '\r'.

		Return:
			None
		'''
		elapsed_time = time.time() - self.system_time_at_start_of_simulation
		m_elapsed, s_elapsed = divmod(elapsed_time, 60)
		h_elapsed, m_elapsed = divmod(m_elapsed, 60)
		remaining_time = (self.Nt/step -1)*elapsed_time
		m_remaining, s_remaining = divmod(remaining_time, 60)
		h_remaining, m_remaining = divmod(m_remaining, 60)
		str_0 = 'Running simulation. Progress: '
		str_1 = ', elapsed time: '
		str_2 = ', remaining time: '
		str_3 = '\t\t\t'
		print("{7}{0}%{8}{1:d}:{2:02d}:{3:02d}{9}{4:d}:{5:02d}:{6:02d}{10}".format(
				int(step/self.Nt*100.),int(np.round(h_elapsed)),
				int(np.round(m_elapsed)),int(np.round(s_elapsed)),
				int(np.round(h_remaining)),int(np.round(m_remaining)),
				int(np.round(s_remaining)),str_0,str_1,str_2,str_3),
			  end=end)

	def simulation_return_results(self):
		'''Return results of numerical simulation

		This function returns a dictionary containing the results of a numerical
		simulation of the Fokker-Planck equation.

		Args:
			None

		Return:
			dictionary: Contains results from numerical simulation
		'''
		output_dictionary = {'t':self.tDL_array[::self.saving_stride],
							'x':self.xDL_array,
							'P':self.PDL_array,
							'dx':self.dxDL,'Nx':self.Nx,
							'dt':self.dtDL,'Nt':self.Nt,
							'saving_stride':self.saving_stride,
							}
		exit_rate_results = self.simulation_get_exit_rate(result=output_dictionary)
		#
		for key, value in exit_rate_results.items():
			output_dictionary[key] = value
		return output_dictionary

	def simulation_create_dimensionless_P0(self):
		'''Reformulate input probability density in dimensionless variables

		This method takes a given initial distribution and creates a lambda
		function that represents the corresponding dimensionless initial
		distribution inside the tube at the initial time. The lambda function
		is stored in self.P0_lambda.

		Args:
			None

		Return:
			None
		'''
		#
		expr = self.P0 / self.R
		#
		expr = expr.subs(self.x,self.path + self.R*self.x)
		expr = expr.subs(self.t,self.ti)
		expr = sympy.expand(expr)
		#
		self.P0_lambda = np.vectorize( sympy.lambdify(self.x,expr) )

	def simulation_determine_timestep(self,Nt_probe=1001):
		'''Estimate proper timestep for numerical simulation

		This method uses the von Neumann stability criterion
		```
		dt < 0.5 * dx**2 / D
		```
		to determine an appropriate timestep for numerical simulations (see
		https://en.wikipedia.org/wiki/FTCS_scheme#Stability).
		For this, the method evaluates D(path(t)) on an array of times in
		[t_i,t_f] and sets
		```
		dt = 0.25 * dx**2 / max_t\{ D( path(t) ) \}.
		```
		The number of simulation timesteps corresponding to this timestep
		is stored in self.Nt

		Args:
			Nt_probe (int, optional): Number of times in interval [t_i,t_f]
				for evaluation of D(path(t)). Defaults to 1001

		Return:
			None
		'''
		#
		t = np.linspace(np.float(self.ti/self.tD),
						np.float(self.tf/self.tD),
						num=Nt_probe,
						endpoint=True)
		#
		D_eval = self.D_lambda(self.path_lambda(t),t)
		epsilon_eval = self.epsilon_lambda(t)
		D_max = np.max(D_eval/epsilon_eval**2)
		#
		dt = 0.25 * self.dxDL**2 / D_max
		#
		self.Nt = int(np.ceil( np.float((self.tf-self.ti)/self.tD) / dt ))

	def simulation_get_exit_rate(self,result):
		'''Evaluate exit rate from numerical solution of Fokker-Planck equation

		This function calculates the instantaneous exit rate from a numerical
		solution of the Fokker-Planck equation, and returns the results as a
		dictionary.

		Args:
			result (dictionary): Dictionary with simulation results. Needs to
				contain keys 't', 'x', 'P' corresponding to times, positions,
				and absorbing-boundary solution at all (time, position) tuples

		Return:
			dictionary: Contains time 't' (identical to 't' from input
				dictionary), 'integrals' with sojourn probability as a function
				of time, and 'exit_rate' as a function of time.
		'''
		#
		x = result['x']
		y = result['P']
		t = result['t']
		dt = t[1] - t[0]
		#
		integrals = np.trapz(y,x,axis=1)
		exit_rate = np.zeros_like(t,dtype=float)
		#
		exit_rate[0] = (integrals[1] - integrals[0])/(dt)
		exit_rate[0] /= integrals[0]
		#
		exit_rate[1:-1] = (integrals[2:] - integrals[:-2])/(2*dt)
		exit_rate[1:-1] /= integrals[1:-1]
		#
		exit_rate[-1] = (integrals[-1] - integrals[-2])/(dt)
		exit_rate[-1] /= integrals[-1]
		#
		exit_rate *= -1
		#
		output_dictionary = {'integrals':integrals,
							'exit_rate':exit_rate,
							't':t}
		#
		return output_dictionary

	def simulate(self,P0=None):
		'''Solve absorbing-boundary Fokker-Planck equation numerically

		This function solves the absorbing-boundary Fokker-Planck equation
		numerically, for a time-dependent domain [path(t) - R(t),path(t) + R(t)]
		and a given initial distribution inside the domain.
		The results of the simulation are resulted as a dictionary.

		Args:
			P0 (numpy array, optional): Initial distribution inside the tube.
				Must be of either length self.Nx (not containing boundary points)
				or self.Nx + 2 (containing boundary points, but those will be
				discarded).
				If no P0 is given, then the symbolic self.P0 is used as initial
				condition. This self.P0 is set via the method set_P0.
				If no P0 is given and no symbolic self.P0 is defined, the
				normalized free-diffusion steady-state is used as initial
				condition.
		Return:
			dictionary: Contains simulation results including exit rate.
		'''
		#
		# check if all parameters are set
		#if self.set_Nx == False:
		#	raise RuntimeError("Parameter Nx not set")
		#if self.set_dt == False:
		#	raise RuntimeError("Parameter dt not set")
		if self.Nt_set == False:
			self.simulation_determine_timestep()
		#
		self.PDL_array = np.zeros([self.Nt//self.saving_stride + 1,self.Nx+2],
						dtype=float)
		self.tDL_array = np.linspace(np.float(self.ti/self.tD),
								np.float(self.tf/self.tD),
								num=self.Nt+1,
								endpoint=True)
		self.dtDL = self.tDL_array[1] - self.tDL_array[0]
		#
		#if self.verbose:
		#	print('Timestep dt =',self.dtDL,'T')
		#
		if P0 is not None:
			if len(P0) == self.Nx:
				self.PDL_array[0,1:-1] = P0.copy()
			elif len(P0) == (self.Nx+2):
				self.PDL_array[0] = P0.copy()
				if self.verbose:
					print("Provided boundary values from initial condition will "\
						+ "be discarded.")
			else:
				raise RuntimeError("Invalid initial condition. Initial condition " \
						+ "must be array of length Nx or Nx+2.")
		else:
			if self.P0_set:
				self.simulation_create_dimensionless_P0()
				self.PDL_array[0] = self.P0_lambda(self.xDL_array)
			else:
				self.PDL_array[0] = np.pi/4 * np.cos(np.pi*self.xDL_array/2.)
		#
		self.PDL_array[0,0] = 0.
		self.PDL_array[0,-1] = 0.
		#
		self.system_time_at_start_of_simulation = time.time()
		#
		current_P = self.PDL_array[0].copy()
		for step,current_time in enumerate(self.tDL_array[:-1]):
			if self.verbose:
				if ( step % int(self.Nt/100) == 0 ) and step > 0:
					self.__simulation_print_time_remaining(step)
			#
			epsilon_eval = self.epsilon_lambda(current_time)
			epsilon_inv_eval = 1/epsilon_eval
			epsilon_inv_sq_eval = 1/epsilon_eval**2
			d_epsilon_eval = self.d_epsilon_lambda(current_time)
			path_eval = self.path_lambda(current_time)
			dot_path_eval = self.dot_path_lambda(current_time)
			#d_log_eps_eval = d_epsilon_eval*epsilon_inv_eval
			#
			current_D_array = self.D_lambda(
							path_eval + epsilon_eval*self.xDL_array,
							current_time)
			current_a_array = self.a_lambda(
							path_eval + epsilon_eval*self.xDL_array,
							current_time
											) \
								- dot_path_eval \
								- d_epsilon_eval * self.xDL_array
			#
			current_D_array = current_D_array.astype(float)
			current_a_array = current_a_array.astype(float)
			#
			# calculate values of interior points in next timestep
			next_P = current_P.copy()
			next_P[1:-1] +=  self.dtDL * self.one_by_dxDL_squared \
				* ( -self.dxDL * ( current_a_array[2:]*current_P[2:] - \
								 current_a_array[:-2]*current_P[:-2] \
								 ) * epsilon_inv_eval /2.  \
					+ ( current_D_array[2:] * current_P[2:] \
						- 2 * current_D_array[1:-1] * current_P[1:-1] \
						+ current_D_array[:-2]*current_P[:-2] \
						) * epsilon_inv_sq_eval
				)
			#
			# update boundary points
			next_P[0] = 0.
			next_P[-1] = 0.
			#
			# save to output array
			if (step+1)%self.saving_stride == 0:
				self.PDL_array[(step+1)//self.saving_stride] = next_P.copy()
			#
			current_P = next_P
		#
		if self.verbose:
			self.__simulation_print_time_remaining(step+1,end='\n')
		#
		return self.simulation_return_results()
