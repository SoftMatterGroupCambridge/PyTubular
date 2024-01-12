#!/usr/bin/env python

import numpy as np

class numerical:
	'''
	This class contains a numpy implementation of the perturbative 
	exit rate
	'''
	def __init__(self,parameters=None):
		#
		#
		self.dot_path_function = None
		self.d_log_R_function = None

		self.factorials = [1]
		for i in range(6):
			self.factorials.append(self.factorials[-1]*(i+1))
		self.pi_squared = np.pi**2

		self.set_parameters(parameters=parameters)


	def set_parameters(self,parameters=None):
		#
		if parameters is None:
			return 
		#
		#
		try:
			self.a_functions = parameters['a']
		except KeyError:
			pass
		#
		try:
			self.D_functions = parameters['D']
		except KeyError:
			pass
		#
		try:
			self.R_function = parameters['R']
		except KeyError:
			pass
		#
		try:
			self.d_log_R_function = parameters['d_log_R']
		except KeyError:
			pass
		#
		try:
			self.path_function = parameters['path']
		except KeyError:
			pass
		#
		try:
			self.dot_path_function = parameters['dot_path']
		except KeyError:
			pass
		#



	def set_a(self,a):
		#
		self.a_functions = a

	def set_D(self,D):
		#
		self.D_functions = D
		
	def set_R(self,R):
		#
		self.R_function = R

	def set_d_log_R(self,d_log_R):
		#
		self.d_log_R_function = d_log_R

	def set_path(self,path):
		#
		self.path_function = path

	def set_dot_path(self,dot_path):
		#
		self.dot_path_function = dot_path


	def evaluate_exit_rate_free(self,
						D,
						):
		#
		return self.pi_squared*D[0]/4

	def evaluate_exit_rate_0(self,
						E,
						D,
						):
		#
		a_0 = E[1]**2/4 \
			+3*D[1]**2/16 \
			+D[1]*E[1]/2 \
			-D[0]*E[2] \
			-D[0]*D[2]/2 \
			-self.pi_squared*D[1]**2/16 \
			+self.pi_squared*D[0]*D[2]/12
		#
		a_0 /= D[0]
		#
		return a_0

	def evaluate_exit_rate_2(self,
						E,dE,
						D,dD,
						d_log_R
						):
		#
		a_2 = -9*D[1]**4/32 \
			-D[0]**2*dD[2] \
			-3*D[0]**2*dE[2] \
			-2*E[1]**2*dD[0] \
			-2*D[0]**2*E[2]**2 \
			+6*D[0]**3*D[4] \
			+12*D[0]**3*E[4] \
			-3*D[0]**2*D[2]**2/2 \
			-3*D[1]**3*E[1]/4 \
			-3*D[1]**2*dD[0]/4 \
			-3*D[1]**2*E[1]**2/8 \
			-self.pi_squared**2*D[1]**4/64 \
			+3*self.pi_squared*D[1]**4/32 \
			+D[0]*D[2]*dD[0] \
			+D[0]*D[2]*E[1]**2/2 \
			-self.pi_squared*D[0]**3*D[4] \
			-8*D[0]**2*E[2]*d_log_R \
			-4*D[0]**2*D[2]*E[2] \
			-4*D[0]**2*D[2]*d_log_R \
			-3*D[0]**2*D[3]*E[1] \
			-3*D[0]**2*E[1]*E[3] \
			-2*self.pi_squared*D[0]**3*E[4] \
			+2*D[0]*E[1]*dE[1] \
			+2*D[0]*E[1]**2*d_log_R \
			+3*D[0]*E[2]*dD[0] \
			-13*D[1]*E[1]*dD[0]/4 \
			-3*D[0]**2*D[1]*D[3]/2 \
			-3*D[0]**2*D[1]*E[3]/2 \
			-self.pi_squared**2*D[0]**2*D[2]**2/60 \
			+self.pi_squared*D[0]**2*dE[2]/3 \
			+self.pi_squared*D[0]**2*E[2]**2/3 \
			+self.pi_squared*D[1]**3*E[1]/4 \
			+self.pi_squared*E[1]**2*dD[0]/4 \
			+self.pi_squared*D[0]**2*D[2]**2/4 \
			+self.pi_squared*D[0]**2*dD[2]/6 \
			+self.pi_squared*D[1]**2*E[1]**2/8 \
			+self.pi_squared**2*D[0]**3*D[4]/20 \
			+3*D[0]*D[1]**2*D[2]/2 \
			+3*D[0]*E[1]*dD[1]/2 \
			+3*D[0]*D[1]**2*E[2]/2 \
			+3*D[0]*D[1]**2*d_log_R/2 \
			+3*D[0]*D[1]*dD[1]/4 \
			+3*self.pi_squared*D[1]**2*dD[0]/16 \
			+7*D[0]*D[1]*dE[1]/4 \
			+self.pi_squared*D[0]**2*D[1]*E[3] \
			+self.pi_squared*D[0]**2*E[2]*d_log_R \
			+self.pi_squared*D[1]*E[1]*dD[0]/2 \
			+self.pi_squared*D[0]**2*D[3]*E[1]/2 \
			+self.pi_squared*D[0]**2*E[1]*E[3]/2 \
			+4*D[0]*D[1]*E[1]*d_log_R \
			-7*self.pi_squared*D[0]*D[1]**2*D[2]/16 \
			-7*self.pi_squared*D[0]*D[1]**2*d_log_R/16 \
			-7*self.pi_squared*D[0]*E[1]*dD[1]/24 \
			-5*self.pi_squared*D[0]*D[1]*dE[1]/24 \
			-3*self.pi_squared*D[0]*D[1]*dD[1]/16 \
			-3*self.pi_squared**2*D[0]**2*D[1]*D[3]/40 \
			-self.pi_squared*D[0]*D[1]**2*E[2]/2 \
			-self.pi_squared*D[0]*E[2]*dD[0]/3 \
			-self.pi_squared*D[0]*E[1]*dE[1]/4 \
			-self.pi_squared*D[0]*E[1]**2*d_log_R/4 \
			-self.pi_squared*D[0]*D[2]*dD[0]/6 \
			-self.pi_squared*D[0]*D[2]*E[1]**2/12 \
			+self.pi_squared**2*D[0]*D[1]**2*D[2]/16 \
			+2*self.pi_squared*D[0]**2*D[2]*E[2]/3 \
			+2*self.pi_squared*D[0]**2*D[2]*d_log_R/3 \
			+3*D[0]*D[1]*E[1]*E[2]/2 \
			+5*D[0]*D[1]*D[2]*E[1]/2 \
			+5*self.pi_squared*D[0]**2*D[1]*D[3]/8 \
			-3*self.pi_squared*D[0]*D[1]*E[1]*d_log_R/4 \
			-2*self.pi_squared*D[0]*D[1]*D[2]*E[1]/3 \
			-self.pi_squared*D[0]*D[1]*E[1]*E[2]/2
		#
		a_2 /= self.pi_squared * D[0]**3
		#
		return a_2
		

	def numerical_derivative(self,t,x):
		#
		dx = np.zeros_like(x)
		#
		dx[0] = x[1] - x[0]
		dx[0] /= t[1] - t[0]
		#
		dx[1:-1] = x[2:] - x[:-2]
		dx[1:-1] /= t[2:] - t[:-2]
		#
		dx[-1] = x[-1] - x[-2]
		dx[-1] /= t[-1] - t[-2]
		#
		return dx


	def evaluate_E(self,
					t,
					order=None,
					path=None,
					dot_path=None,
					d_log_R=None,
					):
		#
		if order is None:
			order = self.order
		#
		if path is None:
			path = self.path_function(t)
		#
		if dot_path is None:
			dot_path = self.dot_path_function(t)
		#
		if d_log_R is None:
			d_log_R = self.d_log_R_function(t)
		#
		E = np.zeros([order+1,len(t)],
					dtype=float)
		dE = np.zeros_like(E)
		#
		for n in range(1,min(order+1,len(self.a_functions))):
			#
			a_function = self.a_functions[n-1]
			#
			E[n] = -a_function(path)/self.factorials[n] 
			#
			if n == 1:
				E[n] += dot_path
			elif n == 2:
				E[n] += d_log_R/2.
			#
			dE[n] = self.numerical_derivative(t=t,x=E[n])
		#
		return E, dE



	def evaluate_D(self,
					t,
					order=None,
					path=None,
					):
		#
		if order is None:
			order = self.order
		#
		if path is None:
			path = self.path_function(t)
		#
		D = np.zeros([order+1,len(t)],
						dtype=float)
		dD = np.zeros_like(D)
		#
		for n in range(min(order+1,len(self.D_functions))):
			#
			D_function = self.D_functions[n]
			#
			D[n] = D_function(path)/self.factorials[n]
			#
			dD[n] = self.numerical_derivative(t=t,x=D[n])
		#
		return D, dD


	def eval_exit_rate(self,t,
							order=None,
							full_output=True):
		#
		if order is None:
			order = self.order
		#
		path_eval = self.path_function(t)
		#
		if self.dot_path_function is None:
			dot_path_eval = self.numerical_derivative(t=t,x=path_eval)
		else:
			dot_path_eval = self.dot_path_function(t)
		#
		R = self.R_function(t)
		if self.d_log_R_function is None:
			d_log_R = self.numerical_derivative(t=t,x=R)
			d_log_R /= R
		else:
			d_log_R = self.d_log_R_function(t)
		#
		E, dE = self.evaluate_E(
					t=t,
					order=order,
					path=path_eval,
					dot_path=dot_path_eval,
					)
		#
		D, dD = self.evaluate_D(
					t=t,
					order=order,
					path=path_eval,
					)
		#
		a_free = self.evaluate_exit_rate_free(
						D=D,
						)
		#
		if order >= 2:
			a_0 = self.evaluate_exit_rate_0(
							E=E,
							D=D,
							)
		#
		if order >= 4:
			a_2 = self.evaluate_exit_rate_2(
							E=E,
							dE=dE,
							D=D,
							dD=dD,
							d_log_R=d_log_R,
							)
		#
		if order < 2:
			exit_rate = a_free/R**2
		elif order < 4:
			exit_rate = a_free/R**2 + a_0
		else:
			exit_rate = a_free/R**2 + a_0 + a_2 * R**2
		#
		if full_output:
			output_dictionary = {'t':t,
								'R':R,
								'exit_rate':exit_rate,
								'a_free':a_free
			}
			if order >=2:
				output_dictionary['a_0'] = a_0
			if order >= 4:
				output_dictionary['a_2'] = a_2
			return output_dictionary
		else:
			return exit_rate
