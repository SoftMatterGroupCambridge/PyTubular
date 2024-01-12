import sympy

N_max = 6

Dk = sympy.symbols('D_0:%d'%N_max)
d_Dk = sympy.symbols('\dot{D}_0:%d'%N_max)
Ek = sympy.symbols('E_0:%d'%N_max)
d_Ek = sympy.symbols('\dot{E}_0:%d'%N_max)
d_log_eps = sympy.symbols('d_log_eps',real=True)
epsilon = sympy.symbols('epsilon',real=True,positive=True)

exit_rate_dict = {}

exit_rate_dict[0] = sympy.pi**2*Dk[0]/4

exit_rate_dict[1] = 0

exit_rate_dict[2] =  -Ek[2] \
	 -Dk[2]/2 \
	 +Ek[1]**2/(4*Dk[0]) \
	 +sympy.pi**2*Dk[2]/12 \
	 +3*Dk[1]**2/(16*Dk[0]) \
	 +Dk[1]*Ek[1]/(2*Dk[0]) \
	 -sympy.pi**2*Dk[1]**2/(16*Dk[0])

exit_rate_dict[3] = 0

exit_rate_dict[4] =  -Dk[4] \
	 -2*Ek[4] \
	 +6*Dk[4]/sympy.pi**2 \
	 +12*Ek[4]/sympy.pi**2 \
	 +d_Ek[2]/(3*Dk[0]) \
	 +Ek[2]**2/(3*Dk[0]) \
	 +Dk[2]**2/(4*Dk[0]) \
	 +d_Dk[2]/(6*Dk[0]) \
	 +sympy.pi**2*Dk[4]/20 \
	 +3*Dk[1]**4/(32*Dk[0]**3) \
	 +Dk[1]*Ek[3]/Dk[0] \
	 +Ek[2]*d_log_eps/Dk[0] \
	 +Dk[3]*Ek[1]/(2*Dk[0]) \
	 +Ek[1]*Ek[3]/(2*Dk[0]) \
	 -d_Dk[2]/(sympy.pi**2*Dk[0]) \
	 -3*d_Ek[2]/(sympy.pi**2*Dk[0]) \
	 -2*Ek[2]**2/(sympy.pi**2*Dk[0]) \
	 -9*Dk[1]**4/(32*sympy.pi**2*Dk[0]**3) \
	 -7*Dk[1]**2*Dk[2]/(16*Dk[0]**2) \
	 -7*Dk[1]**2*d_log_eps/(16*Dk[0]**2) \
	 -7*Ek[1]*d_Dk[1]/(24*Dk[0]**2) \
	 -5*Dk[1]*d_Ek[1]/(24*Dk[0]**2) \
	 -3*Dk[2]**2/(2*sympy.pi**2*Dk[0]) \
	 -3*Dk[1]*d_Dk[1]/(16*Dk[0]**2) \
	 -Dk[1]**2*Ek[2]/(2*Dk[0]**2) \
	 -Ek[2]*d_Dk[0]/(3*Dk[0]**2) \
	 -Ek[1]*d_Ek[1]/(4*Dk[0]**2) \
	 -Ek[1]**2*d_log_eps/(4*Dk[0]**2) \
	 -Dk[2]*d_Dk[0]/(6*Dk[0]**2) \
	 -Dk[2]*Ek[1]**2/(12*Dk[0]**2) \
	 -sympy.pi**2*Dk[2]**2/(60*Dk[0]) \
	 -sympy.pi**2*Dk[1]**4/(64*Dk[0]**3) \
	 +Dk[1]**3*Ek[1]/(4*Dk[0]**3) \
	 +Ek[1]**2*d_Dk[0]/(4*Dk[0]**3) \
	 +Dk[1]**2*Ek[1]**2/(8*Dk[0]**3) \
	 +2*Dk[2]*Ek[2]/(3*Dk[0]) \
	 +2*Dk[2]*d_log_eps/(3*Dk[0]) \
	 +3*Dk[1]**2*d_Dk[0]/(16*Dk[0]**3) \
	 +5*Dk[1]*Dk[3]/(8*Dk[0]) \
	 +Dk[2]*d_Dk[0]/(sympy.pi**2*Dk[0]**2) \
	 +Dk[1]*Ek[1]*d_Dk[0]/(2*Dk[0]**3) \
	 +Dk[2]*Ek[1]**2/(2*sympy.pi**2*Dk[0]**2) \
	 -8*Ek[2]*d_log_eps/(sympy.pi**2*Dk[0]) \
	 -4*Dk[2]*Ek[2]/(sympy.pi**2*Dk[0]) \
	 -4*Dk[2]*d_log_eps/(sympy.pi**2*Dk[0]) \
	 -3*Dk[3]*Ek[1]/(sympy.pi**2*Dk[0]) \
	 -3*Ek[1]*Ek[3]/(sympy.pi**2*Dk[0]) \
	 -2*Ek[1]**2*d_Dk[0]/(sympy.pi**2*Dk[0]**3) \
	 +2*Ek[1]*d_Ek[1]/(sympy.pi**2*Dk[0]**2) \
	 +2*Ek[1]**2*d_log_eps/(sympy.pi**2*Dk[0]**2) \
	 +3*Ek[2]*d_Dk[0]/(sympy.pi**2*Dk[0]**2) \
	 -3*Dk[1]*Dk[3]/(2*sympy.pi**2*Dk[0]) \
	 -3*Dk[1]*Ek[3]/(2*sympy.pi**2*Dk[0]) \
	 -3*Dk[1]*Ek[1]*d_log_eps/(4*Dk[0]**2) \
	 -3*Dk[1]**3*Ek[1]/(4*sympy.pi**2*Dk[0]**3) \
	 -3*Dk[1]**2*d_Dk[0]/(4*sympy.pi**2*Dk[0]**3) \
	 -3*Dk[1]**2*Ek[1]**2/(8*sympy.pi**2*Dk[0]**3) \
	 -3*sympy.pi**2*Dk[1]*Dk[3]/(40*Dk[0]) \
	 -2*Dk[1]*Dk[2]*Ek[1]/(3*Dk[0]**2) \
	 -Dk[1]*Ek[1]*Ek[2]/(2*Dk[0]**2) \
	 +sympy.pi**2*Dk[1]**2*Dk[2]/(16*Dk[0]**2) \
	 +3*Dk[1]**2*Dk[2]/(2*sympy.pi**2*Dk[0]**2) \
	 +3*Ek[1]*d_Dk[1]/(2*sympy.pi**2*Dk[0]**2) \
	 +3*Dk[1]**2*Ek[2]/(2*sympy.pi**2*Dk[0]**2) \
	 +3*Dk[1]**2*d_log_eps/(2*sympy.pi**2*Dk[0]**2) \
	 +3*Dk[1]*d_Dk[1]/(4*sympy.pi**2*Dk[0]**2) \
	 +7*Dk[1]*d_Ek[1]/(4*sympy.pi**2*Dk[0]**2) \
	 +4*Dk[1]*Ek[1]*d_log_eps/(sympy.pi**2*Dk[0]**2) \
	 -13*Dk[1]*Ek[1]*d_Dk[0]/(4*sympy.pi**2*Dk[0]**3) \
	 +3*Dk[1]*Ek[1]*Ek[2]/(2*sympy.pi**2*Dk[0]**2) \
	 +5*Dk[1]*Dk[2]*Ek[1]/(2*sympy.pi**2*Dk[0]**2)

exit_rate_dict[5] = 0



exit_rate_lambda_dict = {}
for i,term in exit_rate_dict.items():
	exit_rate_lambda_dict[i] = sympy.lambdify( (
									Dk,
									d_Dk,
									Ek,
									d_Ek,
									d_log_eps
									), term)

exit_rate = 0
for i,term in exit_rate_dict.items():
	exit_rate += epsilon**i * term
exit_rate_lambda = sympy.lambdify( (
								Dk,
								d_Dk,
								Ek,
								d_Ek,
								d_log_eps,
								epsilon
								), exit_rate)
