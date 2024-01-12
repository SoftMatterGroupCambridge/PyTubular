import sympy

N_max = 6

Dk = sympy.symbols('D_0:%d'%N_max)
Ek = sympy.symbols('E_0:%d'%N_max)

n = sympy.symbols('n',integer=True,nonnegative=True)
epsilon = sympy.symbols('epsilon',real=True,nonnegative=True)


eigenvalues_dict = {}

eigenvalues_dict[0] = sympy.pi**2*Dk[0]*n**2/4

eigenvalues_dict[1] = 0

eigenvalues_dict[2] = -Ek[2]  \
	-Dk[2]/2  \
	+Ek[1]**2/(4*Dk[0])  \
	+3*Dk[1]**2/(16*Dk[0])  \
	+Dk[1]*Ek[1]/(2*Dk[0])  \
	+sympy.pi**2*Dk[2]*n**2/12  \
	-sympy.pi**2*Dk[1]**2*n**2/(16*Dk[0])

eigenvalues_dict[3] = 0

eigenvalues_dict[4] = -Dk[4]  \
	-2*Ek[4]  \
	+Ek[2]**2/(3*Dk[0])  \
	+Dk[2]**2/(4*Dk[0])  \
	+3*Dk[1]**4/(32*Dk[0]**3)  \
	+Dk[1]*Ek[3]/Dk[0]  \
	+Dk[3]*Ek[1]/(2*Dk[0])  \
	+Ek[1]*Ek[3]/(2*Dk[0])  \
	+6*Dk[4]/(sympy.pi**2*n**2)  \
	+12*Ek[4]/(sympy.pi**2*n**2)  \
	-7*Dk[1]**2*Dk[2]/(16*Dk[0]**2)  \
	-Dk[1]**2*Ek[2]/(2*Dk[0]**2)  \
	-Dk[2]*Ek[1]**2/(12*Dk[0]**2)  \
	+Dk[1]**3*Ek[1]/(4*Dk[0]**3)  \
	+Dk[1]**2*Ek[1]**2/(8*Dk[0]**3)  \
	+sympy.pi**2*Dk[4]*n**2/20  \
	+2*Dk[2]*Ek[2]/(3*Dk[0])  \
	+5*Dk[1]*Dk[3]/(8*Dk[0])  \
	-2*Ek[2]**2/(sympy.pi**2*Dk[0]*n**2)  \
	-9*Dk[1]**4/(32*sympy.pi**2*Dk[0]**3*n**2)  \
	-3*Dk[2]**2/(2*sympy.pi**2*Dk[0]*n**2)  \
	-2*Dk[1]*Dk[2]*Ek[1]/(3*Dk[0]**2)  \
	-Dk[1]*Ek[1]*Ek[2]/(2*Dk[0]**2)  \
	-sympy.pi**2*Dk[2]**2*n**2/(60*Dk[0])  \
	-sympy.pi**2*Dk[1]**4*n**2/(64*Dk[0]**3)  \
	+Dk[2]*Ek[1]**2/(2*sympy.pi**2*Dk[0]**2*n**2)  \
	-4*Dk[2]*Ek[2]/(sympy.pi**2*Dk[0]*n**2)  \
	-3*Dk[3]*Ek[1]/(sympy.pi**2*Dk[0]*n**2)  \
	-3*Ek[1]*Ek[3]/(sympy.pi**2*Dk[0]*n**2)  \
	-3*Dk[1]*Dk[3]/(2*sympy.pi**2*Dk[0]*n**2)  \
	-3*Dk[1]*Ek[3]/(2*sympy.pi**2*Dk[0]*n**2)  \
	-3*Dk[1]**3*Ek[1]/(4*sympy.pi**2*Dk[0]**3*n**2)  \
	-3*Dk[1]**2*Ek[1]**2/(8*sympy.pi**2*Dk[0]**3*n**2)  \
	-3*sympy.pi**2*Dk[1]*Dk[3]*n**2/(40*Dk[0])  \
	+sympy.pi**2*Dk[1]**2*Dk[2]*n**2/(16*Dk[0]**2)  \
	+3*Dk[1]**2*Dk[2]/(2*sympy.pi**2*Dk[0]**2*n**2)  \
	+3*Dk[1]**2*Ek[2]/(2*sympy.pi**2*Dk[0]**2*n**2)  \
	+3*Dk[1]*Ek[1]*Ek[2]/(2*sympy.pi**2*Dk[0]**2*n**2)  \
	+5*Dk[1]*Dk[2]*Ek[1]/(2*sympy.pi**2*Dk[0]**2*n**2)

eigenvalues_dict[5] = 0



eigenvalues_lambda_dict = {}
for i,item in eigenvalues_dict.items():
	eigenvalues_lambda_dict[i] = sympy.lambdify( (Dk,Ek,n), item)


eigenvalues = 0
for i,eigenvalue in eigenvalues_dict.items():
	eigenvalues += epsilon**i * eigenvalue
eigenvalues_lambda = sympy.lambdify( (Dk,Ek,n,epsilon), eigenvalues)
