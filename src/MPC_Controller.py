import cvxpy as cp
import numpy as np

# Define problem data
n = 12  # Dimension of state x
m = 4  # Dimension of control input u
T = 10  # Number of time steps

# Define matrices Q and R
Q = np.eye(n)  # Q should be positive semi-definite
Q_f = np.eye(n) * 2
R = np.eye(m)  # R should be positive definite

# Define matrices A, B, and x_g
A = np.load('/A.npy') # np.random.randn(n,n)  # Example: Identity matrix
B = np.load('/B.npy')# np.random.randn(n,m)  # Example: First m columns of the identity matrix
x_g = np.random.randn(n)  # Given value of x_g

# Define variables
x = cp.Variable((n, T+1))
u = cp.Variable((m, T))

# Define cost function
cost = 0
for k in range(T):
    cost += 0.5 * cp.quad_form(x[:, k] - x_g, Q) + 0.5 * cp.quad_form(u[:, k], R)
cost += 0.5 * cp.quad_form(x[:, T] - x_g, Q_f)

# Adding equality constraints
constraints = []
for k in range(T):
    constraints.append(x[:, k+1] == A @ x[:, k] + B @ u[:, k])

# Inequality constraints on controls
v_min = 0 # -np.ones((n, 1)) * 10  # Minimum value for each state
v_max = 2200 # np.ones((n, 1)) * 10  # Maximum value for each state

omega_min = 0
omega_max = 5.5

roll_min = -10
roll_max = 10

pitch_min = -10
pitch_max = 10

for k in range(T+1):
    constraints += [x[3, k] <= roll_max, x[3, k] >= roll_min]
    constraints += [x[4, k] <= pitch_max, x[4, k] >= pitch_min]

    # v_norm = cp.norm(x[6:9, k], 'inf') #x[6, k]**2 + x[7, k]**2 + x[8, k]**2 

    constraints += [cp.norm(x[6:9, k]) <= v_max]#, v_norm >= v_min]
    constraints += [cp.norm(x[9:12, k]) <= omega_max]#, cp.norm(x[9:12, k]) >= omega_min]

# Equality constraints on initial and final state
xi = np.zeros(n)
xg = np.ones(3)

constraints.append(x[:,0] == xi)
constraints.append(x[:3,-1] == xg)


# Inequality constraints on controls
u_min = -np.ones(m) # Minimum value for each control
u_max = np.ones(m) # Maximum value for each control
u_min[-1] = 0
for k in range(T):
    constraints += [u[:, k] <= u_max, u[:, k] >= u_min]

# Adding the cone constraints
# gamma = 1 # Assuming 45 deg cone
# for k in range(T+1):
#   dot_product = np.dot(x[:3, k], xg[:3])
#   b_norm_sq = np.dot(xg[:3], xg[:3])
#   projection = (dot_product / b_norm_sq) * xg[:3]
#   constraints += [projection - x[:3] < gamma*projection] 
# Adding the cone constraints (scalarization)
gamma = 1  # Assuming 45 deg cone
for k in range(T + 1):
    dot_product = cp.sum(x[:3, k] @ xg[:3])
    b_norm_sq = cp.sum(xg[:3] @ xg[:3])
    projection = (dot_product / b_norm_sq) * xg[:3]
    # constraints += [cp.norm(x[:3, k] - projection) <= gamma * cp.norm(x[:3, k])]
    # constraints += [cp.norm(x[3, k]) <= cp.norm(x[2, k])]





# Define the optimization problem
prob = cp.Problem(cp.Minimize(cost), constraints)

# Solve the problem
prob.solve()

# Print results
print("Optimal xi:")
print(x.value)
print("Optimal u:")
print(u.value)
print("Optimal cost:", prob.value)
