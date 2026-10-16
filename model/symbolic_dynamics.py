import numpy as np
import sympy as sp
import scipy
import mujoco
from sympy.physics.mechanics import dynamicsymbols

# ------------------------------------------------------------------
# 0. Load MuJoCo model (used to read parameters and to verify results)
# ------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path("model/robot.xml")
data = mujoco.MjData(model)

# ------------------------------------------------------------------
# 1. Symbols
# ------------------------------------------------------------------
t = dynamicsymbols._t
g = sp.Symbol('g')
z0 = sp.Symbol('z0')  # torso reference height (slide-joint qpos is measured from it)

m_torso, m_thigh, m_shin, m_foot = sp.symbols('m_torso m_thigh m_shin m_foot')
I_torso, I_thigh, I_shin, I_foot = sp.symbols('I_torso I_thigh I_shin I_foot')
l_thigh, l_shin = sp.symbols('l_thigh l_shin')
c_torso, c_thigh, c_shin = sp.symbols('c_torso c_thigh c_shin')  # CoM z-offsets in body frame
c_fx, c_fz = sp.symbols('c_fx c_fz')                             # foot CoM offset in foot frame
s_x, s_z = sp.symbols('s_x s_z')                                 # foot site offset in foot frame
armature_hip, armature_knee, armature_ankle = sp.symbols(
    'armature_hip armature_knee armature_ankle')

# ------------------------------------------------------------------
# 2. Generalized coordinates (same order as MuJoCo qpos)
# ------------------------------------------------------------------
x, z, theta = dynamicsymbols('x z theta')
hL, kL, aL, hR, kR, aR = dynamicsymbols('hip_L knee_L ankle_L hip_R knee_R ankle_R')

q = sp.Matrix([x, z, theta, hL, kL, aL, hR, kR, aR])
q_dot = q.diff(t)
num_dof = len(q)


# ------------------------------------------------------------------
# 3. Planar kinematics
#    Rotation about +y by angle phi maps a body-frame point (cx, cz) to
#    world offset (cx*cos(phi) + cz*sin(phi), -cx*sin(phi) + cz*cos(phi)).
# ------------------------------------------------------------------
def place(origin, phi, cx, cz):
    return (origin[0] + cx * sp.cos(phi) + cz * sp.sin(phi),
            origin[1] - cx * sp.sin(phi) + cz * sp.cos(phi))


def build_leg(hip_origin, phi_torso, h, k, a):
    phi_thigh = phi_torso + h
    phi_shin = phi_thigh + k
    phi_foot = phi_shin + a

    com_thigh = place(hip_origin, phi_thigh, 0, c_thigh)
    knee = place(hip_origin, phi_thigh, 0, -l_thigh)

    com_shin = place(knee, phi_shin, 0, c_shin)
    ankle = place(knee, phi_shin, 0, -l_shin)

    com_foot = place(ankle, phi_foot, c_fx, c_fz)
    site = place(ankle, phi_foot, s_x, s_z)

    bodies = [(m_thigh, I_thigh, com_thigh, phi_thigh),
              (m_shin, I_shin, com_shin, phi_shin),
              (m_foot, I_foot, com_foot, phi_foot)]
    return bodies, site


torso_origin = (x, z0 + z)
com_torso = place(torso_origin, theta, 0, c_torso)

left_bodies, left_site = build_leg(torso_origin, theta, hL, kL, aL)
right_bodies, right_site = build_leg(torso_origin, theta, hR, kR, aR)

bodies = [(m_torso, I_torso, com_torso, theta)] + left_bodies + right_bodies

# ------------------------------------------------------------------
# 4. Kinetic and potential energy
# ------------------------------------------------------------------
half = sp.Rational(1, 2)
T = 0
V = 0
for m, I, com, phi in bodies:
    vx = sp.diff(com[0], t)
    vz = sp.diff(com[1], t)
    T += half * m * (vx**2 + vz**2) + half * I * sp.diff(phi, t)**2
    V += m * g * com[1]

# Rotor armature: MuJoCo adds it to the diagonal of M, so include it in T here
# (do NOT add it to M_sym again afterwards)
armatures = [armature_hip, armature_knee, armature_ankle] * 2
for qd, arm in zip(list(q_dot[3:]), armatures):
    T += half * arm * qd**2

# ------------------------------------------------------------------
# 5. Extract M(q) as a 9x9 matrix and g(q) as a 9x1 vector
# ------------------------------------------------------------------
M_sym = sp.zeros(num_dof, num_dof)
for i in range(num_dof):
    dT_dqdot_i = sp.diff(T, q_dot[i])
    for j in range(i, num_dof):  # M is symmetric, so compute the upper triangle only
        M_sym[i, j] = sp.diff(dT_dqdot_i, q_dot[j])
        M_sym[j, i] = M_sym[i, j]

g_sym = sp.zeros(num_dof, 1)
for i in range(num_dof):
    g_sym[i, 0] = sp.diff(V, q[i])

foot_sym = sp.Matrix(left_site)  # world (x, z) of left_foot_site


# ------------------------------------------------------------------
# 6. Read numeric parameters from the MuJoCo model
# ------------------------------------------------------------------
def body_params(name):
    b = model.body(name)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, b.iquat)
    R = R.reshape(3, 3)
    I_full = R @ np.diag(b.inertia) @ R.T  # inertia about CoM in the body frame
    return float(b.mass[0]), b.ipos.copy(), float(I_full[1, 1])  # I_yy for the sagittal plane


def dof_armature(joint_name):
    return float(model.dof_armature[model.joint(joint_name).dofadr[0]])


mT, pT, IT = body_params("torso")
mTh, pTh, ITh = body_params("left_thigh")
mS, pS, IS = body_params("left_shin")
mF, pF, IF = body_params("left_foot")
site_pos = model.site("left_foot_site").pos

params = {
    g: float(-model.opt.gravity[2]),
    z0: float(model.body("torso").pos[2]),
    m_torso: mT, I_torso: IT, c_torso: float(pT[2]),
    m_thigh: mTh, I_thigh: ITh, c_thigh: float(pTh[2]),
    l_thigh: -float(model.body("left_shin").pos[2]),
    m_shin: mS, I_shin: IS, c_shin: float(pS[2]),
    l_shin: -float(model.body("left_foot").pos[2]),
    m_foot: mF, I_foot: IF, c_fx: float(pF[0]), c_fz: float(pF[2]),
    s_x: float(site_pos[0]), s_z: float(site_pos[2]),
    armature_hip: dof_armature("left_hip"),
    armature_knee: dof_armature("left_knee"),
    armature_ankle: dof_armature("left_ankle"),
}

# ------------------------------------------------------------------
# 7. Turn the symbolic expressions into fast numeric functions
# ------------------------------------------------------------------
qs = sp.symbols('qs0:%d' % num_dof)
q_to_plain = dict(zip(list(q), qs))


def to_func(expr):
    expr = expr.subs(params).subs(q_to_plain)
    return sp.lambdify(qs, expr, 'numpy', cse=True)


M_fn = to_func(M_sym)
g_fn = to_func(g_sym)
foot_fn = to_func(foot_sym)


# ------------------------------------------------------------------
# 8. Compare with MuJoCo
# ------------------------------------------------------------------
def mujoco_dynamics(q_sample):
    data.qpos[:] = q_sample
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)

    foot_pos = data.site("left_foot_site").xpos.copy()
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    g_vec = data.qfrc_bias.copy()
    return foot_pos, M, g_vec




rng = np.random.default_rng(0)
for trial in range(20):
    q_sample = rng.uniform(-0.5, 0.5, num_dof)

    foot_mj, M_mj, g_mj = mujoco_dynamics(q_sample)
    M_sy = np.array(M_fn(*q_sample), dtype=float)
    g_sy = np.array(g_fn(*q_sample), dtype=float).ravel()
    foot_sy = np.array(foot_fn(*q_sample), dtype=float).ravel()

    print(f"trial {trial}: "
          f"|dM|max = {np.abs(M_sy - M_mj).max():.2e}, "
          f"|dg|max = {np.abs(g_sy - g_mj).max():.2e}, "
          f"|dfoot|max = {np.abs(foot_sy - foot_mj[[0, 2]]).max():.2e}")

def generate_validation_table(num_samples=20):
    results = []
    rng = np.random.default_rng(42)

    for trial in range(num_samples):
        q_sample = rng.uniform(-0.5, 0.5, num_dof)

        foot_mj, M_mj, g_mj = mujoco_dynamics(q_sample)

        M_sy = np.array(M_fn(*q_sample), dtype=float)
        g_sy = np.array(g_fn(*q_sample), dtype=float).ravel()
        foot_sy = np.array(foot_fn(*q_sample), dtype=float).ravel()

        err_M = np.abs(M_sy - M_mj).max()
        err_g = np.abs(g_sy - g_mj).max()
        err_p = np.abs(foot_sy - foot_mj[[0, 2]]).max()

        status = "PASS" if max(err_M, err_g, err_p) <= 1e-5 else "FAIL"
        results.append((trial + 1, err_p, err_M, err_g, status))

    return results


# ------------------------------------------------------------------
# 9. Coriolis / centrifugal matrix C(q, q_dot) via Christoffel symbols
# ------------------------------------------------------------------
qd_s = sp.symbols('qd0:%d' % num_dof)

# M as a function of plain symbols with numeric parameters substituted
M_p = M_sym.subs(params).subs(q_to_plain)

# dM[k][i, j] = dM_ij / dq_k
dM = [M_p.diff(qs[k]) for k in range(num_dof)]

# C_ij = sum_k Gamma_ijk * qd_k,  Gamma_ijk = 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
C_sym = sp.zeros(num_dof, num_dof)
for i in range(num_dof):
    for j in range(num_dof):
        s = 0
        for k in range(num_dof):
            s += sp.Rational(1, 2) * (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qd_s[k]
        C_sym[i, j] = s

# M_dot, used for the skew-symmetry check
Mdot_sym = sp.zeros(num_dof, num_dof)
for k in range(num_dof):
    Mdot_sym += dM[k] * qd_s[k]

C_fn = sp.lambdify((*qs, *qd_s), C_sym, 'numpy', cse=True)
Mdot_fn = sp.lambdify((*qs, *qd_s), Mdot_sym, 'numpy', cse=True)


# ------------------------------------------------------------------
# 10. Compare C(q, qd) qd + g(q) with MuJoCo's qfrc_bias at nonzero velocity
# ------------------------------------------------------------------
def mujoco_bias(q_sample, qd_sample):
    data.qpos[:] = q_sample
    data.qvel[:] = qd_sample
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.qfrc_bias.copy()


print("\nCoriolis check:")
for trial in range(20):
    q_sample = rng.uniform(-0.5, 0.5, num_dof)
    qd_sample = rng.uniform(-2.0, 2.0, num_dof)

    C_num = np.array(C_fn(*q_sample, *qd_sample), dtype=float)
    Mdot_num = np.array(Mdot_fn(*q_sample, *qd_sample), dtype=float)
    g_num = np.array(g_fn(*q_sample), dtype=float).ravel()

    bias_sy = C_num @ qd_sample + g_num
    bias_mj = mujoco_bias(q_sample, qd_sample)

    N = Mdot_num - 2.0 * C_num  # should be skew-symmetric
    print(f"trial {trial}: "
          f"|dbias|max = {np.abs(bias_sy - bias_mj).max():.2e}, "
          f"|N + N^T|max = {np.abs(N + N.T).max():.2e}")


# ------------------------------------------------------------------
# 11. Heel / toe contact points and the contact Jacobian J_c
# ------------------------------------------------------------------
heel_x, toe_x, sole_z = sp.symbols('heel_x toe_x sole_z')  # corner coords in the foot frame

foot_geom = model.geom("left_foot_geom")
heel_val = float(foot_geom.pos[0] - foot_geom.size[0])
toe_val = float(foot_geom.pos[0] + foot_geom.size[0])
sole_val = float(foot_geom.pos[2] - foot_geom.size[2])
params.update({heel_x: heel_val, toe_x: toe_val, sole_z: sole_val})


def foot_corners(foot_body):
    # foot_body = (m, I, com_world, phi_world); corners are offsets from the foot CoM
    _, _, com_foot, phi_foot = foot_body
    heel = place(com_foot, phi_foot, heel_x - c_fx, sole_z - c_fz)
    toe = place(com_foot, phi_foot, toe_x - c_fx, sole_z - c_fz)
    return heel, toe


L_heel, L_toe = foot_corners(left_bodies[2])
R_heel, R_toe = foot_corners(right_bodies[2])

# Stacked world positions: [Lheel_x, Lheel_z, Ltoe_x, Ltoe_z, Rheel_x, Rheel_z, Rtoe_x, Rtoe_z]
contact_pts = sp.Matrix([*L_heel, *L_toe, *R_heel, *R_toe])  # 8x1
J_c_sym = contact_pts.jacobian(q)                            # 8x9

pts_fn = to_func(contact_pts)
Jc_fn = to_func(J_c_sym)

# Actuation selection matrix: base (x, z, theta) is unactuated, 6 joint torques
B = np.vstack([np.zeros((3, 6)), np.eye(6)])  # 9x6


# ------------------------------------------------------------------
# 12. Compare with MuJoCo (mj_jac at the corner points)
# ------------------------------------------------------------------
def mujoco_contacts(q_sample):
    data.qpos[:] = q_sample
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pts, Js = [], []
    for body_name in ("left_foot", "right_foot"):
        bid = model.body(body_name).id
        R = data.xmat[bid].reshape(3, 3)
        for lx in (heel_val, toe_val):
            p = data.xpos[bid] + R @ np.array([lx, 0.0, sole_val])
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, None, p, bid)
            pts += [p[0], p[2]]
            Js.append(jacp[[0, 2], :])  # keep the x and z rows
    return np.array(pts), np.vstack(Js)


print("\nContact Jacobian check:")
for trial in range(20):
    q_sample = rng.uniform(-0.5, 0.5, num_dof)
    pts_mj, Jc_mj = mujoco_contacts(q_sample)
    pts_sy = np.array(pts_fn(*q_sample), dtype=float).ravel()
    Jc_sy = np.array(Jc_fn(*q_sample), dtype=float)
    print(f"trial {trial}: "
          f"|dpts|max = {np.abs(pts_sy - pts_mj).max():.2e}, "
          f"|dJc|max = {np.abs(Jc_sy - Jc_mj).max():.2e}")

# Sanity check: in the 'stand' keyframe all four corners should sit close to the floor
mujoco.mj_resetDataKeyframe(model, data, 0)
q_stand = data.qpos.copy()
print("\nCorner heights in 'stand' pose (m):",
      np.round(np.array(pts_fn(*q_stand), dtype=float).ravel()[1::2], 4))



# ------------------------------------------------------------------
# 13. Quasi-static standing check:  B tau + J_c^T lam = g(q)   (qd = 0, qdd = 0)
#     NumPy-only QP: null-space elimination + Hildreth's dual method
# ------------------------------------------------------------------
mu = float(model.geom("left_foot_geom").friction[0])
tau_max = model.actuator_ctrlrange[:, 1].copy()  # per-actuator torque limit (6,)
n_tau, n_lam = 6, 8
n_var = n_tau + n_lam

mujoco.mj_resetDataKeyframe(model, data, 0)
q_stand = data.qpos.copy()

g_st = np.array(g_fn(*q_stand), dtype=float).ravel()      # 9
Jc_st = np.array(Jc_fn(*q_stand), dtype=float)            # 8 x 9
pts_st = np.array(pts_fn(*q_stand), dtype=float).ravel()  # 8: [x, z] per corner

# Unknowns u = [tau (6), lam (8)],  equality: A_eq u = g
A_eq = np.hstack([B, Jc_st.T])                            # 9 x 14

# Inequalities  G u >= h : lam_z >= 0, friction cone, and torque limits
rows, rhs = [], []
for i in range(4):
    ix, iz = n_tau + 2 * i, n_tau + 2 * i + 1
    r_ = np.zeros(n_var); r_[iz] = 1.0;              rows.append(r_); rhs.append(0.0)
    r_ = np.zeros(n_var); r_[iz] = mu; r_[ix] = -1.0; rows.append(r_); rhs.append(0.0)
    r_ = np.zeros(n_var); r_[iz] = mu; r_[ix] = 1.0;  rows.append(r_); rhs.append(0.0)
for k in range(n_tau):
    r_ = np.zeros(n_var); r_[k] = 1.0;  rows.append(r_); rhs.append(-tau_max[k])
    r_ = np.zeros(n_var); r_[k] = -1.0; rows.append(r_); rhs.append(-tau_max[k])
G = np.array(rows)
h = np.array(rhs)

# Objective: 1/2 u^T W u  (torque effort, small weight on contact forces)
W = np.diag([1.0] * n_tau + [1e-3] * n_lam)

# Eliminate the equality constraints: u = u0 + N z
u0 = np.linalg.pinv(A_eq) @ g_st
_, _, Vt = np.linalg.svd(A_eq)
N = Vt[A_eq.shape[0]:].T                                  # 14 x 5 null-space basis
P = N.T @ W @ N
q_lin = N.T @ W @ u0
D = G @ N
d = h - G @ u0

# Dual problem: min 1/2 mu^T Q mu - r^T mu, mu >= 0, solved by coordinate descent
Pinv = np.linalg.inv(P)
Q = D @ Pinv @ D.T
r_vec = d + D @ Pinv @ q_lin
mu_d = np.zeros(len(d))
converged = False
for sweep in range(50000):
    old = mu_d.copy()
    for i in range(len(d)):
        if Q[i, i] > 1e-12:
            mu_d[i] = max(0.0, mu_d[i] + (r_vec[i] - Q[i] @ mu_d) / Q[i, i])
    if np.abs(mu_d - old).max() < 1e-11:
        converged = True
        break

z = Pinv @ (D.T @ mu_d - q_lin)
u = u0 + N @ z

tau = u[:n_tau]
lam = u[n_tau:].reshape(4, 2)  # rows: L heel, L toe, R heel, R toe; cols: (x, z)
weight = float(model.body_mass.sum() * -model.opt.gravity[2])

print("\nQuasi-static standing check")
print("solver converged:", converged, f"({sweep + 1} sweeps)")
print("equality residual max:", f"{np.abs(A_eq @ u - g_st).max():.2e}")
print("worst inequality slack (>= 0 is feasible):", f"{(G @ u - h).min():.2e}")

print("\njoint torques (Nm)  [L hip, L knee, L ankle, R hip, R knee, R ankle]:")
print(np.round(tau, 3))
print("torque limits   :", tau_max)

names = ["L heel", "L toe", "R heel", "R toe"]
print("\ncontact forces (N):")
for nm, (fx, fz) in zip(names, lam):
    ratio = abs(fx) / fz if fz > 1e-9 else 0.0
    print(f"  {nm}: Fx = {fx:8.3f}, Fz = {fz:8.3f}, |Fx|/Fz = {ratio:.3f} (limit mu = {mu})")

print(f"\nsum Fz = {lam[:, 1].sum():.3f} N   (robot weight = {weight:.3f} N)")
print(f"sum Fx = {lam[:, 0].sum():.3e} N   (should be ~0)")

# Planar ZMP (= centre of pressure on flat ground)
zmp_x = (pts_st[0::2] * lam[:, 1]).sum() / lam[:, 1].sum()
heel_x_w, toe_x_w = pts_st[0], pts_st[2]  # left foot corners, world x
print(f"\nZMP x = {zmp_x:.4f} m   (heel = {heel_x_w:.4f}, toe = {toe_x_w:.4f})")
print(f"margin to heel = {zmp_x - heel_x_w:.4f} m, margin to toe = {toe_x_w - zmp_x:.4f} m")
print("ZMP inside support polygon:", heel_x_w <= zmp_x <= toe_x_w)

# ------------------------------------------------------------------
# 14. Foot-flat constrained forward dynamics
#     M qdd + C qd + g = B tau + Jr^T lam,   Jr qdd + Jr_dot qd = 0
# ------------------------------------------------------------------
# A flat foot is a rigid body with 3 DOF, so its 4 corner constraints (x and z at
# heel and toe) are redundant: the heel-toe distance is fixed. Use 3 independent
# ones per foot: heel x, heel z, toe z  ->  rows 0, 1, 3 (left) and 4, 5, 7 (right).
rows_r = [0, 1, 3, 4, 5, 7]
J_r_p = J_c_sym.extract(rows_r, list(range(num_dof))).subs(params).subs(q_to_plain)

Jdot_p = sp.zeros(len(rows_r), num_dof)
for k in range(num_dof):
    Jdot_p += J_r_p.diff(qs[k]) * qd_s[k]

Jr_fn = sp.lambdify(qs, J_r_p, 'numpy', cse=True)
Jdot_fn = sp.lambdify((*qs, *qd_s), Jdot_p, 'numpy', cse=True)


def constrained_dynamics(q_, qd_, tau_):
    M_ = np.array(M_fn(*q_), dtype=float)
    C_ = np.array(C_fn(*q_, *qd_), dtype=float)
    g_ = np.array(g_fn(*q_), dtype=float).ravel()
    J_ = np.array(Jr_fn(*q_), dtype=float)
    Jd_ = np.array(Jdot_fn(*q_, *qd_), dtype=float)
    n_c = J_.shape[0]
    K = np.block([[M_, -J_.T], [J_, np.zeros((n_c, n_c))]])
    rhs = np.concatenate([B @ tau_ - C_ @ qd_ - g_, -Jd_ @ qd_])
    sol = np.linalg.solve(K, rhs)
    return sol[:num_dof], sol[num_dof:]  # qdd, lam


tau_ff = tau.copy()  # feedforward torques from the quasi-static QP

# (a) Equilibrium: stand pose + QP torques must give qdd = 0 and the QP contact forces
qdd_eq, lam_eq = constrained_dynamics(q_stand, np.zeros(num_dof), tau_ff)
print("\nConstrained dynamics at equilibrium")
print("max |qdd| =", f"{np.abs(qdd_eq).max():.2e}")
print("lam [L heel x, L heel z, L toe z, R heel x, R heel z, R toe z] =", np.round(lam_eq, 3))

# (b) Check J_dot against a finite difference of J along the motion
q_t = rng.uniform(-0.5, 0.5, num_dof)
qd_t = rng.uniform(-2.0, 2.0, num_dof)
eps = 1e-6
Jdot_fd = (np.array(Jr_fn(*(q_t + eps * qd_t)), dtype=float)
           - np.array(Jr_fn(*(q_t - eps * qd_t)), dtype=float)) / (2 * eps)
Jdot_an = np.array(Jdot_fn(*q_t, *qd_t), dtype=float)
print("max |Jdot - finite difference| =", f"{np.abs(Jdot_an - Jdot_fd).max():.2e}")


# ------------------------------------------------------------------
# 15. Live simulation: feedforward + PD standing controller, with a push
# ------------------------------------------------------------------
USE_VIEWER = True                     # True opens the MuJoCo viewer window
T_END = 60.0                            # seconds
PUSH_N, PUSH_T0, PUSH_T1 = 40.0, 1.0, 1.1  # push force (N) on the torso, start/end time (s)

q_des = q_stand[3:].copy()
Kp = np.tile([200.0, 200.0, 150.0], 2)  # [hip, knee, ankle] per leg, Nm/rad
Kd = np.tile([8.0, 8.0, 15.0], 2)       # Nm*s/rad
ctrl_lo = model.actuator_ctrlrange[:, 0]
ctrl_hi = model.actuator_ctrlrange[:, 1]
torso_id = model.body("torso").id
z0_val = float(params[z0])

mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[1] -= pts_st[1::2].min()     # lower the robot so the soles start exactly on the floor
mujoco.mj_forward(model, data)
left_foot_x0 = data.body("left_foot").xpos[0]


def zmp_x():
    # centre of pressure from MuJoCo's contact forces (sign-independent)
    num = den = 0.0
    f6 = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f6)
        fw = data.contact[i].frame.reshape(3, 3).T @ f6[:3]
        num += data.contact[i].pos[0] * fw[2]
        den += fw[2]
    return num / den if abs(den) > 1e-6 else np.nan


log = {"t": [], "pitch": [], "height": [], "com_x": [], "zmp": [], "sat": []}


def sim_step():
    u = tau_ff + Kp * (q_des - data.qpos[3:]) - Kd * data.qvel[3:]
    data.ctrl[:] = np.clip(u, ctrl_lo, ctrl_hi)
    pushing = PUSH_T0 <= data.time < PUSH_T1
    data.xfrc_applied[torso_id, 0] = PUSH_N if pushing else 0.0
    mujoco.mj_step(model, data)

    log["t"].append(data.time)
    log["pitch"].append(data.qpos[2])
    log["height"].append(data.qpos[1] + z0_val)
    log["com_x"].append(data.subtree_com[0][0])
    log["zmp"].append(zmp_x())
    log["sat"].append(bool(np.any(np.abs(u) >= ctrl_hi - 1e-9)))


if USE_VIEWER:
    import time
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < T_END:
            t_wall = time.time()
            sim_step()
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t_wall)))
else:
    while data.time < T_END:
        sim_step()

t_log = np.array(log["t"])
pitch_deg = np.degrees(log["pitch"])
com_x = np.array(log["com_x"])
zmp = np.array(log["zmp"])
push = (t_log >= PUSH_T0) & (t_log < PUSH_T1 + 0.5)

print("\nStanding simulation summary")
print(f"simulated {data.time:.2f} s")
print(f"max |torso pitch| = {np.abs(pitch_deg).max():.2f} deg, final pitch = {pitch_deg[-1]:.3f} deg")
print(f"torso height: start {log['height'][0]:.4f} m, final {log['height'][-1]:.4f} m")
print(f"CoM x drift (final - start) = {com_x[-1] - com_x[0]:+.4f} m")
print(f"left foot slide = {data.body('left_foot').xpos[0] - left_foot_x0:+.4f} m")
print(f"ZMP x range during push: [{np.nanmin(zmp[push]):.4f}, {np.nanmax(zmp[push]):.4f}] m "
      f"(heel = {pts_st[0]:.4f}, toe = {pts_st[2]:.4f})")
print(f"fraction of steps with a saturated motor: {np.mean(log['sat']):.3f}")
fell = abs(pitch_deg[-1]) > 10 or log["height"][-1] < 0.5
print("RESULT:", "FELL" if fell else "stayed standing")
generate_validation_table