import mujoco
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)


def get_mujoco_dynamics(q_sample):
    # Set configuration
    data.qpos[:] = q_sample
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    # Compute forward kinematics & dynamics terms
    mujoco.mj_forward(model, data)

    # 1. Forward Kinematics (e.g., position of foot or site)
    foot_pos = data.site("left_foot_site").xpos.copy()

    # 2. Dense Mass Matrix M(q)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    # 3. Gravity Vector g(q)
    # qfrc_bias computes C(q, qdot)*qdot + g(q); with qdot=0, it gives purely g(q)
    g = data.qfrc_bias.copy()

    return foot_pos, M, g

import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

# 1. Time and physical constants
t = sp.Symbol('t')
g = sp.Symbol('g')

# Masses, lengths, CoM offsets, link inertias, and gear armatures
# (Match these EXACTLY to your robot's .xml file values)
m_torso, m_thigh, m_shin = sp.symbols('m_torso m_thigh m_shin')
l_thigh, l_shin = sp.symbols('l_thigh l_shin')
I_torso, I_thigh, I_shin = sp.symbols('I_torso I_thigh I_shin')
armature_hip, armature_knee = sp.symbols('armature_hip armature_knee')

# Extract M(q) as a 7x7 matrix
num_dof = len(q)
M_sym = sp.zeros(num_dof, num_dof)
for i in range(num_dof):
    for j in range(num_dof):
        M_sym[i, j] = sp.diff(sp.diff(T, q_dot[i]), q_dot[j])

# Extract g(q) as a 7x1 vector
g_sym = sp.zeros(num_dof, 1)
for i in range(num_dof):
    g_sym[i, 0] = sp.diff(V, q[i])
