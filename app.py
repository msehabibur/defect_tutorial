import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="DFT Defect Formation Energy Tutorial",
    layout="wide"
)

# ============================================================
# Title
# ============================================================
st.title("Defect Formation Energy using DFT")
st.markdown(
    """
    **A realistic, research-grade tutorial with multiple charge states,
    defect levels, and animation.**

    This app follows the same methodology used in
    *semiconductor defect literature (CdTe, CdSeTe, CZTS, CIGS, perovskites)*.
    """
)

# ============================================================
# Section 1: Theory
# ============================================================
st.header("1️⃣ Theory")

st.latex(
    r"""
    E_f(D^q) =
    E_\text{defect}^q
    - E_\text{bulk}
    + \sum_i n_i \mu_i
    + q(E_F + E_\text{VBM})
    + E_\text{corr}
    """
)

st.markdown(
    """
    **Important physical insights**

    - Each **charge state** forms a straight line
    - The **slope** equals the charge state *q*
    - **Crossing points** between charge states are
      **charge transition levels (CTLs)**
    - CTLs define **defect levels inside the band gap**
    """
)

# ============================================================
# Section 2: Sidebar Inputs
# ============================================================
st.sidebar.header("🔧 DFT Inputs")

E_bulk = st.sidebar.number_input("Bulk total energy (eV)", value=-500.0)

st.sidebar.subheader("Defect total energies (per charge state)")
E_defect_q = {
    -2: st.sidebar.number_input("E_defect (q = -2)", value=-494.0),
    -1: st.sidebar.number_input("E_defect (q = -1)", value=-494.5),
     0: st.sidebar.number_input("E_defect (q =  0)", value=-495.0),
     1: st.sidebar.number_input("E_defect (q = +1)", value=-494.6),
     2: st.sidebar.number_input("E_defect (q = +2)", value=-494.2),
}

E_VBM = st.sidebar.number_input("VBM (eV)", value=0.0)
band_gap = st.sidebar.number_input("Band gap (eV)", value=1.5)

E_corr = st.sidebar.number_input("Correction energy E_corr (eV)", value=0.0)

st.sidebar.subheader("Chemical Potentials")
mu_A = st.sidebar.number_input("μ_A (eV)", value=-1.0)
mu_B = st.sidebar.number_input("μ_B (eV)", value=-2.0)

st.sidebar.subheader("Atoms Added / Removed")
n_A = st.sidebar.number_input("n_A", value=-1)
n_B = st.sidebar.number_input("n_B", value=0)

# ============================================================
# Core computation
# ============================================================
E_F = np.linspace(0, band_gap, 400)

def formation_energy(E_def, q, Ef):
    return (
        E_def
        - E_bulk
        + n_A * mu_A
        + n_B * mu_B
        + q * (Ef + E_VBM)
        + E_corr
    )

Ef_dict = {
    q: formation_energy(E_defect_q[q], q, E_F)
    for q in E_defect_q
}

# ============================================================
# Section 3: Multiple Charge States Plot
# ============================================================
st.header("3️⃣ Multiple Charge States (Realistic Plot)")

colors = {
    -2: "purple",
    -1: "blue",
     0: "black",
     1: "green",
     2: "orange",
}

fig, ax = plt.subplots()

for q in sorted(Ef_dict):
    ax.plot(
        E_F,
        Ef_dict[q],
        label=f"q = {q}",
        linewidth=2,
        color=colors[q]
    )

ax.set_xlim(0, band_gap)
ax.set_xlabel("Fermi Level (eV)")
ax.set_ylabel("Defect Formation Energy (eV)")
ax.set_title("Defect Formation Energy vs Fermi Level")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

# ============================================================
# Section 4: Charge Transition Levels (CTLs)
# ============================================================
st.header("4️⃣ Charge Transition Levels (Defect Levels)")

ctl_list = []

for q1 in Ef_dict:
    for q2 in Ef_dict:
        if q2 == q1 + 1:
            diff = Ef_dict[q1] - Ef_dict[q2]
            sign_change = np.where(np.diff(np.sign(diff)))[0]
            if len(sign_change) > 0:
                idx = sign_change[0]
                ctl_energy = E_F[idx]
                ctl_list.append((q1, q2, ctl_energy))

if ctl_list:
    for q1, q2, e in ctl_list:
        st.markdown(
            f"""
            **ε({q1}/{q2}) = {e:.2f} eV**

            ➜ Transition from charge **{q1} → {q2}**
            """
        )
else:
    st.warning("No CTLs found within the band gap.")

# ============================================================
# Section 5: Interactive Animation (Slider)
# ============================================================
st.header("5️⃣ Interactive Fermi-Level Animation")

Ef_slider = st.slider(
    "Move Fermi level (eV)",
    min_value=0.0,
    max_value=band_gap,
    value=0.0,
    step=0.01
)

fig, ax = plt.subplots()

for q in Ef_dict:
    ax.plot(E_F, Ef_dict[q], color=colors[q], alpha=0.4)

for q in Ef_dict:
    Ef_now = formation_energy(E_defect_q[q], q, Ef_slider)
    ax.scatter(Ef_slider, Ef_now, color=colors[q], s=60)

ax.axvline(Ef_slider, linestyle="--", color="red")
ax.set_xlim(0, band_gap)
ax.set_xlabel("Fermi Level (eV)")
ax.set_ylabel("Formation Energy (eV)")
ax.set_title(f"Fermi Level = {Ef_slider:.2f} eV")

st.pyplot(fig)
plt.close(fig)

# ============================================================
# Section 6: Teaching Notes (Very Important)
# ============================================================
st.header("6️⃣ Physical Interpretation (Why This Matters)")

st.markdown(
    """
    **How to read this plot like a researcher**

    - The **lowest line** at a given Fermi level is the
      **thermodynamically stable charge state**
    - CTLs indicate **defect levels inside the band gap**
    - Donor-like defects: CTL near CBM
    - Acceptor-like defects: CTL near VBM
    - Deep CTLs → recombination centers
    - Shallow CTLs → effective dopants

    This is exactly how defect tolerance and dopability
    are evaluated in real DFT studies.
    """
)

st.success("🎉 You now have a **realistic, publication-level defect formation energy tutorial**.")
