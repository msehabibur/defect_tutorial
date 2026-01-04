import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# Page config
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
    **An interactive tutorial with animation**

    This app explains **what defect formation energy is**,  
    **how it is computed using DFT**,  
    and **why it depends on the Fermi level**.
    """
)

# ============================================================
# Section 1: Theory
# ============================================================
st.header("1️⃣ Theory: Defect Formation Energy")

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
    **Meaning of terms**

    - **E_defect**: Total energy of defective supercell (DFT)
    - **E_bulk**: Total energy of pristine supercell (DFT)
    - **nᵢ**: Number of atoms added (+) or removed (−)
    - **μᵢ**: Chemical potential of element *i*
    - **q**: Defect charge state
    - **E_F**: Fermi level (from VBM → CBM)
    - **E_VBM**: Valence band maximum
    - **E_corr**: Finite-size / charge correction (FNV, etc.)
    """
)

# ============================================================
# Section 2: Workflow
# ============================================================
st.header("2️⃣ DFT Workflow")

st.markdown(
    """
    **Step 1 — Bulk calculation**
    - Relax pristine supercell
    - Extract total energy and VBM

    **Step 2 — Defect calculation**
    - Create vacancy / interstitial / substitution
    - Relax structure
    - Compute total energy for each charge state

    **Step 3 — Chemical potentials**
    - Choose growth conditions (rich / poor)

    **Step 4 — Scan Fermi level**
    - From VBM to CBM
    - Plot formation energy
    """
)

# ============================================================
# Sidebar Inputs
# ============================================================
st.sidebar.header("🔧 Input Parameters")

E_bulk = st.sidebar.number_input("Bulk total energy (eV)", value=-500.0)
E_defect = st.sidebar.number_input("Defect total energy (eV)", value=-495.0)

q = st.sidebar.selectbox("Defect charge state (q)", [-2, -1, 0, 1, 2])

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
# Core calculation
# ============================================================
E_F = np.linspace(0, band_gap, 300)

E_form = (
    E_defect
    - E_bulk
    + n_A * mu_A
    + n_B * mu_B
    + q * (E_F + E_VBM)
    + E_corr
)

# ============================================================
# Section 3: Static Plot
# ============================================================
st.header("3️⃣ Static Defect Formation Energy Plot")

fig, ax = plt.subplots()
ax.plot(E_F, E_form, linewidth=2)
ax.set_xlabel("Fermi Level (eV)")
ax.set_ylabel("Defect Formation Energy (eV)")
ax.set_title("Defect Formation Energy vs Fermi Level")
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

# ============================================================
# Section 4: Interactive Slider Animation
# ============================================================
st.header("4️⃣ Interactive Animation (Recommended for Teaching)")

Ef_slider = st.slider(
    "Move Fermi Level (eV)",
    min_value=0.0,
    max_value=band_gap,
    value=0.0,
    step=0.01
)

Ef_current = (
    E_defect
    - E_bulk
    + n_A * mu_A
    + n_B * mu_B
    + q * (Ef_slider + E_VBM)
    + E_corr
)

fig, ax = plt.subplots()
ax.plot(E_F, E_form, label="Formation Energy")
ax.scatter(Ef_slider, Ef_current, color="red", s=100, zorder=3)
ax.axvline(Ef_slider, linestyle="--", color="red")
ax.set_xlabel("Fermi Level (eV)")
ax.set_ylabel("Formation Energy (eV)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
plt.close(fig)

st.info(
    "🎓 This slider shows **why defect stability depends on the Fermi level**."
)

# ============================================================
# Section 5: Auto Animation
# ============================================================
st.header("5️⃣ Automatic Animation")

animate = st.checkbox("▶ Play Fermi-level animation")

placeholder = st.empty()

if animate:
    for i in range(len(E_F)):
        fig, ax = plt.subplots()
        ax.plot(E_F, E_form, color="gray", alpha=0.4)
        ax.scatter(E_F[i], E_form[i], color="red", s=80)
        ax.axvline(E_F[i], linestyle="--", color="red")

        ax.set_xlim(0, band_gap)
        ax.set_ylim(min(E_form) - 0.5, max(E_form) + 0.5)
        ax.set_xlabel("Fermi Level (eV)")
        ax.set_ylabel("Formation Energy (eV)")
        ax.set_title(f"Fermi Level = {E_F[i]:.2f} eV")

        placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.03)

# ============================================================
# Section 6: Teaching Notes
# ============================================================
st.header("6️⃣ Key Teaching Takeaways")

st.markdown(
    """
    ✔ Charged defects depend linearly on Fermi level  
    ✔ Slopes = charge state **q**  
    ✔ Line crossings = **charge transition levels (CTLs)**  
    ✔ Growth conditions change chemical potentials  
    ✔ Corrections are essential for charged defects  

    This framework applies to:
    - CdTe, CdSeTe, CZTS, CIGS
    - Vacancies, interstitials, substitutionals
    """
)

st.success("🎉 You now understand and can *visualize* defect formation energy using DFT!")
