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

st.title("Defect Formation Energy using DFT")
st.markdown(
    """
    **Interactive tutorial for beginners and researchers**

    This app explains how defect formation energies are computed
    using Density Functional Theory (DFT), with increasing realism.
    """
)

# ============================================================
# Tabs
# ============================================================
tab_basic, tab_advanced = st.tabs(
    ["🟦 Single Charge State (Basics)", "🟩 Multiple Charge States (Research)"]
)

# ============================================================
# ---------------- TAB 1: BASIC ------------------------------
# ============================================================
with tab_basic:

    st.header("Single Charge State Defect Formation Energy")

    st.latex(
        r"""
        E_f(D^q) =
        E_\text{defect}
        - E_\text{bulk}
        + \sum_i n_i \mu_i
        + q(E_F + E_\text{VBM})
        + E_\text{corr}
        """
    )

    st.markdown(
        """
        **Goal of this tab**

        - Understand how **one defect charge state** behaves
        - Learn why formation energy depends on the Fermi level
        - Build intuition before moving to realistic plots
        """
    )

    # Sidebar inputs (basic)
    st.subheader("DFT Inputs")

    E_bulk = st.number_input("Bulk total energy (eV)", value=-500.0)
    E_defect = st.number_input("Defect total energy (eV)", value=-495.0)
    q = st.selectbox("Charge state (q)", [-2, -1, 0, 1, 2])

    E_VBM = st.number_input("VBM (eV)", value=0.0)
    band_gap = st.number_input("Band gap (eV)", value=1.5)
    E_corr = st.number_input("Correction energy (eV)", value=0.0)

    st.subheader("Chemical Potentials")
    mu_A = st.number_input("μ_A (eV)", value=-1.0)
    mu_B = st.number_input("μ_B (eV)", value=-2.0)

    st.subheader("Atoms Added / Removed")
    n_A = st.number_input("n_A", value=-1)
    n_B = st.number_input("n_B", value=0)

    # Calculation
    E_F = np.linspace(0, band_gap, 300)

    E_form = (
        E_defect
        - E_bulk
        + n_A * mu_A
        + n_B * mu_B
        + q * (E_F + E_VBM)
        + E_corr
    )

    # Static plot
    fig, ax = plt.subplots()
    ax.plot(E_F, E_form, linewidth=2)
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Defect Formation Energy (eV)")
    ax.set_title(f"Charge state q = {q}")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    # Slider animation
    st.subheader("Interactive Fermi-Level Slider")

    Ef_slider = st.slider(
        "Move Fermi Level (eV)",
        0.0, band_gap, 0.0, 0.01
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
    ax.plot(E_F, E_form, alpha=0.4)
    ax.scatter(Ef_slider, Ef_current, color="red", s=100)
    ax.axvline(Ef_slider, linestyle="--", color="red")
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Defect Formation Energy (eV)")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        **Key insight**  
        The slope of the line equals the charge state **q**.
        """
    )

# ============================================================
# ---------------- TAB 2: ADVANCED ----------------------------
# ============================================================
with tab_advanced:

    st.header("Multiple Charge States & Defect Levels")

    st.markdown(
        """
        **This tab reflects how defect formation energies are analyzed
        in real DFT studies.**
        """
    )

    # Inputs
    E_bulk = st.number_input("Bulk total energy (eV)", value=-500.0, key="bulk2")

    st.subheader("Defect energies by charge state")
    E_defect_q = {
        -2: st.number_input("E_defect (q = -2)", value=-494.0),
        -1: st.number_input("E_defect (q = -1)", value=-494.5),
         0: st.number_input("E_defect (q =  0)", value=-495.0),
         1: st.number_input("E_defect (q = +1)", value=-494.6),
         2: st.number_input("E_defect (q = +2)", value=-494.2),
    }

    E_VBM = st.number_input("VBM (eV)", value=0.0, key="vbm2")
    band_gap = st.number_input("Band gap (eV)", value=1.5, key="gap2")

    E_corr = st.number_input("Correction energy (eV)", value=0.0, key="corr2")

    st.subheader("Chemical Potentials")
    mu_A = st.number_input("μ_A (eV)", value=-1.0, key="muA2")
    mu_B = st.number_input("μ_B (eV)", value=-2.0, key="muB2")

    st.subheader("Atoms Added / Removed")
    n_A = st.number_input("n_A", value=-1, key="nA2")
    n_B = st.number_input("n_B", value=0, key="nB2")

    # Computation
    E_F = np.linspace(0, band_gap, 400)

    def formation_energy(E_def, q, Ef):
        return (
            E_def - E_bulk
            + n_A * mu_A + n_B * mu_B
            + q * (Ef + E_VBM)
            + E_corr
        )

    Ef_dict = {q: formation_energy(E_defect_q[q], q, E_F) for q in E_defect_q}

    # Plot
    colors = {-2: "purple", -1: "blue", 0: "black", 1: "green", 2: "orange"}

    fig, ax = plt.subplots()
    for q in sorted(Ef_dict):
        ax.plot(E_F, Ef_dict[q], label=f"q = {q}", color=colors[q], linewidth=2)

    ax.set_xlim(0, band_gap)
    ax.set_xlabel("Fermi Level (eV)")
    ax.set_ylabel("Defect Formation Energy (eV)")
    ax.set_title("Multiple Charge States")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    # CTL detection
    st.subheader("Charge Transition Levels (CTLs)")

    found = False
    for q in sorted(Ef_dict):
        if q + 1 in Ef_dict:
            diff = Ef_dict[q] - Ef_dict[q + 1]
            idx = np.where(np.diff(np.sign(diff)))[0]
            if len(idx) > 0:
                found = True
                ctl = E_F[idx[0]]
                st.markdown(f"**ε({q}/{q+1}) = {ctl:.2f} eV**")

    if not found:
        st.warning("No CTLs found within the band gap.")

    st.success(
        """
        **How researchers use this plot**

        - Lowest line → stable charge state
        - Line crossings → defect levels
        - Deep levels → recombination centers
        - Shallow levels → good dopants
        """
    )
