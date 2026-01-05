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
# DFT Workflow Explanation
# ============================================================
with st.expander("📚 How is Defect Formation Energy Obtained from DFT?", expanded=False):
    st.markdown("""
    ### Step-by-Step DFT Workflow for Defect Formation Energy
    
    #### 1️⃣ **Build the Bulk Supercell**
    - Create a large periodic supercell of your material (e.g., 3×3×3 unit cells)
    - Run DFT calculation to obtain total energy: **E_bulk**
    - Extract valence band maximum (VBM) from band structure
    
    #### 2️⃣ **Create the Defect Supercell**
    - Remove an atom (vacancy), add an atom (interstitial), or substitute an atom
    - For charged defects: add or remove electrons from the system
    - Run DFT calculation to obtain: **E_defect(q)**
    
    #### 3️⃣ **Calculate Chemical Potentials**
    - Compute elemental reference energies (e.g., bulk metals, gas molecules)
    - Chemical potential μ represents the energy cost to add/remove atoms
    - Range: from element-rich to element-poor conditions
    
    #### 4️⃣ **Apply Finite-Size Corrections**
    - Image charge interaction: charged defect interacts with periodic images
    - Potential alignment: correct for artificial potential shifts
    - Band filling: correct for electron removal from valence/conduction bands
    - Common schemes: FNV, Kumagai, Freysoldt corrections
    
    #### 5️⃣ **Compute Formation Energy**
    - Apply the formation energy formula (shown below)
    - Plot vs. Fermi level to identify stable charge states
    - Find charge transition levels (CTLs) where charge states cross
    
    #### 🔬 **Typical DFT Codes Used**
    - VASP, Quantum ESPRESSO, ABINIT, CASTEP, FHI-aims
    - Post-processing: PyCDT, PyMatGen, AIDE, eFermi
    """)

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
    with st.sidebar:
        st.header("⚙️ DFT Inputs (Basic)")
        
        st.subheader("Total Energies")
        E_bulk = st.number_input("Bulk total energy (eV)", value=-500.0, key="bulk1")
        E_defect = st.number_input("Defect total energy (eV)", value=-495.0, key="defect1")
        
        st.subheader("Charge State")
        q = st.selectbox("Charge state (q)", [-2, -1, 0, 1, 2], index=2, key="q1")

        st.subheader("Band Structure")
        E_VBM = st.number_input("VBM (eV)", value=0.0, key="vbm1")
        band_gap = st.number_input("Band gap (eV)", value=1.5, key="gap1")
        
        st.subheader("Corrections")
        E_corr = st.number_input("Correction energy (eV)", value=0.0, key="corr1")

        st.subheader("Chemical Potentials")
        mu_A = st.number_input("μ_A (eV)", value=-1.0, key="muA1")
        mu_B = st.number_input("μ_B (eV)", value=-2.0, key="muB1")

        st.subheader("Atoms Added / Removed")
        n_A = st.number_input("n_A", value=-1, key="nA1")
        n_B = st.number_input("n_B", value=0, key="nB1")
        
        st.divider()
        animate = st.checkbox("▶️ Auto-animate Fermi level", value=False, key="anim1")

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

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Formation Energy vs Fermi Level")
        # Static plot - smaller size
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(E_F, E_form, linewidth=2, color='steelblue')
        ax.set_xlabel("Fermi Level (eV)", fontsize=10)
        ax.set_ylabel("Defect Formation Energy (eV)", fontsize=10)
        ax.set_title(f"Charge state q = {q}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"""
            **Key insight**  
            The slope of the line equals the charge state **q = {q}**.
            
            Slope = dE_f/dE_F = q
            """
        )

    with col2:
        st.subheader("Interactive Fermi-Level Explorer")

        if not animate:
            Ef_slider = st.slider(
                "Move Fermi Level (eV)",
                0.0, band_gap, band_gap/2, 0.01, key="slider1"
            )
        else:
            # Animation placeholder
            animation_placeholder = st.empty()
            
            for step in range(100):
                Ef_slider = (step / 100) * band_gap
                
                Ef_current = (
                    E_defect
                    - E_bulk
                    + n_A * mu_A
                    + n_B * mu_B
                    + q * (Ef_slider + E_VBM)
                    + E_corr
                )

                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(E_F, E_form, alpha=0.4, linewidth=2, color='steelblue')
                ax.scatter(Ef_slider, Ef_current, color="red", s=100, zorder=5)
                ax.axvline(Ef_slider, linestyle="--", color="red", alpha=0.6)
                ax.set_xlabel("Fermi Level (eV)", fontsize=10)
                ax.set_ylabel("Formation Energy (eV)", fontsize=10)
                ax.set_title(f"E_F = {Ef_slider:.3f} eV, E_form = {Ef_current:.3f} eV", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
                plt.tight_layout()
                
                animation_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.05)
            
            st.rerun()

        if not animate:
            Ef_current = (
                E_defect
                - E_bulk
                + n_A * mu_A
                + n_B * mu_B
                + q * (Ef_slider + E_VBM)
                + E_corr
            )

            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(E_F, E_form, alpha=0.4, linewidth=2, color='steelblue')
            ax.scatter(Ef_slider, Ef_current, color="red", s=100, zorder=5)
            ax.axvline(Ef_slider, linestyle="--", color="red", alpha=0.6)
            ax.set_xlabel("Fermi Level (eV)", fontsize=10)
            ax.set_ylabel("Formation Energy (eV)", fontsize=10)
            ax.set_title(f"E_F = {Ef_slider:.3f} eV, E_form = {Ef_current:.3f} eV", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("Formation Energy at E_F", f"{Ef_current:.3f} eV")

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

    # Sidebar inputs (advanced)
    with st.sidebar:
        st.header("⚙️ DFT Inputs (Advanced)")
        
        E_bulk2 = st.number_input("Bulk total energy (eV)", value=-500.0, key="bulk2")

        st.subheader("Defect energies by charge state")
        E_defect_q = {
            -2: st.number_input("E_defect (q = -2)", value=-494.0, key="e_2"),
            -1: st.number_input("E_defect (q = -1)", value=-494.5, key="e_1"),
             0: st.number_input("E_defect (q =  0)", value=-495.0, key="e0"),
             1: st.number_input("E_defect (q = +1)", value=-494.6, key="e1"),
             2: st.number_input("E_defect (q = +2)", value=-494.2, key="e2"),
        }

        E_VBM2 = st.number_input("VBM (eV)", value=0.0, key="vbm2")
        band_gap2 = st.number_input("Band gap (eV)", value=1.5, key="gap2")

        E_corr2 = st.number_input("Correction energy (eV)", value=0.0, key="corr2")

        st.subheader("Chemical Potentials")
        mu_A2 = st.number_input("μ_A (eV)", value=-1.0, key="muA2")
        mu_B2 = st.number_input("μ_B (eV)", value=-2.0, key="muB2")

        st.subheader("Atoms Added / Removed")
        n_A2 = st.number_input("n_A", value=-1, key="nA2")
        n_B2 = st.number_input("n_B", value=0, key="nB2")
        
        st.divider()
        animate2 = st.checkbox("▶️ Auto-animate Fermi level", value=False, key="anim2")

    # Computation
    E_F2 = np.linspace(0, band_gap2, 400)

    def formation_energy(E_def, q, Ef):
        return (
            E_def - E_bulk2
            + n_A2 * mu_A2 + n_B2 * mu_B2
            + q * (Ef + E_VBM2)
            + E_corr2
        )

    Ef_dict = {q: formation_energy(E_defect_q[q], q, E_F2) for q in E_defect_q}

    # Find minimum envelope
    E_min = np.minimum.reduce([Ef_dict[q] for q in Ef_dict])

    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Formation Energy Diagram")
        
        # Plot
        colors = {-2: "purple", -1: "blue", 0: "black", 1: "green", 2: "orange"}

        fig, ax = plt.subplots(figsize=(6, 4))
        for q in sorted(Ef_dict):
            ax.plot(E_F2, Ef_dict[q], label=f"q = {q:+d}", color=colors[q], linewidth=2, alpha=0.7)
        
        # Highlight stable regions
        ax.plot(E_F2, E_min, 'k-', linewidth=3, label='Stable (lowest)', zorder=10)

        ax.set_xlim(0, band_gap2)
        ax.set_xlabel("Fermi Level (eV)", fontsize=10)
        ax.set_ylabel("Defect Formation Energy (eV)", fontsize=10)
        ax.set_title("Multiple Charge States", fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # CTL detection
        st.subheader("Charge Transition Levels")

        ctls_found = []
        for q in sorted(Ef_dict):
            if q + 1 in Ef_dict:
                diff = Ef_dict[q] - Ef_dict[q + 1]
                idx = np.where(np.diff(np.sign(diff)))[0]
                if len(idx) > 0:
                    ctl = E_F2[idx[0]]
                    ctls_found.append((q, q+1, ctl))
                    st.markdown(f"**ε({q:+d}/{q+1:+d}) = {ctl:.3f} eV**")

        if not ctls_found:
            st.warning("No CTLs found within the band gap.")
        
        st.divider()
        
        st.success(
            """
            **Physical Interpretation**
            
            - **Deep levels** (mid-gap): Recombination/trapping centers
            - **Shallow levels** (near VBM/CBM): Effective dopants
            - **Acceptor**: (+/0) level near VBM
            - **Donor**: (0/−) level near CBM
            """
        )
    
    # Interactive slider for advanced tab
    st.subheader("🎯 Interactive Fermi Level Analysis")
    
    if not animate2:
        Ef_slider2 = st.slider(
            "Explore Fermi Level Position (eV)",
            0.0, band_gap2, band_gap2/2, 0.01, key="slider2"
        )
    else:
        animation_placeholder2 = st.empty()
        
        for step in range(100):
            Ef_slider2 = (step / 100) * band_gap2
            
            # Find stable charge state at this E_F
            energies_at_ef = {q: formation_energy(E_defect_q[q], q, Ef_slider2) for q in E_defect_q}
            stable_q = min(energies_at_ef, key=energies_at_ef.get)
            stable_E = energies_at_ef[stable_q]
            
            fig, ax = plt.subplots(figsize=(7, 4))
            for q in sorted(Ef_dict):
                ax.plot(E_F2, Ef_dict[q], color=colors[q], linewidth=1.5, alpha=0.4)
            
            ax.plot(E_F2, E_min, 'k-', linewidth=2.5, alpha=0.3)
            ax.axvline(Ef_slider2, linestyle="--", color="red", linewidth=2, alpha=0.7)
            ax.scatter(Ef_slider2, stable_E, color="red", s=150, zorder=10, edgecolors='black', linewidth=2)
            
            ax.set_xlim(0, band_gap2)
            ax.set_xlabel("Fermi Level (eV)", fontsize=10)
            ax.set_ylabel("Formation Energy (eV)", fontsize=10)
            ax.set_title(f"E_F = {Ef_slider2:.3f} eV | Stable: q = {stable_q:+d} | E_form = {stable_E:.3f} eV", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            
            animation_placeholder2.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05)
        
        st.rerun()
    
    if not animate2:
        # Find stable charge state at this E_F
        energies_at_ef = {q: formation_energy(E_defect_q[q], q, Ef_slider2) for q in E_defect_q}
        stable_q = min(energies_at_ef, key=energies_at_ef.get)
        stable_E = energies_at_ef[stable_q]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Fermi Level", f"{Ef_slider2:.3f} eV")
        col2.metric("Stable Charge State", f"q = {stable_q:+d}")
        col3.metric("Formation Energy", f"{stable_E:.3f} eV")
        
        fig, ax = plt.subplots(figsize=(7, 4))
        for q in sorted(Ef_dict):
            ax.plot(E_F2, Ef_dict[q], color=colors[q], linewidth=2, alpha=0.4, label=f"q = {q:+d}")
        
        ax.plot(E_F2, E_min, 'k-', linewidth=3, alpha=0.3, label='Stable')
        ax.axvline(Ef_slider2, linestyle="--", color="red", linewidth=2, alpha=0.7)
        ax.scatter(Ef_slider2, stable_E, color="red", s=150, zorder=10, edgecolors='black', linewidth=2)
        
        ax.set_xlim(0, band_gap2)
        ax.set_xlabel("Fermi Level (eV)", fontsize=10)
        ax.set_ylabel("Formation Energy (eV)", fontsize=10)
        ax.set_title(f"Stable Charge State: q = {stable_q:+d}", fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
