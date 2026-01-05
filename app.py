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
tab_basic, tab_advanced, tab_atomistic = st.tabs(
    ["🟦 Single Charge State (Basics)", "🟩 Multiple Charge States (Research)", "⚛️ Atomistic Relaxation"]
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

# ============================================================
# ---------------- TAB 3: ATOMISTIC RELAXATION ---------------
# ============================================================
with tab_atomistic:
    st.header("⚛️ Atomistic View: DFT Defect Relaxation")
    
    st.markdown("""
    **Understand how atoms relax around a defect during DFT geometry optimization**
    
    This visualization shows:
    - Initial unrelaxed atomic positions after defect creation
    - Iterative atomic relaxation driven by forces
    - Final relaxed structure with minimized energy
    """)
    
    with st.expander("🔬 DFT Relaxation Process", expanded=False):
        st.markdown("""
        ### How DFT Relaxes Defect Structures
        
        1. **Create Defect**: Remove/add/substitute atoms in supercell
        2. **Calculate Forces**: DFT computes Hellmann-Feynman forces on each atom
        3. **Move Atoms**: Move atoms along force directions to minimize energy
        4. **Iterate**: Repeat until forces < threshold (typically 0.01-0.05 eV/Å)
        5. **Converged**: Final structure with minimum total energy
        
        **Relaxation Algorithms**: 
        - Conjugate Gradient (CG)
        - Quasi-Newton (BFGS, L-BFGS)
        - FIRE (Fast Inertial Relaxation Engine)
        - Damped Molecular Dynamics
        
        **Typical Convergence**: 10-50 ionic steps for point defects
        """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚛️ Atomistic Visualization")
        
        defect_type = st.selectbox(
            "Defect Type",
            ["Vacancy", "Interstitial", "Substitution"],
            key="defect_type"
        )
        
        lattice_size = st.slider("Supercell Size (atoms per side)", 3, 7, 5, key="lattice")
        
        relaxation_strength = st.slider(
            "Relaxation Magnitude", 
            0.1, 1.0, 0.5, 0.1,
            help="How much atoms move around defect",
            key="relax_strength"
        )
        
        show_forces = st.checkbox("Show Force Vectors", value=True, key="forces")
        show_energy = st.checkbox("Show Energy Convergence", value=True, key="energy_conv")
        
        st.divider()
        play_animation = st.checkbox("▶️ Play Relaxation Animation", value=False, key="play_relax")
        
        if not play_animation:
            relaxation_step = st.slider(
                "Relaxation Step",
                0, 20, 0, 1,
                key="relax_step"
            )
    
    # Generate atomic structure
    def generate_lattice(size, defect_type, step, max_steps=20, strength=0.5):
        """Generate atomic positions for lattice with defect"""
        np.random.seed(42)
        
        # Create perfect lattice
        positions = []
        atom_types = []
        
        center = size // 2
        defect_pos = np.array([center, center, center])
        
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    pos = np.array([i, j, k], dtype=float)
                    
                    # Skip center for vacancy
                    if defect_type == "Vacancy" and np.allclose(pos, defect_pos):
                        continue
                    
                    positions.append(pos)
                    
                    # Mark substitution
                    if defect_type == "Substitution" and np.allclose(pos, defect_pos):
                        atom_types.append("impurity")
                    else:
                        atom_types.append("host")
        
        # Add interstitial
        if defect_type == "Interstitial":
            positions.append(defect_pos + np.array([0.5, 0.5, 0.5]))
            atom_types.append("impurity")
        
        positions = np.array(positions)
        
        # Apply relaxation displacement
        progress = step / max_steps
        damping = np.exp(-3 * progress)  # Exponential damping
        
        relaxed_positions = positions.copy()
        forces = np.zeros_like(positions)
        
        for i, pos in enumerate(positions):
            r = pos - (defect_pos + (0.5 if defect_type == "Interstitial" else 0))
            dist = np.linalg.norm(r)
            
            if dist > 0.1:
                # Displacement decreases with distance
                displacement_mag = strength * np.exp(-dist / 2) * (1 - progress)
                direction = r / dist
                
                # Inward for vacancy, outward for interstitial
                if defect_type == "Vacancy":
                    displacement = -displacement_mag * direction
                elif defect_type == "Interstitial":
                    displacement = displacement_mag * direction
                else:  # Substitution
                    displacement = 0.3 * displacement_mag * direction
                
                relaxed_positions[i] += displacement
                forces[i] = -displacement * damping * 5  # Force proportional to displacement
        
        return relaxed_positions, atom_types, forces, progress
    
    def calculate_energy(step, max_steps=20):
        """Simulate energy convergence"""
        progress = step / max_steps
        # Exponential decay to minimum
        E_initial = 0.0
        E_final = -2.5
        energy = E_initial + (E_final - E_initial) * (1 - np.exp(-3 * progress))
        return energy
    
    # Main visualization
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader(f"3D Structure: {defect_type} Defect")
        
        if play_animation:
            structure_placeholder = st.empty()
            
            for step in range(21):
                positions, types, forces, progress = generate_lattice(
                    lattice_size, defect_type, step, strength=relaxation_strength
                )
                
                # Create 3D plot
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(7, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot atoms
                host_atoms = positions[np.array(types) == "host"]
                impurity_atoms = positions[np.array(types) == "impurity"]
                
                if len(host_atoms) > 0:
                    ax.scatter(host_atoms[:, 0], host_atoms[:, 1], host_atoms[:, 2],
                             c='steelblue', s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                             label='Host atoms')
                
                if len(impurity_atoms) > 0:
                    ax.scatter(impurity_atoms[:, 0], impurity_atoms[:, 1], impurity_atoms[:, 2],
                             c='orange', s=150, alpha=0.9, edgecolors='black', linewidth=1,
                             label='Impurity/Interstitial')
                
                # Mark defect location for vacancy
                if defect_type == "Vacancy":
                    center = lattice_size // 2
                    ax.scatter([center], [center], [center],
                             c='red', s=200, alpha=0.3, marker='x', linewidths=3,
                             label='Vacancy site')
                
                # Show forces
                if show_forces and step < 15:
                    force_scale = 0.5
                    for i, (pos, force) in enumerate(zip(positions, forces)):
                        if np.linalg.norm(force) > 0.01:
                            ax.quiver(pos[0], pos[1], pos[2],
                                    force[0], force[1], force[2],
                                    color='red', alpha=0.6, arrow_length_ratio=0.3,
                                    linewidth=1.5, length=force_scale)
                
                ax.set_xlabel('X (Å)', fontsize=9)
                ax.set_ylabel('Y (Å)', fontsize=9)
                ax.set_zlabel('Z (Å)', fontsize=9)
                ax.set_title(f'Relaxation Step {step}/20 | Progress: {progress*100:.0f}%', fontsize=11)
                ax.legend(fontsize=8, loc='upper right')
                ax.set_box_aspect([1,1,1])
                
                plt.tight_layout()
                structure_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Update energy plot during animation
                steps_arr = np.arange(21)
                energies = [calculate_energy(s) for s in steps_arr]
                
                if show_energy:
                    fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
                    ax_e.plot(steps_arr[:step+1], energies[:step+1], 'o-', color='steelblue', linewidth=2, markersize=4)
                    ax_e.plot(steps_arr[step:], energies[step:], 'o-', color='lightblue', linewidth=1, markersize=3, alpha=0.3)
                    ax_e.scatter([step], [energies[step]], 
                             color='red', s=100, zorder=5, edgecolors='black', linewidth=2)
                    ax_e.axvline(step, linestyle='--', color='red', alpha=0.5)
                    
                    ax_e.set_xlabel('Ionic Step', fontsize=9)
                    ax_e.set_ylabel('Total Energy (eV)', fontsize=9)
                    ax_e.set_title('DFT Energy Minimization', fontsize=10)
                    ax_e.grid(True, alpha=0.3)
                    ax_e.tick_params(labelsize=8)
                    plt.tight_layout()
                    energy_placeholder.pyplot(fig_e)
                    plt.close(fig_e)
                    
                    # Update metrics
                    current_energy = energies[step]
                    if step > 0:
                        energy_change = energies[step] - energies[step-1]
                        energy_metric_placeholder.markdown(f"**Current Energy:** {current_energy:.3f} eV | **Change:** {energy_change:.4f} eV")
                    else:
                        energy_metric_placeholder.markdown(f"**Current Energy:** {current_energy:.3f} eV")
                else:
                    energy_placeholder.empty()
                    energy_metric_placeholder.empty()
                
                # Update force plot during animation
                max_forces = []
                for s in range(21):
                    _, _, f_temp, _ = generate_lattice(lattice_size, defect_type, s, strength=relaxation_strength)
                    max_force = np.max([np.linalg.norm(f) for f in f_temp]) if len(f_temp) > 0 else 0
                    max_forces.append(max_force)
                
                fig_f, ax_f = plt.subplots(figsize=(5, 3.5))
                ax_f.plot(steps_arr[:step+1], max_forces[:step+1], 's-', color='green', linewidth=2, markersize=4)
                ax_f.plot(steps_arr[step:], max_forces[step:], 's-', color='lightgreen', linewidth=1, markersize=3, alpha=0.3)
                ax_f.axhline(0.05, linestyle='--', color='red', label='Convergence criterion', linewidth=1.5)
                ax_f.scatter([step], [max_forces[step]], 
                         color='red', s=100, zorder=5, edgecolors='black', linewidth=2)
                ax_f.axvline(step, linestyle='--', color='red', alpha=0.5)
                
                ax_f.set_xlabel('Ionic Step', fontsize=9)
                ax_f.set_ylabel('Max Force (eV/Å)', fontsize=9)
                ax_f.set_title('Force Convergence', fontsize=10)
                ax_f.legend(fontsize=8)
                ax_f.grid(True, alpha=0.3)
                ax_f.tick_params(labelsize=8)
                plt.tight_layout()
                force_placeholder.pyplot(fig_f)
                plt.close(fig_f)
                
                # Update force metrics and status
                current_force = max_forces[step]
                force_metric_placeholder.markdown(f"**Max Force:** {current_force:.4f} eV/Å")
                
                if current_force < 0.05:
                    status_placeholder.success("✅ Structure Converged!")
                else:
                    status_placeholder.warning("⚠️ Still Relaxing...")
                
                time.sleep(0.15)
            
            st.rerun()
        
        else:
            positions, types, forces, progress = generate_lattice(
                lattice_size, defect_type, relaxation_step, strength=relaxation_strength
            )
            
            # Create 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot atoms
            host_atoms = positions[np.array(types) == "host"]
            impurity_atoms = positions[np.array(types) == "impurity"]
            
            if len(host_atoms) > 0:
                ax.scatter(host_atoms[:, 0], host_atoms[:, 1], host_atoms[:, 2],
                         c='steelblue', s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                         label='Host atoms')
            
            if len(impurity_atoms) > 0:
                ax.scatter(impurity_atoms[:, 0], impurity_atoms[:, 1], impurity_atoms[:, 2],
                         c='orange', s=150, alpha=0.9, edgecolors='black', linewidth=1,
                         label='Impurity/Interstitial')
            
            # Mark defect location for vacancy
            if defect_type == "Vacancy":
                center = lattice_size // 2
                ax.scatter([center], [center], [center],
                         c='red', s=200, alpha=0.3, marker='x', linewidths=3,
                         label='Vacancy site')
            
            # Show forces
            if show_forces and relaxation_step < 15:
                force_scale = 0.5
                for i, (pos, force) in enumerate(zip(positions, forces)):
                    if np.linalg.norm(force) > 0.01:
                        ax.quiver(pos[0], pos[1], pos[2],
                                force[0], force[1], force[2],
                                color='red', alpha=0.6, arrow_length_ratio=0.3,
                                linewidth=1.5, length=force_scale)
            
            ax.set_xlabel('X (Å)', fontsize=9)
            ax.set_ylabel('Y (Å)', fontsize=9)
            ax.set_zlabel('Z (Å)', fontsize=9)
            ax.set_title(f'Relaxation Step {relaxation_step}/20', fontsize=11)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_box_aspect([1,1,1])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    with col2:
        if play_animation:
            # Create placeholders for animated plots
            st.subheader("Energy Convergence")
            energy_placeholder = st.empty()
            energy_metric_placeholder = st.empty()
            
            st.subheader("Max Force on Atoms")
            force_placeholder = st.empty()
            force_metric_placeholder = st.empty()
            status_placeholder = st.empty()
        
        # Energy convergence
        if show_energy:
            if not play_animation:
                st.subheader("Energy Convergence")
            
            steps = np.arange(21)
            energies = [calculate_energy(s) for s in steps]
            
            if play_animation:
                current_step = 20
            else:
                current_step = relaxation_step
            
            if not play_animation:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(steps, energies, 'o-', color='steelblue', linewidth=2, markersize=4)
                
                ax.scatter([current_step], [energies[current_step]], 
                         color='red', s=100, zorder=5, edgecolors='black', linewidth=2)
                ax.axvline(current_step, linestyle='--', color='red', alpha=0.5)
                
                ax.set_xlabel('Ionic Step', fontsize=9)
                ax.set_ylabel('Total Energy (eV)', fontsize=9)
                ax.set_title('DFT Energy Minimization', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Current metrics
                current_energy = energies[current_step]
                st.metric("Current Energy", f"{current_energy:.3f} eV")
                
                if current_step > 0:
                    energy_change = energies[current_step] - energies[current_step-1]
                    st.metric("Energy Change", f"{energy_change:.4f} eV")
        
        # Force convergence
        if not play_animation:
            st.subheader("Max Force on Atoms")
        
        max_forces = []
        for s in range(21):
            _, _, forces, _ = generate_lattice(lattice_size, defect_type, s, strength=relaxation_strength)
            max_force = np.max([np.linalg.norm(f) for f in forces]) if len(forces) > 0 else 0
            max_forces.append(max_force)
        
        if not play_animation:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(steps, max_forces, 's-', color='green', linewidth=2, markersize=4)
            ax.axhline(0.05, linestyle='--', color='red', label='Convergence criterion', linewidth=1.5)
            
            ax.scatter([current_step], [max_forces[current_step]], 
                     color='red', s=100, zorder=5, edgecolors='black', linewidth=2)
            ax.axvline(current_step, linestyle='--', color='red', alpha=0.5)
            
            ax.set_xlabel('Ionic Step', fontsize=9)
            ax.set_ylabel('Max Force (eV/Å)', fontsize=9)
            ax.set_title('Force Convergence', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            current_force = max_forces[current_step]
            st.metric("Max Force", f"{current_force:.4f} eV/Å")
            
            if current_force < 0.05:
                st.success("✅ Structure Converged!")
            else:
                st.warning("⚠️ Still Relaxing...")
    
    # Bottom explanation
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔵 Vacancy Defect**
        - Missing atom creates void
        - Neighbors relax **inward**
        - Reduces strain energy
        - Common in irradiated materials
        """)
    
    with col2:
        st.markdown("""
        **🟠 Interstitial Defect**
        - Extra atom in lattice
        - Neighbors push **outward**
        - High formation energy
        - Important for doping
        """)
    
    with col3:
        st.markdown("""
        **🟡 Substitutional Defect**
        - Atom replaced by impurity
        - Size mismatch → relaxation
        - Moderate distortion
        - Most common dopant type
        """)
