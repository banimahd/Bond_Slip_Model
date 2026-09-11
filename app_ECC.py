import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bond-Slip Prediction Model",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.markdown(
        "<div style='font-size: 60px; text-align: center;'>📊</div>",
        unsafe_allow_html=True
    )

with col_title:
    st.markdown(
        "<h1 style='margin-bottom: 0;'>Bond-Slip Prediction Model</h1>"
        "<p style='font-size: 18px; color: gray; margin-top: 0;'>"
        "Deformed Steel Rebar in ECC and Concrete</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================================
# TABS (2 tabs only)
# ============================================================
tab1, tab2 = st.tabs(["🔵 Application", "ℹ️ About"])


# ============================================================
# TAB 1: APPLICATION
# ============================================================
with tab1:
    # --- Sidebar Inputs ---
    st.sidebar.header("Inputs")
    
    db = st.sidebar.selectbox(
        "Bar diameter (mm)",
        options=[10, 12, 16, 20],
        index=0
    )
    
    Confinment = st.sidebar.selectbox(
        "Confinement condition",
        options=[0, 1],
        format_func=lambda x: "Unconfined" if x == 0 else "Confined",
        index=1
    )
    
    Material = st.sidebar.selectbox(
        "Material type",
        options=[0, 1],
        format_func=lambda x: "Concrete" if x == 0 else "ECC",
        index=1
    )
    
    run_button = st.sidebar.button("▶ Run", type="primary", use_container_width=True)
    
    # --- Made by ---
    st.sidebar.markdown(
        "<div style='text-align: center; margin-top: 25px; font-size: 13px; color: gray;'>"
        "Made by "
        "<a href='mailto:banimahd@gmail.com' "
        "style='color: #1f77b4; text-decoration: none; font-weight: bold;'>"
        "Seyed Amir Banimahd</a>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # ============================================================
    # MODEL COEFFICIENTS
    # ============================================================
    T1 = np.array([
        [-0.00001,  0.00609,  0.00169,  0.01524, -0.00639, -0.00582,  0.00234, -0.01145],
        [ 0.00706, -0.25656, -0.05913, -0.68619,  0.27556,  0.25794, -0.09031,  0.51169],
        [-0.13658,  3.25625,  0.71650,  9.84392, -3.58808, -3.68508,  1.38875, -7.31508],
        [ 2.02000, -9.80000, -1.72000, -42.78000, 15.42000, 19.13000, -5.91000, 35.74000],
        [-0.02710,  0.00609, -0.00405,  0.01524, -0.01482, -0.00582, -0.00650, -0.01145],
        [ 1.27913, -0.25656,  0.18356, -0.68619,  0.64869,  0.25794,  0.32075,  0.51169],
        [-19.48483, 3.25625, -2.53842,  9.84392, -8.83558, -3.68508, -4.67050, -7.31508],
        [98.66000, -9.80000, 13.46000, -42.78000, 39.62000, 19.13000, 23.44000, 35.74000],
        [ 0.00824, -0.00545, -0.02388,  0.00749, -0.00047,  0.00034,  0.01832, -0.00253],
        [-0.27769,  0.26119,  1.09225, -0.34169,  0.08281, -0.01681, -0.82169,  0.10994],
        [ 2.82492, -4.10808, -15.53900, 5.00092, -1.55125,  0.30975, 12.62258, -1.49225],
        [-1.15000, 21.70000, 77.04000, -22.86000, 13.60000, -1.56000, -55.46000, 6.94000]
    ])
    
    X4 = np.array([
        [15.1753, 8.6464,  13.9963, 8.6464],
        [15.6701, 15.6701, 8.0067,  14.1081],
        [14.6768, 16.4155, 9.8250,  11.6549],
        [13.2137, 13.2273, 12.4297, 15.0762]
    ])
    
    # ============================================================
    # COMPUTE MODEL
    # ============================================================
    def compute_model(db, Confinment, Material):
        if Material == 0 and Confinment == 0:
            col_x, col_y = 0, 1
            label = 'Concrete - Unconfined (CU)'
            col_idx = 0
        elif Material == 0 and Confinment == 1:
            col_x, col_y = 2, 3
            label = 'Concrete - Confined (CC)'
            col_idx = 1
        elif Material == 1 and Confinment == 0:
            col_x, col_y = 4, 5
            label = 'ECC - Unconfined (EU)'
            col_idx = 2
        elif Material == 1 and Confinment == 1:
            col_x, col_y = 6, 7
            label = 'ECC - Confined (EC)'
            col_idx = 3
        else:
            raise ValueError("Invalid input")
        
        vec = np.array([db**3, db**2, db, 1])
        
        x1 = vec @ T1[0:4, col_x]
        y1 = vec @ T1[0:4, col_y]
        x2 = vec @ T1[4:8, col_x]
        y2 = vec @ T1[4:8, col_y]
        x3 = vec @ T1[8:12, col_x]
        y3 = vec @ T1[8:12, col_y]
        
        diameters = [10, 12, 16, 20]
        row_idx = diameters.index(db)
        x4 = X4[row_idx, col_idx]
        y4 = y3
        
        x_model = np.array([0, x1, x2, x3, x4])
        y_model = np.array([0, y1, y2, y3, y4])
        
        return {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'x3': x3, 'y3': y3,
            'x4': x4, 'y4': y4,
            'x_model': x_model,
            'y_model': y_model,
            'tau_max': np.max(y_model),
            'tau_res': y3,
            'S_peak': x2,
            'S_res': x3,
            'label': label,
            'db': db
        }
    
    # ============================================================
    # PLOT FUNCTION
    # ============================================================
    def plot_data(results):
        x1, y1 = results['x1'], results['y1']
        x2, y2 = results['x2'], results['y2']
        x3, y3 = results['x3'], results['y3']
        x4, y4 = results['x4'], results['y4']
        x_model = results['x_model']
        y_model = results['y_model']
        label = results['label']
        db = results['db']
        
        dist_12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dist_34 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
        
        threshold_x = 0.10 * np.max(x_model)
        threshold_y = 0.10 * np.max(y_model)
        
        if dist_12 < threshold_x:
            offset_x1, offset_x2 = -0.5, +0.5
        else:
            offset_x1, offset_x2 = 0, 0
        
        if dist_34 < threshold_x:
            offset_x3, offset_x4 = -0.5, +0.5
        else:
            offset_x3, offset_x4 = 0, 0
        
        offset_y3 = -0.10 * np.max(y_model) if (abs(y1 - y3) < threshold_y and abs(x1 - x3) < threshold_x) else 0
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(x_model, y_model, 'b-', linewidth=3, label='Model Prediction')
        ax.plot([x1, x2, x3, x4], [y1, y2, y3, y4], 'ro',
                markersize=14, markerfacecolor='r', markeredgewidth=2,
                label='Characteristic Points')
        
        y_offset = 0.05 * np.max(y_model)
        
        ax.text(x1 + offset_x1, y1 + y_offset, 
                f'P1\nS={x1:.3f}\nτ={y1:.3f}',
                fontsize=13, color='k', fontweight='bold',
                va='bottom', ha='center')
        
        ax.text(x2 + offset_x2, y2 + y_offset, 
                f'P2\nS={x2:.3f}\nτ={y2:.3f}',
                fontsize=13, color='k', fontweight='bold',
                va='bottom', ha='center')
        
        ax.text(x3 + offset_x3, y3 + y_offset + offset_y3, 
                f'P3\nS={x3:.3f}\nτ={y3:.3f}',
                fontsize=13, color='k', fontweight='bold',
                va='bottom', ha='center')
        
        ax.text(x4 + offset_x4, y4 + y_offset, 
                f'P4\nS={x4:.3f}\nτ={y4:.3f}',
                fontsize=13, color='k', fontweight='bold',
                va='bottom', ha='center')
        
        ax.set_xlabel('Slip (mm)', fontsize=18, color='k', fontweight='bold')
        ax.set_ylabel('Normalized Bond Stress', fontsize=18, color='k', fontweight='bold')
        ax.set_title(f'Predicted Bond-Slip for $d_b$ = {db} mm - {label}',
                     fontsize=18, color='k', fontweight='bold')
        
        ax.grid(True, linewidth=0.8)
        ax.tick_params(labelsize=15, colors='k')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        ax.set_ylim(0, 1.55 * np.max(y_model))
        ax.set_xlim(0, np.max(x_model) * 1.25)
        
        ax.legend(fontsize=13, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    # ============================================================
    # MAIN LAYOUT
    # ============================================================
    if run_button:
        results = compute_model(db, Confinment, Material)
        
        fig = plot_data(results)
        st.pyplot(fig)
        
        st.divider()
        st.markdown("### Outputs")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**P1**")
            st.markdown(f"S<sub>1</sub> = `{results['x1']:.3f}` mm", unsafe_allow_html=True)
            st.markdown(f"τ<sub>1</sub> = `{results['y1']:.3f}`", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**P2**")
            st.markdown(f"S<sub>2</sub> = `{results['x2']:.3f}` mm", unsafe_allow_html=True)
            st.markdown(f"τ<sub>2</sub> = `{results['y2']:.3f}`", unsafe_allow_html=True)
        
        with col3:
            st.markdown("**P3**")
            st.markdown(f"S<sub>3</sub> = `{results['x3']:.3f}` mm", unsafe_allow_html=True)
            st.markdown(f"τ<sub>3</sub> = `{results['y3']:.3f}`", unsafe_allow_html=True)
        
        with col4:
            st.markdown("**P4**")
            st.markdown(f"S<sub>4</sub> = `{results['x4']:.3f}` mm", unsafe_allow_html=True)
            st.markdown(f"τ<sub>4</sub> = `{results['y4']:.3f}`", unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### Key Metrics")
        
        m1, m2, m3, m4 = st.columns(4)
        
        m1.markdown("**τ<sub>max</sub>**", unsafe_allow_html=True)
        m1.markdown(f"`{results['tau_max']:.3f}`")
        
        m2.markdown("**τ<sub>res</sub>**", unsafe_allow_html=True)
        m2.markdown(f"`{results['tau_res']:.3f}`")
        
        m3.markdown("**S<sub>peak</sub> (mm)**", unsafe_allow_html=True)
        m3.markdown(f"`{results['S_peak']:.3f}`")
        
        m4.markdown("**S<sub>res</sub> (mm)**", unsafe_allow_html=True)
        m4.markdown(f"`{results['S_res']:.3f}`")
    
    else:
        st.info("👈 Set the inputs in the sidebar and click **Run** to generate the plot.")


# ============================================================
# TAB 2: ABOUT
# ============================================================
with tab2:
    st.markdown("# ℹ️ About")
    
    st.markdown("## Bond-Slip Prediction Model")
    
    st.markdown(
        "This application predicts the **bond stress-slip behavior** of deformed steel bars "
        "embedded in **Engineered Cementitious Composites (ECC)** and **conventional concrete** "
        "under monotonic pull-out loading."
    )
    
    st.markdown("---")
    
    # Developer info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Developer")
        st.markdown("**Seyed Amir Banimahd**")
        st.markdown("📧 [banimahd@gmail.com](mailto:banimahd@gmail.com)")
        st.markdown("🏛️ Ardakan University")
        st.markdown("📍 Department of Civil Engineering")
    
    with col2:
        st.markdown("### 📋 Features")
        st.markdown("- 4 bar diameters: 10, 12, 16, 20 mm")
        st.markdown("- Confined / Unconfined conditions")
        st.markdown("- Concrete and ECC materials")
        st.markdown("- Empirical piecewise bond-slip model")
        st.markdown("- Characteristic points P1–P4")
    
    st.markdown("---")
    
    # Validity range
    st.markdown("### ✅ Validity Range")
    
    validity_data = {
        "Parameter": [
            "Bar diameter",
            "Concrete f'c",
            "ECC f'c",
            "Bonded length",
            "Loading",
            "Confinement"
        ],
        "Range": [
            "10, 12, 16, 20 mm",
            "63.26 – 70.30 MPa",
            "43.37 – 50.85 MPa",
            "5 × d_b",
            "Monotonic",
            "Two 8 mm stirrups @ 50 mm"
        ]
    }
    st.table(validity_data)
    
    st.markdown("---")
    
    st.markdown("### 📚 Reference")
    st.markdown(
        "Based on the experimental investigation and empirical model developed at "
        "Ardakan University, Department of Civil Engineering."
    )
    
    st.markdown("---")
    
    st.caption("© 2026 Seyed Amir Banimahd. All rights reserved.")
