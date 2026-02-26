"""Method Explanation (BNL) page.

Comparative explanation of the two self-absorption correction methods:
  - Reconstruction (Panpan) — PyTorch AD + Adam optimizer
  - FL Correction (BNL)     — Explicit H-matrix + MLEM

Reference (Reconstruction):
  Huang, P. (2022). PhD Thesis, Chapter 5 — Self-absorption correction in XRF tomography.

Reference (BNL FL Correction):
  Ge, M. et al., Commun. Mater. 3, 37 (2022).
"""

from nicegui import ui


# ──────────────────────────────────────────────────────────────────────────────
# SVG Diagrams
# ──────────────────────────────────────────────────────────────────────────────

_SETUP_SVG = """
<div style="text-align:center; padding:12px 0 4px;">
<svg width="700" height="230" viewBox="0 0 700 230" style="max-width:100%;"
     font-family="Arial, sans-serif">
  <defs>
    <marker id="ab" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/>
    </marker>
    <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/>
    </marker>
    <marker id="ag" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/>
    </marker>
  </defs>
  <rect width="700" height="230" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Source -->
  <rect x="18" y="88" width="88" height="54" rx="8" fill="#1e293b"/>
  <text x="62" y="110" text-anchor="middle" fill="white" font-size="10" font-weight="bold">X-ray</text>
  <text x="62" y="126" text-anchor="middle" fill="#94a3b8" font-size="9">Source  I₀</text>
  <!-- Beam -->
  <line x1="106" y1="115" x2="238" y2="115" stroke="#f59e0b" stroke-width="3" marker-end="url(#ab)"/>
  <line x1="106" y1="115" x2="238" y2="93"  stroke="#f59e0b" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>
  <line x1="106" y1="115" x2="238" y2="137" stroke="#f59e0b" stroke-width="1" stroke-dasharray="5,3" opacity="0.5"/>
  <text x="170" y="104" text-anchor="middle" fill="#b45309" font-size="10">Probe beam</text>
  <!-- Sample -->
  <circle cx="312" cy="115" r="74" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <line x1="248" y1="85"  x2="376" y2="85"  stroke="#93c5fd" stroke-width="0.6"/>
  <line x1="248" y1="115" x2="376" y2="115" stroke="#93c5fd" stroke-width="0.6"/>
  <line x1="248" y1="145" x2="376" y2="145" stroke="#93c5fd" stroke-width="0.6"/>
  <line x1="280" y1="43"  x2="280" y2="187" stroke="#93c5fd" stroke-width="0.6"/>
  <line x1="312" y1="41"  x2="312" y2="189" stroke="#93c5fd" stroke-width="0.6"/>
  <line x1="344" y1="43"  x2="344" y2="187" stroke="#93c5fd" stroke-width="0.6"/>
  <!-- voxel v -->
  <rect x="296" y="99" width="32" height="32" rx="3" fill="#2563eb" fill-opacity="0.35" stroke="#2563eb" stroke-width="1.5"/>
  <text x="312" y="119" text-anchor="middle" fill="#1e40af" font-size="11" font-style="italic" font-weight="bold">v</text>
  <text x="312" y="206" text-anchor="middle" fill="#1d4ed8" font-size="10">Rotating sample  (angle θ, position p)</text>
  <!-- rotation arc -->
  <path d="M312,34 A82,82 0 0 1 387,72" stroke="#94a3b8" stroke-width="2" fill="none"
        stroke-dasharray="5,3" marker-end="url(#ag)"/>
  <text x="375" y="58" fill="#64748b" font-size="13" font-style="italic">θ</text>
  <!-- XRF -->
  <line x1="340" y1="104" x2="492" y2="68"  stroke="#dc2626" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#ar)"/>
  <line x1="340" y1="114" x2="492" y2="90"  stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.6" marker-end="url(#ar)"/>
  <text x="425" y="68" fill="#b91c1c" font-size="10" text-anchor="middle">XRF  D^R_{θ,p,l}</text>
  <!-- XRF detector -->
  <rect x="500" y="52" width="88" height="72" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="544" y="80"  text-anchor="middle" fill="#92400e" font-size="10" font-weight="bold">XRF</text>
  <text x="544" y="94"  text-anchor="middle" fill="#92400e" font-size="9">Detector</text>
  <!-- Transmitted -->
  <line x1="386" y1="115" x2="590" y2="115" stroke="#f59e0b" stroke-width="2" opacity="0.45" marker-end="url(#ab)"/>
  <!-- XRT detector -->
  <rect x="598" y="88" width="88" height="54" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="642" y="111" text-anchor="middle" fill="#166534" font-size="10" font-weight="bold">XRT</text>
  <text x="642" y="126" text-anchor="middle" fill="#166534" font-size="9">D^T_{θ,p}</text>
</svg>
<div style="font-size:0.78rem; color:#64748b; margin-top:6px; font-style:italic;">
Figure 1 — XRF tomography setup (Huang 2022, Fig. 5.1 / 5.3). The probe beam enters the rotating
sample at angle θ, position p. XRF and XRT are recorded simultaneously.
</div>
</div>
"""


_SELF_ABSORPTION_SVG = """
<div style="text-align:center; padding:12px 0 4px;">
<svg width="680" height="270" viewBox="0 0 680 270" style="max-width:100%;"
     font-family="Arial, sans-serif">
  <defs>
    <marker id="sa-ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#dc2626"/>
    </marker>
    <marker id="sa-or" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#f59e0b"/>
    </marker>
  </defs>
  <rect width="680" height="270" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Voxel grid -->
  <rect x="155" y="55"  width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="210" y="55"  width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="265" y="55"  width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="155" y="110" width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="265" y="110" width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="155" y="165" width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="210" y="165" width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <rect x="265" y="165" width="55" height="55" rx="3" fill="#fee2e2" stroke="#fca5a5"/>
  <!-- Source voxel v (blue) -->
  <rect x="210" y="110" width="55" height="55" rx="3" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>
  <text x="237" y="137" text-anchor="middle" fill="#1d4ed8" font-size="10" font-weight="bold">voxel v</text>
  <text x="237" y="152" text-anchor="middle" fill="#1d4ed8" font-size="9">(source)</text>
  <!-- Legend -->
  <rect x="122" y="110" width="14" height="14" rx="2" fill="#fee2e2" stroke="#fca5a5"/>
  <text x="140" y="121" fill="#7f1d1d" font-size="9">Other elements</text>
  <rect x="122" y="130" width="14" height="14" rx="2" fill="#bfdbfe" stroke="#2563eb"/>
  <text x="140" y="141" fill="#1e3a8a" font-size="9">Source voxel v</text>
  <!-- Sample outline -->
  <rect x="155" y="55" width="165" height="165" rx="4" fill="none" stroke="#475569" stroke-width="2"/>
  <!-- Probe beam -->
  <line x1="70" y1="138" x2="208" y2="138" stroke="#f59e0b" stroke-width="3" marker-end="url(#sa-or)"/>
  <text x="118" y="127" fill="#b45309" font-size="10">Probe  A^{θ,p}_v</text>
  <!-- XRF path 1: short path, reaches detector -->
  <line x1="268" y1="128" x2="510" y2="82"  stroke="#dc2626" stroke-width="2.5" marker-end="url(#sa-ar)"/>
  <text x="400" y="88" fill="#991b1b" font-size="10" font-weight="bold">Detected  ✓</text>
  <text x="360" y="105" fill="#64748b" font-size="9">P_{v,v',d} — short path</text>
  <text x="360" y="117" fill="#64748b" font-size="9">B_{l,v} close to 1</text>
  <!-- XRF path 2: long path, absorbed -->
  <line x1="237" y1="165" x2="440" y2="230" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4" opacity="0.55"/>
  <text x="450" y="234" fill="#dc2626" font-size="14" font-weight="bold">✗</text>
  <text x="470" y="234" fill="#7f1d1d" font-size="10">Absorbed</text>
  <text x="340" y="222" fill="#64748b" font-size="9">P_{v,v',d} — long path</text>
  <text x="340" y="234" fill="#64748b" font-size="9">B_{l,v} ≪ 1</text>
  <!-- Detector -->
  <rect x="512" y="60" width="80" height="60" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="552" y="87"  text-anchor="middle" fill="#92400e" font-size="10" font-weight="bold">XRF Det.</text>
  <text x="552" y="104" text-anchor="middle" fill="#92400e" font-size="9">D^R_{θ,p,l}</text>
</svg>
<div style="font-size:0.78rem; color:#64748b; margin-top:6px; font-style:italic;">
Figure 2 — Self-absorption. B_{l,v} (Eq. 5.9) is the survival probability of XRF photons from voxel v
to detector point d, averaged over n_d detector sampling points.
</div>
</div>
"""


_FORWARD_SVG = """
<div style="text-align:center; padding:12px 0 4px;">
<svg width="700" height="155" viewBox="0 0 700 155" style="max-width:100%;"
     font-family="Arial, sans-serif" font-size="11">
  <defs>
    <marker id="fm-ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>
  <rect width="700" height="155" rx="10" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
  <!-- Box 1 -->
  <rect x="14" y="32" width="90" height="68" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="59" y="54"  text-anchor="middle" fill="#5b21b6" font-weight="bold">X = [X_{v,e}]</text>
  <text x="59" y="70"  text-anchor="middle" fill="#6d28d9" font-size="9">Density of elem e</text>
  <text x="59" y="83"  text-anchor="middle" fill="#6d28d9" font-size="9">at voxel v</text>
  <!-- arrow -->
  <line x1="104" y1="66" x2="130" y2="66" stroke="#475569" stroke-width="1.5" marker-end="url(#fm-ar)"/>
  <!-- Box 2 -->
  <rect x="132" y="32" width="108" height="68" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="186" y="54"  text-anchor="middle" fill="#5b21b6" font-weight="bold">A^{θ,p}_v</text>
  <text x="186" y="69"  text-anchor="middle" fill="#6d28d9" font-size="9">Probe attenuation</text>
  <text x="186" y="82"  text-anchor="middle" fill="#6d28d9" font-size="9">Eq. 5.6</text>
  <!-- arrow -->
  <line x1="240" y1="66" x2="268" y2="66" stroke="#475569" stroke-width="1.5" marker-end="url(#fm-ar)"/>
  <!-- Box 3 -->
  <rect x="270" y="32" width="110" height="68" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
  <text x="325" y="54"  text-anchor="middle" fill="#1e40af" font-weight="bold">F^{θ,p}_{l,v}</text>
  <text x="325" y="69"  text-anchor="middle" fill="#1d4ed8" font-size="9">Emission at v</text>
  <text x="325" y="82"  text-anchor="middle" fill="#1d4ed8" font-size="9">= I₀·A·τ·C·Δs  (Eq. 5.8)</text>
  <!-- arrow -->
  <line x1="380" y1="66" x2="408" y2="66" stroke="#475569" stroke-width="1.5" marker-end="url(#fm-ar)"/>
  <!-- Box 4 -->
  <rect x="410" y="32" width="108" height="68" rx="8" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>
  <text x="464" y="54"  text-anchor="middle" fill="#991b1b" font-weight="bold">B_{l,v}</text>
  <text x="464" y="69"  text-anchor="middle" fill="#b91c1c" font-size="9">Self-absorption</text>
  <text x="464" y="82"  text-anchor="middle" fill="#b91c1c" font-size="9">Eq. 5.9</text>
  <!-- arrow -->
  <line x1="518" y1="66" x2="546" y2="66" stroke="#475569" stroke-width="1.5" marker-end="url(#fm-ar)"/>
  <!-- Box 5 -->
  <rect x="548" y="32" width="136" height="68" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="616" y="50"  text-anchor="middle" fill="#166534" font-weight="bold">F̂^R_{θ,p,l}(X)</text>
  <text x="616" y="66"  text-anchor="middle" fill="#166534" font-size="9">Predicted XRF</text>
  <text x="616" y="79"  text-anchor="middle" fill="#166534" font-size="9">= Σ_v F·ε·B  (Eq. 5.10)</text>
  <!-- caption -->
  <text x="350" y="124" text-anchor="middle" fill="#64748b" font-size="10" font-style="italic">
    Eq. 5.10:  F̂^R_{θ,p,l}(X) = Σ_v  I₀ · A^{θ,p}_v · τ^Ep_l · X_{v,e} · Δs · ε · B_{l,v}
  </text>
  <text x="350" y="140" text-anchor="middle" fill="#64748b" font-size="9" font-style="italic">
    (shared by both methods — only the inversion strategy differs)
  </text>
</svg>
<div style="font-size:0.78rem; color:#64748b; margin-top:6px; font-style:italic;">
Figure 3 — Common forward model pipeline (Huang 2022, Eqs. 5.6–5.10). This physics is shared by
all three methods — only the inversion strategy differs.
</div>
</div>
"""


_COMPARE_SVG = """
<div style="padding:16px 0 4px;">
<svg width="960" height="490" viewBox="0 0 960 490" style="width:100%; display:block;"
     font-family="Arial, sans-serif" font-size="12">
  <defs>
    <marker id="cp-ar" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#64748b"/>
    </marker>
    <marker id="cp-bl" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#2563eb"/>
    </marker>
    <marker id="cp-pu" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#7c3aed"/>
    </marker>
    <marker id="cp-gr" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L7,3.5 L0,7 Z" fill="#059669"/>
    </marker>
  </defs>
  <rect width="960" height="490" rx="12" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.5"/>

  <!-- ── Shared: Forward model ── -->
  <rect x="330" y="14" width="300" height="68" rx="10" fill="#f1f5f9" stroke="#94a3b8" stroke-width="2"/>
  <text x="480" y="39"  text-anchor="middle" fill="#334155" font-size="15" font-weight="bold">Shared: Forward Model</text>
  <text x="480" y="58"  text-anchor="middle" fill="#475569" font-size="12">F̂ᴿ(X) = Σᵥ I₀ · A · τ · X · Δs · ε · B</text>
  <text x="480" y="75"  text-anchor="middle" fill="#94a3b8" font-size="10">(same physics in all three methods)</text>

  <!-- ── Measured data ── -->
  <rect x="310" y="96" width="340" height="58" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="480" y="121" text-anchor="middle" fill="#92400e" font-size="15" font-weight="bold">Measured data</text>
  <text x="480" y="143" text-anchor="middle" fill="#b45309" font-size="12">Dᴿ (XRF sinogram)  +  Dᵀ (XRT optical depth)</text>

  <!-- arrows to three method boxes -->
  <line x1="412" y1="154" x2="158" y2="180" stroke="#64748b" stroke-width="2" marker-end="url(#cp-ar)"/>
  <line x1="480" y1="154" x2="480" y2="180" stroke="#64748b" stroke-width="2" marker-end="url(#cp-ar)"/>
  <line x1="548" y1="154" x2="802" y2="180" stroke="#64748b" stroke-width="2" marker-end="url(#cp-ar)"/>

  <!-- ── LEFT: Di et al. 2017 ── -->
  <rect x="8" y="182" width="300" height="246" rx="10" fill="#f0fdf4" stroke="#059669" stroke-width="2.5"/>
  <text x="158" y="207" text-anchor="middle" fill="#065f46" font-size="14" font-weight="bold">Di et al. 2017 (ANL)</text>
  <line x1="26" y1="216" x2="290" y2="216" stroke="#bbf7d0" stroke-width="1.2"/>
  <text x="158" y="236" text-anchor="middle" fill="#047857" font-size="11.5">Full spectrum Mₑ (Gaussian detector blur)</text>
  <text x="158" y="257" text-anchor="middle" fill="#047857" font-size="11.5">Ω_v pyramid region for self-absorption</text>
  <text x="158" y="278" text-anchor="middle" fill="#047857" font-size="11.5">Loss = Poisson LL(XRF) + β₁·Poisson LL(XRT)</text>
  <text x="158" y="299" text-anchor="middle" fill="#047857" font-size="11.5">Freeze Aⁱ, Pⁱ each outer iteration</text>
  <text x="158" y="320" text-anchor="middle" fill="#047857" font-size="11.5">Inner: Truncated Newton (k = 52 steps)</text>
  <text x="158" y="341" text-anchor="middle" fill="#047857" font-size="11.5">Backtracking line search (Algorithm 2)</text>
  <text x="158" y="362" text-anchor="middle" fill="#047857" font-size="11.5">Converges in ~3 outer iterations</text>
  <text x="158" y="385" text-anchor="middle" fill="#6b7280" font-size="10.5" font-style="italic">No code in this repo (theory reference)</text>
  <line x1="158" y1="428" x2="158" y2="452" stroke="#059669" stroke-width="2.5" marker-end="url(#cp-gr)"/>
  <text x="158" y="472" text-anchor="middle" fill="#065f46" font-size="12" font-weight="bold">3D W* = [W*_{v,e}]  (g/cm³)</text>

  <!-- ── CENTER: Panpan ── -->
  <rect x="330" y="182" width="300" height="246" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/>
  <text x="480" y="207" text-anchor="middle" fill="#1e40af" font-size="14" font-weight="bold">Reconstruction (Panpan)</text>
  <line x1="348" y1="216" x2="612" y2="216" stroke="#bfdbfe" stroke-width="1.2"/>
  <text x="480" y="236" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Per-line τ (no full spectrum model)</text>
  <text x="480" y="257" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Same alternating freeze idea as Di et al.</text>
  <text x="480" y="278" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Loss = MSE(XRF) + λ·MSE(XRT)  (Eq. 5.12)</text>
  <text x="480" y="299" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Gradients via PyTorch autograd</text>
  <text x="480" y="320" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Update: Adam optimizer</text>
  <text x="480" y="341" text-anchor="middle" fill="#1d4ed8" font-size="11.5">MPI minibatch (3D slabs, parallel ranks)</text>
  <text x="480" y="362" text-anchor="middle" fill="#1d4ed8" font-size="11.5">Extends Di et al. to 3D with GPU + MPI</text>
  <text x="480" y="385" text-anchor="middle" fill="#6b7280" font-size="10.5" font-style="italic">src/reconstruction/</text>
  <line x1="480" y1="428" x2="480" y2="452" stroke="#2563eb" stroke-width="2.5" marker-end="url(#cp-bl)"/>
  <text x="480" y="472" text-anchor="middle" fill="#1e40af" font-size="12" font-weight="bold">3D X* = [X*_{v,e}]</text>

  <!-- ── RIGHT: BNL ── -->
  <rect x="652" y="182" width="300" height="246" rx="10" fill="#fdf4ff" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="802" y="207" text-anchor="middle" fill="#5b21b6" font-size="14" font-weight="bold">FL Correction (BNL)</text>
  <line x1="670" y1="216" x2="934" y2="216" stroke="#e9d5ff" stroke-width="1.2"/>
  <text x="802" y="236" text-anchor="middle" fill="#6d28d9" font-size="11.5">Explicit H-matrix (attenuated Radon)</text>
  <text x="802" y="257" text-anchor="middle" fill="#6d28d9" font-size="11.5">H encodes geometry + self-absorption</text>
  <text x="802" y="278" text-anchor="middle" fill="#6d28d9" font-size="11.5">XRF-only — no XRT term used</text>
  <text x="802" y="299" text-anchor="middle" fill="#6d28d9" font-size="11.5">Solve H·C = I by MLEM</text>
  <text x="802" y="320" text-anchor="middle" fill="#6d28d9" font-size="11.5">Cₙₑw = C · (HᵀI/HC) / Hᵀ·1</text>
  <text x="802" y="341" text-anchor="middle" fill="#6d28d9" font-size="11.5">Slice-by-slice (2D per element)</text>
  <text x="802" y="362" text-anchor="middle" fill="#6d28d9" font-size="11.5">CPU multiprocess or GPU (numba CUDA)</text>
  <text x="802" y="385" text-anchor="middle" fill="#6b7280" font-size="10.5" font-style="italic">src/fl_correction/</text>
  <line x1="802" y1="428" x2="802" y2="452" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#cp-pu)"/>
  <text x="802" y="472" text-anchor="middle" fill="#5b21b6" font-size="12" font-weight="bold">Corrected 3D tomo slices</text>
</svg>
<div style="font-size:0.8rem; color:#64748b; margin-top:8px; font-style:italic;">
Figure 4 — All three methods share the same forward model physics. They differ in loss function,
optimizer, self-absorption approximation, and whether XRT is used.
</div>
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Equation HTML
# ──────────────────────────────────────────────────────────────────────────────

_EQ_CSS = "font-family:Georgia,serif; font-size:1.05rem; color:#1e293b; text-align:center; padding:10px; border-radius:6px;"

_EQ_PROBE_ATT = f"""
<div style="{_EQ_CSS} background:#f5f3ff;">
  A<sup>θ,p</sup><sub>v</sub> = exp
  (&minus;Δ<sub>s</sub> &Sigma;<sub>v'∈U<sup>v</sup><sub>θ,p</sub>, e</sub>
  μ̃<sup>Ep</sup><sub>e</sub> · X<sub>v',e</sub>)
  <div style="font-size:0.78rem; color:#6d28d9; margin-top:4px;">
  Eq. 5.6 — upstream voxels U<sup>v</sup><sub>θ,p</sub> attenuate the probe before it reaches v
  </div>
</div>
"""

_EQ_EMISSION = f"""
<div style="{_EQ_CSS} background:#eff6ff;">
  F<sup>θ,p</sup><sub>l,v</sub> = I<sub>0</sub> · A<sup>θ,p</sup><sub>v</sub>
  · τ<sup>Ep</sup><sub>l</sub> · X<sub>v,e</sub> · Δ<sub>s</sub>
  <div style="font-size:0.78rem; color:#1d4ed8; margin-top:4px;">
  Eq. 5.8 — τ<sup>Ep</sup><sub>l</sub> = mass production cross section (from xraylib)
  </div>
</div>
"""

_EQ_SA = f"""
<div style="{_EQ_CSS} background:#fff1f2;">
  B<sub>l,v</sub> =
  (1/n<sub>d</sub>) &Sigma;<sub>d</sub>
  exp(&minus;&Sigma;<sub>v',e</sub> μ̃<sup>l</sup><sub>e</sub> · X<sub>v',e</sub> · P<sub>v,v',d</sub>)
  <div style="font-size:0.78rem; color:#b91c1c; margin-top:4px;">
  Eq. 5.9 — P<sub>v,v',d</sub> = intersecting path length of v' with FL beam from v to detector d
  </div>
</div>
"""

_EQ_XRF = f"""
<div style="{_EQ_CSS} background:#f0fdf4; font-size:0.95rem;">
  F̂<sup>R</sup><sub>θ,p,l</sub>(X) = &Sigma;<sub>v∈T<sub>θ,p</sub></sub>
  F<sup>θ,p</sup><sub>l,v</sub> · ε · B<sub>l,v</sub>
  <div style="font-size:0.78rem; color:#166534; margin-top:4px;">
  Eq. 5.10 — ε = total detector efficiency (solid angle × quantum efficiency)
  </div>
</div>
"""

_EQ_XRT = f"""
<div style="{_EQ_CSS} background:#fefce8; font-size:0.95rem;">
  F̂<sup>T</sup><sub>θ,p</sub>(X) = Δ<sub>s</sub>
  &Sigma;<sub>v'∈T<sub>θ,p</sub>, e</sub>
  μ̃<sup>Ep</sup><sub>e</sub> · R<sub>z</sub>(X, θ)<sub>v',e</sub>
  <div style="font-size:0.78rem; color:#854d0e; margin-top:4px;">
  Eq. 5.5 — optical density (Beer-Lambert) summed along probe path.
  Measured as D<sup>T</sup><sub>θ,p</sub> = &minus;log(I<sub>T</sub>/I<sub>0</sub>)
  </div>
</div>
"""

_EQ_LOSS = f"""
<div style="{_EQ_CSS} background:#f0fdf4;">
  φ<sub>θ,slice</sub> =
  (1/n<sub>p</sub>n<sub>l</sub>) &Sigma;<sub>p,l</sub>
  (F̂<sup>R</sup><sub>θ,p,l</sub> &minus; D<sup>R</sup><sub>θ,p,l</sub>)²
  + λ · (1/n<sub>p</sub>) &Sigma;<sub>p</sub>
  (F̂<sup>T</sup><sub>θ,p</sub> &minus; D<sup>T</sup><sub>θ,p</sub>)²
  <div style="font-size:0.78rem; color:#166534; margin-top:4px;">
  Eq. 5.12 — λ is the XRT regularization weight (code: b1); balances XRF vs XRT fidelity
  </div>
</div>
"""

_EQ_POISSON = f"""
<div style="{_EQ_CSS} background:#f0fdf4;">
  φ(W) =
  &Sigma;<sub>θ,τ</sub>[F<sup>R</sup><sub>θ,τ</sub>(W) &minus; ln(F<sup>R</sup><sub>θ,τ</sub>(W)) D<sup>R</sup><sub>θ,τ</sub>]
  + β<sub>1</sub>&Sigma;<sub>θ,τ</sub>[F<sup>T</sup><sub>θ,τ</sub>(W) &minus; ln(F<sup>T</sup><sub>θ,τ</sub>(W)) β<sub>2</sub>D<sup>T</sup><sub>θ,τ</sub>]
  <div style="font-size:0.78rem; color:#065f46; margin-top:4px;">
  Eq. 8 (Di et al. 2017) — Poisson log-likelihood objective. β<sub>1</sub> balances XRF/XRT modalities
  (optimal: β<sub>1</sub>=1 via L-curve); β<sub>2</sub>=100 normalises magnitude difference between datasets.
  </div>
</div>
"""

_EQ_MLEM = f"""
<div style="{_EQ_CSS} background:#fdf4ff;">
  C<sup>(n+1)</sup> = C<sup>(n)</sup> ·
  (H<sup>T</sup> (I / H C<sup>(n)</sup>)) / (H<sup>T</sup> 1)
  <div style="font-size:0.78rem; color:#6d28d9; margin-top:4px;">
  MLEM multiplicative update — H is the attenuated Radon matrix (generate_H),
  I is the measured XRF (generate_I)
  </div>
</div>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Variable table
# ──────────────────────────────────────────────────────────────────────────────

_VARIABLE_TABLE = """
<!-- Role badge legend -->
<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; font-size:0.79rem;">
  <span style="background:#fee2e2;color:#b91c1c;padding:2px 9px;border-radius:10px;font-weight:600;">Unknown</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— 3D distribution being solved for</span>
  <span style="background:#dbeafe;color:#1d4ed8;padding:2px 9px;border-radius:10px;font-weight:600;">Measured data</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— recorded by detector, provided as input</span>
  <span style="background:#fef3c7;color:#92400e;padding:2px 9px;border-radius:10px;font-weight:600;">Scan param.</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— set by scan setup (energy, geometry, step size)</span>
  <span style="background:#f1f5f9;color:#475569;padding:2px 9px;border-radius:10px;font-weight:600;">Physical const.</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— tabulated (xraylib / detector response)</span>
  <span style="background:#d1fae5;color:#065f46;padding:2px 9px;border-radius:10px;font-weight:600;">Derived f(X)</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— computed from X during forward pass</span>
  <span style="background:#ede9fe;color:#5b21b6;padding:2px 9px;border-radius:10px;font-weight:600;">Hyperparameter</span>
  <span style="font-size:0.75rem;color:#64748b;align-self:center;">— user-chosen (L-curve, manual tuning)</span>
</div>
<div style="overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:0.82rem; min-width:750px;">
  <thead>
    <tr style="background:#f1f5f9; border-bottom:2px solid #cbd5e1; position:sticky; top:0; z-index:1;">
      <th style="padding:7px 10px; text-align:left; color:#374151; white-space:nowrap;">Symbol</th>
      <th style="padding:7px 10px; text-align:left; color:#374151; white-space:nowrap;">Role</th>
      <th style="padding:7px 10px; text-align:left; color:#374151;">Meaning</th>
      <th style="padding:7px 10px; text-align:left; color:#065f46; font-family:monospace; font-size:0.76rem;">Di et al. 2017</th>
      <th style="padding:7px 10px; text-align:left; color:#2563eb; font-family:monospace; font-size:0.76rem;">Reconstruction (Panpan)</th>
      <th style="padding:7px 10px; text-align:left; color:#7c3aed; font-family:monospace; font-size:0.76rem;">FL Correction (BNL)</th>
    </tr>
  </thead>
  <tbody>

    <!-- ══ SECTION: Unknown ══ -->
    <tr style="background:#fca5a530;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#b91c1c; letter-spacing:0.05em;">UNKNOWN — THE RECONSTRUCTION TARGET</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fff5f5;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">X<sub>v,e</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fee2e2;color:#b91c1c;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Unknown</span></td>
      <td style="padding:6px 10px; color:#374151;">3D elemental mass density — what the algorithm recovers. Shape: (n_element, n_z, n_y, n_x).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">W<sub>v,e</sub> (g/cm³)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">xp = nn.Parameter(grid_concentration[…])<br>shape: (n_elem, minibatch, n_y)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">img2D per slice (n_row × n_col);<br>img3D = stack of corrected slices</td>
    </tr>

    <!-- ══ SECTION: Measured data ══ -->
    <tr style="background:#93c5fd30;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#1d4ed8; letter-spacing:0.05em;">MEASURED DATA — DETECTOR RECORDINGS (PROVIDED AS INPUT FILES)</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#eff6ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">D<sup>R</sup><sub>θ,p,l</sub></td>
      <td style="padding:6px 8px;"><span style="background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Measured data</span></td>
      <td style="padding:6px 10px; color:#374151;">XRF sinogram — measured fluorescence counts at each angle θ, probe position p, and energy channel l. Recorded by energy-dispersive detector (EDD).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">D<sup>R</sup><sub>θ,τ</sub> (|I|-vector, 800 energy channels per beamlet)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">y1_true — shape (n_lines_roi, n_theta, n_height × n_width);<br>loaded from exchange/data in HDF5</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">I_tot — 1D vector, shape (n_col × n_angles);<br>from generate_I() reading elem_ref_prj_*.tiff</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#eff6ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">D<sup>T</sup><sub>θ,p</sub></td>
      <td style="padding:6px 8px;"><span style="background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Measured data</span></td>
      <td style="padding:6px 10px; color:#374151;">XRT optical density — −log(I_transmitted / I₀), recorded by downstream ion chamber. Constrains total attenuation independently of element identity.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">D<sup>T</sup><sub>θ,τ</sub></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">y2_true = −log(abs_ic);<br>from XRT_ratio_dataset_idx in scaler HDF5</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not used — BNL is XRF-only</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#eff6ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">ref<sup>3D</sup></td>
      <td style="padding:6px 8px;"><span style="background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Measured data</span></td>
      <td style="padding:6px 10px; color:#374151;">Reference 3D tomography volume from which the BNL H-matrix attenuation is computed. Provides the geometry for building H. Typically a conventional XRF tomo recon.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>not used</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;"><em>not used — X is the direct unknown</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">ref3D_tomo — shape (n_sli, n_row, n_col);<br>passed to generate_H() and generate_I()</td>
    </tr>

    <!-- ══ SECTION: Scan parameters ══ -->
    <tr style="background:#fde68a30;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#92400e; letter-spacing:0.05em;">SCAN PARAMETERS — SET BY EXPERIMENT SETUP (REQUIRED INPUT)</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">E<sub>p</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Probe X-ray energy in keV. Used to look up mass attenuation coefficients μ at probe energy from xraylib.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">E (keV)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">probe_energy (array, e.g. [20.0])</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">probe_energy (float, e.g. 13.577) → internal XEng</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">I₀</td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Incident probe photon flux/intensity. Scales the absolute magnitude of the forward-modelled XRF signal. Calibrated from standard reference or set manually.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">I₀</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">probe_cts (from calibrate_incident_probe_intensity or probe_intensity);<br>init_probe() → probe_before_attenuation_flat</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>implicit — absorbed into I_tot signal</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">Θ = {θ}</td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Set of rotation angles in degrees/radians. Defines the tomographic projections. Shuffled randomly each epoch (Panpan) to reduce correlation between updates.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">Θ (angle set)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">theta_ls_dataset = "exchange/theta" → theta_ls (×π/180 → rad);<br>shuffled per epoch as theta_ls_rand</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">theta_ls_dataset = "thetas" → angle_list (×π/180 → rad);<br>passed to generate_H(), generate_I(), cal_atten_with_direction()</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">Δs</td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Voxel side length in cm. Sets the physical scale of attenuation integrals. Computed as sample_size_cm / sample_size_n.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">L<sup>θ,τ</sup><sub>v</sub> — exact ray-voxel intersection length per beamlet</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">sample_size_cm / sample_size_n;<br>used in att_exponent_acc = lac_acc × Δs</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">pix = pixel_size_nm × 1e-7 cm;<br>implicit in bilinear interp weights inside H</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">n<sub>z</sub>, n<sub>y</sub>, n<sub>x</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Volume dimensions in pixels. n_z = number of slices (height), n_y × n_x = cross-section grid. Determines memory usage and compute cost.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implied by W shape</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">sample_height_n (n_z), sample_size_n (n_y = n_x)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">ref3D_tomo.shape = (n_sli, n_row, n_col)</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">n<sub>d</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Number of detector sampling points. The finite detector area is modelled as n_d discrete point detectors to average the self-absorption path length B.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implicit in Ω_v pyramid area</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">n_det = P_minibatch.shape[0];<br>P_minibatch shape (n_det, 3, dia_len_n × batch)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not explicit — bilinear interp in H</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">d<sub>det</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Detector diameter in cm (simulation). Used to compute solid angle ε and distribute n_d sampling points on the detector plane.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>not explicit</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">det_dia_cm → det_solid_angle_ratio = π(d/2)² / 4π r²</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not used</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">r<sub>det</sub></td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Sample-to-detector distance in cm. Used alongside d_det to compute solid angle ε (simulation) or to place detector coordinate points (experiment).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>not explicit</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">det_from_sample_cm;<br>set_det_coord_cm for real data</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not used</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fffbeb;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">elem list</td>
      <td style="padding:6px 8px;"><span style="background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Scan param.</span></td>
      <td style="padding:6px 10px; color:#374151;">Which chemical elements to reconstruct and which FL lines to include. Drives all xraylib lookups and sets n_element, n_lines.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">element set</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">element_symbols → this_aN_dic (name→Z via xraylib);<br>element_lines_roi; n_line_group_each_element</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">element_symbols → elem_type list (e.g. ['Zr','La','Hf']);<br>passed to get_atten_coef(), generate_H()</td>
    </tr>

    <!-- ══ SECTION: Physical constants ══ -->
    <tr style="background:#94a3b830;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#334155; letter-spacing:0.05em;">PHYSICAL CONSTANTS — TABULATED FROM XRAYLIB / DETECTOR SPECS (NOT FITTED)</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">μ̃<sup>E<sub>p</sub></sup><sub>e</sub></td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">Mass attenuation coefficient of element e at probe energy E_p (cm²/g). Governs Beer-Lambert probe attenuation A through the sample.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">μ<sup>E</sup><sub>e</sub> (xraylib CS_Total)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">probe_attCS_ls = xlib_np.CS_Total(aN_ls, probe_energy)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">cs[f'{elem}-x'] = xraylib.CS_Total(Z, XEng)</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fafafa;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">μ̃<sup>E<sub>l</sub></sup><sub>e</sub></td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">Mass attenuation coefficient of element e at fluorescence energy E_l (cm²/g). Governs self-absorption B — how strongly each element absorbs the emitted XRF photons.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">μ<sup>E<sub>e'</sub></sup><sub>e</sub> (xraylib)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">FL_line_attCS_ls = xlib_np.CS_Total(aN_ls, fl_energy);<br>shape (n_elem, n_lines) → lac tensor</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">cs[f'{ei}-{ej}'] = xraylib.CS_Total(Z_i, em_E[ej]);<br>used in cal_atten_3D → atten_xrf</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">τ<sup>E<sub>p</sub></sup><sub>l,e</sub></td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">XRF production cross-section × fluorescence yield × detector efficiency for line l of element e at probe energy E_p. Converts concentration × probe to emitted FL photons per unit volume.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>folded into full spectrum model M_e (Eq. 5)</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">detected_fl_unit_concentration — from MakeFLlinesDictionary_manual();<br>= τ × ε × Δs (ready-to-use scale factor)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>folded into H matrix during generate_H()</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fafafa;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">M<sub>e</sub>(E)</td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">Full XRF energy spectrum model for element e — xraylib emission peaks convolved with the Gaussian detector energy response function. Unique to Di et al. 2017; allows deconvolution of overlapping lines directly.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">M_e (|I|-vector per element, Eq. 5);<br>σ_det = Gaussian FWHM from detector specs</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;"><em>not modelled — per-line τ only;<br>element_lines_roi picks discrete channels</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not modelled</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">ε</td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">Total detector efficiency = solid angle fraction × quantum efficiency. Scales forward-modelled XRF to detected counts. For real data, absorbed into probe calibration; for simulation data, computed from detector geometry.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>folded into M_e normalization</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">det_solid_angle_ratio = π(d/2)² / 4π r²;<br>signal_attenuation_factor = 1.0 (simulation)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>folded into I_tot signal</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#fafafa;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">E<sub>l</sub></td>
      <td style="padding:6px 8px;"><span style="background:#f1f5f9;color:#475569;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Physical const.</span></td>
      <td style="padding:6px 10px; color:#374151;">Fluorescence line energies for each element (K, L, M shells). Used to look up μ at FL energy, compute τ, and identify which detector channels correspond to each element.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">E_l (emission energies in M_e)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">fl_all_lines_dic["fl_energy"];<br>from MakeFLlinesDictionary_manual()</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">em_E dict (keV) — from load_param();<br>em_eng list</td>
    </tr>

    <!-- ══ SECTION: Derived quantities ══ -->
    <tr style="background:#6ee7b730;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#065f46; letter-spacing:0.05em;">DERIVED QUANTITIES — COMPUTED FROM X DURING FORWARD PASS (RE-EVALUATED EACH ITERATION OR OUTER STEP)</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">A<sup>θ,p</sup><sub>v</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Probe attenuation reaching voxel v at angle θ, position p. Beer-Lambert cumulative product over all upstream voxels U<sup>v</sup><sub>θ,p</sub>. Eq. 5.6 / Eq. 6.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">A<sup>E,θ,τ</sup><sub>v</sub> via exact L<sup>θ,τ</sup><sub>v</sub> (Eq. 6);<br>frozen each outer iteration as A<sup>i</sup></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">att_exponent_acc_map — cumsum of lac_single × Δs;<br>→ attenuation_map_theta_flat = exp(−acc)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">x_ray_atten — from cal_atten_3D();<br>atten3D[elem] per angle, folded into H rows</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">lac<sub>v,l</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Linear attenuation coefficient map — product of μ̃ × X for each voxel. Intermediate used to compute both A (probe attenuation) and B (self-absorption). Frozen per outer iteration.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implicit (μ × W product in Eqs. 6–7)</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">lac = X_ap_rot.view(…) × FL_line_attCS_ls.view(…);<br>shape (n_elem, n_lines, n_voxel_batch, n_voxel)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">atten_fl from cal_atten_3D() — angle-by-angle;<br>μ × C integrated along FL path per voxel</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">B<sub>l,v</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Self-absorption survival probability for FL photons from voxel v to detector. exp(−Σ μ̃<sub>l</sub> × X × path). Averaged over n_d detector points. Eq. 5.9. Frozen per outer iteration in Di/Panpan.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">P<sup>θ,τ</sup><sub>v,e</sub> via Ω_v pyramid region (Eq. 7);<br>angle-independent spatial average</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">SA_theta — init_SA_theta();<br>mean(exp(−sum(att_exponent_voxel_sum)), axis=n_det)<br>shape (n_lines, n_voxel_batch)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">atten_xrf = exp(−xrf_atten × pix);<br>encoded as the attenuation weight in H rows</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">P<sub>v,v',d</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Path length through voxel v' on the FL beam from source voxel v to detector point d. Pre-computed geometry array, independent of X once rotation is set. Used in B calculation.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implicit in Ω_v pyramid (Eq. 7)</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">P_minibatch = P_array[…] from HDF5 file;<br>shape (n_det, 3, dia_len_n × batch_size × sample_n);<br>columns: [voxel_idx, element_idx, path_length]</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not stored explicitly — folded into H via bilinear interp weights (T matrix)</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">F̂<sup>R</sup><sub>θ,p,l</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Predicted XRF signal — output of the forward model. Sum over all voxels in the beam path of emission × self-absorption × probe × efficiency. Compared to D<sup>R</sup> in the loss. Eq. 5.10.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">F<sup>R</sup><sub>θ,τ</sub> (|I|-vector per beamlet, Eq. 8)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">y1_hat = fl_signal_SA_theta × det_solid_angle_ratio × signal_attenuation_factor;<br>from PPM.forward() output1</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">H_tot @ C_flat (during mlem_matrix Pf = p @ A_old)</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">F̂<sup>T</sup><sub>θ,p</sub></td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Predicted XRT optical density — cumulative sum of μ × X × Δs along the probe path. Compared to D<sup>T</sup> in the joint loss. Eq. 5.5.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">F<sup>T</sup><sub>θ,τ</sub> (Eq. 8)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">y2_hat = transmission_att_exponent_theta;<br>= att_exponent_acc_map[:, −1] from PPM.forward()</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not used — XRT term absent in BNL</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">H</td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Attenuated Radon system matrix — encodes geometry, rotation, and self-absorption in one dense matrix. Rows = detector pixels × angles; cols = voxels. Built once from ref3D_tomo using bilinear interpolation + attenuation weights.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implicit — forward model computed on-the-fly</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;"><em>implicit in PPM.forward() — no explicit H matrix</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">H_tot = generate_H(elem, ref3D_tomo, sli, angle_list);<br>shape (n_col × n_angles, n_row × n_col);<br>stored per slice per element via _generate_H_jit</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0fdf4;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">φ / Loss</td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Objective function — measures misfit between predicted and measured data. Minimised by the optimizer (Di/Panpan) or implicitly driven to zero by MLEM (BNL).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">φ(W) = Poisson LL(XRF) + β₁·Poisson LL(XRT);<br>Eq. 8</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">loss = XRF_loss + b1 × XRT_loss;<br>XRF_loss = MSELoss(y1_hat, y1_true);<br>XRT_loss = MSELoss(y2_hat, b2 × y2_true)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>implicit — MLEM converges when H@C ≈ I<br>(no explicit scalar loss computed)</em></td>
    </tr>

    <!-- ══ SECTION: Algorithm state ══ -->
    <tr style="background:#bfdbfe30;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#1e40af; letter-spacing:0.05em;">ALGORITHM STATE — INTERMEDIATE VARIABLES UPDATED DURING OPTIMIZATION</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0f9ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">X<sup>(n)</sup></td>
      <td style="padding:6px 8px;"><span style="background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Measured data</span></td>
      <td style="padding:6px 10px; color:#374151;">Current iterate of the reconstruction — X at iteration n. Updated by optimizer step (Panpan/Di) or MLEM multiplicative update (BNL). Written to checkpoint every N epochs.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">W<sup>i</sup> (frozen each outer iter except during TN inner solve)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">X — read/written to f_recon_grid.h5 each angle;<br>updated_minibatch = model.xp.detach()</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">A_old / A_new in mlem_matrix();<br>img2D → img_cor (corrected 2D slice)</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#f0f9ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">∂φ/∂X</td>
      <td style="padding:6px 8px;"><span style="background:#d1fae5;color:#065f46;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Derived f(X)</span></td>
      <td style="padding:6px 10px; color:#374151;">Gradient of loss w.r.t. X — used by Adam to update X. In Panpan computed automatically by PyTorch autograd through the PPM differentiable forward model.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">∇φ<sup>i</sup> — computed by TN Hessian-vector products inside inner solver</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">loss.backward() — PyTorch autograd;<br>model.xp.grad accumulated</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not used — MLEM is gradient-free;<br>update uses ratio H.T @ (I/Hc) / H.T @ 1</em></td>
    </tr>

    <!-- ══ SECTION: Hyperparameters ══ -->
    <tr style="background:#c4b5fd30;"><td colspan="6" style="padding:4px 10px; font-size:0.72rem; font-weight:700; color:#5b21b6; letter-spacing:0.05em;">HYPERPARAMETERS — USER-CHOSEN TUNING PARAMETERS (NOT PHYSICS-DERIVED)</td></tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">β<sub>1</sub> / λ</td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">XRT regularization weight — scales the XRT fidelity term relative to XRF. Breaks the self-absorption ambiguity. Di et al. use L-curve (optimal ≈ 1); Panpan sets manually.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">β₁ — chosen by L-curve (typically 1);<br>from Eq. 8</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">b1 — loss = XRF_loss + b1 × XRT_loss;<br>Eq. 5.12</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not applicable — no XRT term</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">β<sub>2</sub></td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">Data magnitude normalisation factor — rescales XRT data to be comparable to XRF counts. Di et al.: β₂=100 fixed; Panpan: b2 multiplies y2_true so XRT_loss matches XRF_loss magnitude.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">β₂ = 100 (fixed empirically)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">b2 — XRT_loss = MSE(y2_hat, b2 × y2_true)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not applicable</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">α (lr)</td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">Learning rate — step size for each Adam gradient update. Controls convergence speed vs. stability. Chosen by trial.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>implicit in TN backtracking line search (Algorithm 2) — no fixed lr</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">lr — tc.optim.Adam(model.parameters(), lr=lr)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>not applicable — MLEM is step-size-free</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">n<sub>epoch</sub></td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">Number of full passes over all angles. Panpan: ~60–80 epochs; Di et al.: ~3 outer iterations (much cheaper per outer step). BNL: n_iter MLEM steps (typically 20–50).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">n_outer ≈ 3 (Algorithm 1)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">n_epochs — save_every_n_epochs for checkpointing</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">n_iter — mlem_matrix(img2D, H, I, n_iter=n_iter)</td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">minibatch</td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">Number of 2D horizontal slabs updated per MPI rank per angle step. Larger = more GPU memory but fewer H5 writes. Trade-off between parallelism and I/O.</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;"><em>not explicit — full volume per outer iteration</em></td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">minibatch_size (rows per rank per angle);<br>n_batch = (n_z × n_x) / (n_ranks × minibatch_size)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;"><em>processed slice-by-slice (sli index);<br>parallelism via multiprocessing Pool</em></td>
    </tr>
    <tr style="border-bottom:1px solid #e2e8f0; background:#faf5ff;">
      <td style="padding:6px 10px; font-family:serif; font-style:italic; white-space:nowrap;">X<sup>(0)</sup></td>
      <td style="padding:6px 8px;"><span style="background:#ede9fe;color:#5b21b6;padding:2px 7px;border-radius:8px;font-size:0.73rem;font-weight:600;white-space:nowrap;">Hyperparameter</span></td>
      <td style="padding:6px 10px; color:#374151;">Initial guess for the reconstruction. Can be constant, random, or loaded from a prior run. Affects convergence speed; BNL uses a flat init (MLEM guarantees non-negativity).</td>
      <td style="padding:6px 10px; font-family:monospace; color:#065f46; font-size:0.76rem;">W<sup>0</sup> = zeros(n_voxels, n_elements)</td>
      <td style="padding:6px 10px; font-family:monospace; color:#2563eb; font-size:0.76rem;">ini_kind = 'const'/'rand'/'randn';<br>init_const, ini_rand_amp;<br>initialize_guess_3d()</td>
      <td style="padding:6px 10px; font-family:monospace; color:#7c3aed; font-size:0.76rem;">img2D — initial 2D slice passed to mlem_matrix();<br>often uniform or from standard FBP recon</td>
    </tr>

  </tbody>
</table>
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper builders
# ──────────────────────────────────────────────────────────────────────────────

def _section_header(num: str, title: str, icon_name: str, color: str):
    with ui.row().classes("items-center gap-2 mt-2"):
        ui.html(
            f'<div style="width:28px;height:28px;border-radius:50%;background:{color};'
            f'color:white;font-weight:700;font-size:0.9rem;display:flex;'
            f'align-items:center;justify-content:center;">{num}</div>'
        )
        ui.icon(icon_name, size="1.3rem").style(f"color:{color}")
        ui.label(title).classes("text-lg font-semibold text-slate-800")


def _ref_badge(label: str, color: str, bg: str):
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:12px;'
        f'background:{bg};color:{color};font-size:0.75rem;font-weight:600;'
        f'margin-left:6px;">{label}</span>'
    )


def _method_card(title: str, badge: str, badge_color: str, badge_bg: str,
                 items: list[str], code: str, color: str):
    with ui.card().classes("flex-1").style(f"border-top:4px solid {color}; min-width:280px;"):
        with ui.column().classes("p-2 gap-3"):
            ui.html(
                f'<div class="font-semibold text-slate-700" style="font-size:0.95rem;">'
                f'{title}'
                f'<span style="display:inline-block;padding:1px 8px;border-radius:12px;'
                f'background:{badge_bg};color:{badge_color};font-size:0.75rem;font-weight:600;'
                f'margin-left:6px;">{badge}</span></div>'
            )
            for item in items:
                ui.html(f'<div style="font-size:0.87rem; color:#475569; line-height:1.55;">{item}</div>')
            with ui.expansion("Show code", icon="code").classes("w-full"):
                ui.code(code, language="python").classes("w-full text-xs")


# ──────────────────────────────────────────────────────────────────────────────
# Page factory
# ──────────────────────────────────────────────────────────────────────────────

def create_method_explanation_page():
    """Render the Method Explanation (BNL) comparison page."""

    with ui.column().classes("page-content gap-6"):

        # ── Title ───────────────────────────────────────────────────────────
        with ui.row().classes("items-center gap-3 mt-2"):
            ui.icon("menu_book", size="2.2rem").classes("text-red-600")
            with ui.column().classes("gap-0"):
                ui.label("Method Explanation").classes("text-2xl font-bold text-slate-800")
                ui.label(
                    "Comparing three self-absorption correction approaches"
                ).classes("text-sm text-slate-500")

        # ── Reference banner ─────────────────────────────────────────────────
        with ui.card().classes("w-full").style("background:#eff6ff; border:1px solid #bfdbfe;"):
            with ui.column().classes("p-1 gap-2"):
                with ui.row().classes("items-start gap-3"):
                    ui.icon("article", size="1.3rem").classes("text-blue-600 mt-0.5")
                    with ui.column().classes("gap-1"):
                        ui.label("References").classes("font-semibold text-blue-800 text-sm")
                        ui.html(
                            '<div style="font-size:0.85rem; color:#1e40af; line-height:1.65;">'
                            "<strong>Di et al. 2017 (ANL):</strong> Di, Z.W., Chen, S., Hong, Y.P., et al., "
                            "<em>\"Joint reconstruction of x-ray fluorescence and transmission tomography,\"</em> "
                            "<strong>Opt. Express</strong> 25(12), 13107–13124 (2017). "
                            "Poisson likelihood + truncated Newton + full spectrum model. "
                            "<em>No code in this repo — theory reference only.</em>"
                            "<br>"
                            "<strong>Reconstruction (Panpan):</strong> Huang, P. (2022). "
                            "<em>PhD Thesis</em>, Chapter 5 — Self-absorption correction in "
                            "x-ray fluorescence tomography using automatic differentiation. "
                            "Extends Di et al. with MSE loss + Adam + MPI. "
                            "Implemented in <code>src/reconstruction/</code>."
                            "<br>"
                            "<strong>FL Correction (BNL):</strong> Ge, M., Huang, X., Yan, H. et al., "
                            "<em>\"Three-dimensional imaging of grain boundaries via quantitative "
                            "fluorescence X-ray tomography analysis,\"</em> "
                            "<strong>Commun. Mater.</strong> 3, 37 (2022). "
                            "Implemented in <code>src/fl_correction/</code>."
                            "</div>"
                        )

        # ════════════════════════════════════════════════════════════════════
        # Section 1 — The Common Problem
        # ════════════════════════════════════════════════════════════════════
        _section_header("1", "The Common Problem", "biotech", "#dc2626")

        with ui.card().classes("w-full"):
            with ui.column().classes("p-2 gap-4"):
                ui.html(
                    '<p style="color:#334155; line-height:1.75; font-size:0.93rem;">'
                    "In X-ray fluorescence (XRF) tomography, a focused X-ray probe scans a "
                    "rotating sample and records element-specific fluorescence at each angle. "
                    "Standard reconstruction algorithms assume two things that break down for "
                    "dense or thick samples:"
                    "</p>"
                )
                with ui.row().classes("gap-4 flex-wrap w-full"):
                    with ui.card().classes("flex-1").style("border-left:4px solid #dc2626; min-width:250px;"):
                        with ui.column().classes("p-2 gap-1"):
                            ui.label("Assumption 1 — Unattenuated probe").classes("font-semibold text-slate-700 text-sm")
                            ui.html(
                                '<p style="color:#475569; font-size:0.87rem; line-height:1.55;">'
                                "The probe beam intensity I<sub>0</sub> is assumed constant throughout "
                                "the sample. In reality the beam is attenuated (Beer-Lambert) as "
                                "it travels deeper — voxels far from the entry face receive a "
                                "weaker probe."
                                "</p>"
                            )
                    with ui.card().classes("flex-1").style("border-left:4px solid #d97706; min-width:250px;"):
                        with ui.column().classes("p-2 gap-1"):
                            ui.label("Assumption 2 — No self-absorption").classes("font-semibold text-slate-700 text-sm")
                            ui.html(
                                '<p style="color:#475569; font-size:0.87rem; line-height:1.55;">'
                                "XRF photons emitted inside the sample are assumed to escape freely. "
                                "In reality they are re-absorbed by surrounding material on the "
                                "way to the detector — the deeper the source voxel, the larger "
                                "the bias."
                                "</p>"
                            )
                ui.html(
                    '<div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; '
                    'padding:10px 14px; font-size:0.87rem; color:#7f1d1d; line-height:1.6;">'
                    "<strong>Consequence:</strong> Without correction, reconstructed elemental "
                    "densities are systematically underestimated — especially for low-Z elements "
                    "(which emit lower-energy photons that are more easily absorbed) and for "
                    "voxels near the centre of the sample."
                    "</div>"
                )
                ui.html(_SETUP_SVG)
                ui.html(_SELF_ABSORPTION_SVG)

        # ════════════════════════════════════════════════════════════════════
        # Section 2 — Common Forward Model
        # ════════════════════════════════════════════════════════════════════
        _section_header("2", "Common Physics — The Forward Model  (Eqs. 5.5–5.10)", "functions", "#0891b2")

        with ui.card().classes("w-full"):
            with ui.column().classes("p-2 gap-4"):
                ui.html(
                    '<p style="color:#334155; line-height:1.75; font-size:0.93rem;">'
                    "<strong>Both methods share the same physics forward model</strong> — "
                    "they differ only in how they invert it. Given the current estimate of the "
                    "3D density X, the forward model predicts the XRF signal at each angle θ "
                    "and probe position p through four sequential steps:"
                    "</p>"
                )
                ui.html(_FORWARD_SVG)

                # The four equations as cards
                for letter, title, desc, eq in [
                    ("A", "XRT optical density  (Eq. 5.5)",
                     "The probe beam is attenuated as it crosses the sample. The optical density "
                     "(log-transmission) is just a linear sum of μ·X·Δs along the beam path — "
                     "easy to compute and used as a joint constraint in Panpan's method.",
                     _EQ_XRT),
                    ("B", "Probe attenuation A  (Eq. 5.6 + 5.7)",
                     "Before reaching voxel v, the probe has been attenuated by all upstream "
                     "voxels U<sup>v</sup><sub>θ,p</sub>. The attenuation is a Beer-Lambert "
                     "exponential, accumulated along the beam direction.",
                     _EQ_PROBE_ATT),
                    ("C", "Emission at voxel v  (Eq. 5.8)",
                     "Each voxel generates XRF photons proportional to its density X<sub>v,e</sub>, "
                     "the mass production cross-section τ<sup>Ep</sup><sub>l</sub> (from xraylib), "
                     "and the attenuated probe intensity I<sub>0</sub>·A.",
                     _EQ_EMISSION),
                    ("D", "Self-absorption B  (Eq. 5.9)",
                     "XRF photons must travel from v to the detector. Each voxel v' they cross "
                     "contributes μ̃<sup>l</sup><sub>e</sub>·X<sub>v',e</sub>·P<sub>v,v',d</sub> "
                     "to the absorption exponent. B is averaged over n<sub>d</sub> detector "
                     "sampling points to account for finite detector size.",
                     _EQ_SA),
                ]:
                    with ui.card().classes("w-full").style(f"border-left:4px solid #0891b2;"):
                        with ui.column().classes("p-2 gap-2"):
                            ui.html(
                                f'<div style="font-weight:600; color:#0e7490; font-size:0.9rem;">'
                                f'<span style="background:#ecfeff; border-radius:4px; '
                                f'padding:1px 7px; margin-right:6px; font-weight:700;">{letter}</span>'
                                f'{title}</div>'
                            )
                            ui.html(f'<p style="color:#475569; font-size:0.88rem; line-height:1.6; margin:0;">{desc}</p>')
                            ui.html(eq)

                ui.separator()
                ui.html(
                    '<div style="font-size:0.87rem; color:#334155; font-weight:600; '
                    'margin-bottom:4px;">Full XRF signal — Eq. 5.10 (sum over all voxels in beam path)</div>'
                )
                ui.html(_EQ_XRF)

        # ════════════════════════════════════════════════════════════════════
        # Section 3 — The Differences
        # ════════════════════════════════════════════════════════════════════
        _section_header("3", "How They Differ — Inverting the Forward Model", "compare_arrows", "#059669")

        with ui.card().classes("w-full"):
            with ui.column().classes("p-2 gap-4"):
                ui.html(
                    '<p style="color:#334155; line-height:1.75; font-size:0.93rem;">'
                    "The problem is: given the measured D<sup>R</sup> (XRF) and D<sup>T</sup> "
                    "(XRT), find X such that F̂<sup>R</sup>(X) ≈ D<sup>R</sup>. "
                    "Both methods solve this iteratively — but with completely different "
                    "mathematical strategies:"
                    "</p>"
                )
                ui.html(_COMPARE_SVG)
                with ui.row().classes("gap-4 flex-wrap w-full"):
                    _method_card(
                        "Di et al. 2017 (ANL)", "Opt. Express 2017", "#065f46", "#d1fae5",
                        [
                            "&#x2022; Models the <strong>full XRF energy spectrum</strong> M<sub>e</sub> "
                            "by convolving each element's fluorescence peaks with the detector's Gaussian "
                            "energy response (Eq. 5). Separates overlapping elemental lines implicitly.",
                            "&#x2022; Self-absorption P<sup>θ,τ</sup><sub>v,e</sub> (Eq. 7) is "
                            "approximated over a pyramid region Ω<sub>v</sub> subtended by the detector "
                            "footprint — angle-independent and averaged spatially.",
                            "&#x2022; <strong>Poisson log-likelihood objective</strong> (Eq. 8) — "
                            "statistically optimal for photon-counting data. Scaling: β<sub>1</sub> "
                            "balances XRF/XRT (chosen by L-curve, typically β<sub>1</sub>=1); "
                            "β<sub>2</sub>=100 normalises data magnitudes.",
                            "&#x2022; <strong>Alternating direction</strong>: freeze A<sup>i</sup>, "
                            "P<sup>i</sup> → solve inner subproblem (Eq. 9) with truncated Newton (TN) "
                            "using 52 inner iterations and backtracking line search. Converges in ~3 "
                            "outer iterations.",
                        ],
                        "# Di et al. 2017 — Algorithm 1 (pseudocode; no code in this repo)\n"
                        "W = zeros(n_voxels, n_elements)       # W^0 initial guess\n"
                        "for i in range(n_outer):              # converges in ~3 iters\n"
                        "    A_i = compute_probe_atten(W)      # Eq. 6 — freeze\n"
                        "    P_i = compute_self_absorb(W)      # Eq. 7 — freeze (Ω_v pyramid)\n"
                        "    # Solve Poisson LL subproblem φ^i(W) with TN\n"
                        "    W = truncated_newton(phi_i, W_init=W, k=52)\n"
                        "    W = backtracking_line_search(W)   # Algorithm 2\n"
                        "# Output: W* (g/cm^3) — 3D elemental map",
                        "#059669",
                    )
                    _method_card(
                        "Reconstruction (Panpan)", "src/reconstruction/", "#1e40af", "#dbeafe",
                        [
                            "&#x2022; Defines a <strong>differentiable</strong> forward model "
                            "(PyTorch <code>nn.Module</code> — the <code>PPM</code> class) "
                            "so that PyTorch autograd can compute ∂φ/∂X automatically.",
                            "&#x2022; Minimises the joint XRF + XRT loss (Eq. 5.12) using the "
                            "<strong>Adam optimizer</strong>.",
                            "&#x2022; <strong>MPI parallelism</strong>: 3D volume split into "
                            "horizontal minibatches across ranks — each rank updates its slabs "
                            "simultaneously.",
                            "&#x2022; Self-absorption term B<sub>l,v</sub> is frozen from the "
                            "previous update and treated as a constant (Eq. 5.11) so each "
                            "slice can be updated independently.",
                        ],
                        "# XRF_tomography.py — inner loop per (epoch, angle, minibatch)\n"
                        "model = PPM(...)               # differentiable forward model\n"
                        "y1_hat, y2_hat = model()       # Eq. 5.10 + 5.5\n"
                        "XRF_loss = MSE(y1_hat, y1_true)\n"
                        "XRT_loss = MSE(y2_hat, b2*y2_true)\n"
                        "loss = XRF_loss + b1 * XRT_loss  # Eq. 5.12\n"
                        "optimizer.zero_grad()\n"
                        "loss.backward()                # PyTorch autograd\n"
                        "optimizer.step()               # Adam update",
                        "#2563eb",
                    )
                    _method_card(
                        "FL Correction (BNL)", "src/fl_correction/", "#5b21b6", "#ede9fe",
                        [
                            "&#x2022; Builds an <strong>explicit H-matrix</strong> "
                            "(attenuated Radon system matrix) that encodes both geometry "
                            "and attenuation. The forward model becomes the linear equation H·C = I.",
                            "&#x2022; Inverts H·C = I using <strong>MLEM</strong> "
                            "(Maximum Likelihood EM) — a multiplicative update rule that "
                            "guarantees non-negativity.",
                            "&#x2022; <strong>Slice-by-slice</strong>: the 3D volume is a "
                            "stack of independent 2D slices, each solved separately. "
                            "Parallelism is across slices (CPU multiprocess or GPU CUDA kernel).",
                            "&#x2022; No XRT term — correction is XRF-only.",
                        ],
                        "# FL_correction_core.py — per element per slice\n"
                        "# Build attenuated Radon matrix H\n"
                        "atten = cal_atten_with_direction(img4D, cs, param)\n"
                        "H = generate_H(elem, ref3D_tomo, sli, angle_list, file_path=...)\n"
                        "I = generate_I(elem, ref3D_tomo, sli, angle_list, file_path=...)\n"
                        "\n"
                        "# MLEM iterative solve  (GPU: mlem_cuda via numba @cuda.jit)\n"
                        "C_corrected = mlem_matrix(img2D, H, I, n_iter=n_iter)\n"
                        "# C_new = C * (H.T @ (I / H@C)) / H.T @ ones",
                        "#7c3aed",
                    )

                ui.html(
                    '<div style="font-size:0.87rem; font-weight:600; color:#334155; margin-top:4px; margin-bottom:2px;">'
                    'Objective functions — the key algorithmic difference:'
                    '</div>'
                )
                with ui.row().classes("gap-3 flex-wrap w-full"):
                    with ui.column().classes("flex-1").style("min-width:280px;"):
                        ui.html(
                            '<div style="font-size:0.8rem; font-weight:600; color:#065f46; margin-bottom:4px;">'
                            'Di et al. 2017 — Poisson log-likelihood (Eq. 8)</div>'
                        )
                        ui.html(_EQ_POISSON)
                    with ui.column().classes("flex-1").style("min-width:280px;"):
                        ui.html(
                            '<div style="font-size:0.8rem; font-weight:600; color:#1e40af; margin-bottom:4px;">'
                            'Panpan 2022 — MSE joint loss (Eq. 5.12)</div>'
                        )
                        ui.html(_EQ_LOSS)
                ui.html(
                    '<div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px; '
                    'padding:10px 14px; font-size:0.87rem; color:#0c4a6e; line-height:1.65;">'
                    "<strong>Why does the joint XRT constraint help?</strong> &nbsp;"
                    "XRF is element-specific but suffers from self-absorption ambiguity — a thick "
                    "dense sample looks the same as a thin one with no self-absorption. "
                    "XRT measures total optical density independently of element identity, breaking "
                    "this ambiguity. Di et al. showed ~3 outer iterations suffice; Panpan showed "
                    "XRT cuts epoch count from ~80 to ~60 (thesis Fig. 5.6). "
                    "BNL FL correction avoids XRT entirely by assuming the attenuation is known "
                    "from a prior reference reconstruction."
                    "</div>"
                )

        # ════════════════════════════════════════════════════════════════════
        # Section 4 — Variable Reference
        # ════════════════════════════════════════════════════════════════════
        _section_header("4", "Variable Reference — Three-Way Notation Comparison", "table_chart", "#475569")

        with ui.card().classes("w-full"):
            with ui.column().classes("p-2"):
                ui.html(_VARIABLE_TABLE)

        # bottom padding
        ui.element("div").classes("h-8")
