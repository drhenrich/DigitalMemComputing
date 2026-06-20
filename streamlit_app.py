"""
Default Streamlit Cloud entry point.

Streamlit Community Cloud auto-detects `streamlit_app.py` as the main module,
so a fresh deployment of this repository needs no configuration. It launches the
v3 app (`solar_system_dmm_v3.py`, memory-as-dissipation) — the core method and
the subject of the paper. The v4 solar-system map and the v5 stability
simulation can be run directly (`streamlit run solar_system_dmm_v4.py` / `_v5`).
"""
import runpy

runpy.run_path("solar_system_dmm_v3.py", run_name="__main__")
