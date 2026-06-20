"""
Default Streamlit Cloud entry point.

Streamlit Community Cloud auto-detects `streamlit_app.py` as the main module,
so a fresh deployment of this repository needs no configuration. It launches the
v4 solar-system map (all Sun-planet Lagrange points superposed). The core method
and paper are v3 (`solar_system_dmm_v3.py`, memory-as-dissipation); run that
file directly to use the discovery app.
"""
import runpy

runpy.run_path("solar_system_dmm_v4.py", run_name="__main__")
