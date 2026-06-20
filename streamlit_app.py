"""
Default Streamlit Cloud entry point.

Streamlit Community Cloud auto-detects `streamlit_app.py` as the main module,
so a fresh deployment of this repository needs no configuration. It launches the
current app, v3 (memory-as-dissipation). To pin an existing deployment to v3,
set its "Main file path" to `streamlit_app.py` (or `solar_system_dmm_v3.py`).
"""
import runpy

runpy.run_path("solar_system_dmm_v3.py", run_name="__main__")
