import sys, os, runpy

app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG Data Analyst Agent")
sys.path.insert(0, app_dir)
runpy.run_path(os.path.join(app_dir, "app.py"), run_name="__main__")
