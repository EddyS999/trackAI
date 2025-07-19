"""
Ce fichier contient les configurations de l'utilisateur.
"""

import os
from pathlib import Path
from sys import platform



username = "akirastroworld"  


target_python_link_path = Path(os.path.expanduser("~")) / "Documents" / "TMInterface" / "Plugins" / "Python_Link.as"


trackmania_base_path = Path(os.path.expanduser("~")) / "Documents" / "TmForever"

base_tmi_port = 8478





windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"

windows_TMLoader_profile_name = "default"
