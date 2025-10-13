from pathlib import Path
import sys
import sys


def find_path():
    current_path = Path.cwd()
    # Navigate to the parent directory until reaching 'Multifirefly-Project'
    if 'Multifirefly-Project' in current_path.parts:
        while current_path.name != 'Multifirefly-Project':
            current_path = current_path.parent
        # %cd $current_path
        print("Changed the directory to 'Multifirefly-Project'.")
    else:
        print("Multifirefly-Project directory not found in the current path; try adding Multifirefly-Project to the end of the path.")
        current_path = Path.cwd() / 'Multifirefly-Project'

    # add methods to the path
    path = current_path / 'multiff_analysis' / 'methods'
    if not (str(path) in sys.path):
        sys.path.append(str(path))
        print(f"Added {str(path)} to the path.")
    return current_path
