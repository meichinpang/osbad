# Test osbad installation:
# osbad for open-source benchmark of anomaly detection
from importlib.metadata import version
import osbad

def main():
  print("Hello from osbad!")
  osbad_current_version = version("osbad")
  print(f"osbad current version: {osbad_current_version}")
  print(f"OSBAD package installation is successful!")

if __name__ == "__main__":
  main()
