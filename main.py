from importlib.metadata import version
import osbad

def main():
    print("Hello from osbad, running inside a Docker container! 🐳")
    osbad_current_version = version("osbad")
    print(f"osbad current version: {osbad_current_version}")
    print("OSBAD package installation is successful!")

if __name__ == "__main__":
    main()