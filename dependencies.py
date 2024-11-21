import pkg_resources

def save_dependencies_to_file(filename="dependencies.txt"):
    """
    Save all libraries, dependencies, and their versions to a text file.
    """
    with open(filename, "w") as file:
        installed_packages = pkg_resources.working_set
        for package in sorted(installed_packages, key=lambda x: x.project_name.lower()):
            file.write(f"{package.project_name}=={package.version}\n")
    print(f"Dependencies saved to {filename}")

if __name__ == "__main__":
    save_dependencies_to_file()
