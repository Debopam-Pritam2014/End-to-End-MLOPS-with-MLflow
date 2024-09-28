from setuptools import setup,find_packages

E_DOT="-e ."

def get_requirements(file_path:str)->list:
    """
    This function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[requirement.replace("\n","") for requirement in requirements]
        if E_DOT in requirements:
            requirements.remove(E_DOT)
    return requirements



setup(
    name="mlproject",
    version="0.0.1",
    author="Pritam",
    author_email="letsdecode2014@gmail.com",
    packages=find_packages(),
    # install_requires=["pandas","numpy"],
    install_requires=get_requirements("requirements.txt"),

)
