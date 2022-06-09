from setuptools import setup

setup(
    name="smash",
    version="2.0.0",
    description="Spatially distributed Modelling and ASsimilation for Hydrology",
    url="/",
    author="INRAE/Hydris Hydrologie",
    packages=["smash", "smash.core", "smash.wrapping", "smash.io"],
    install_requires=[
        "numpy>=1.21.2",
        "pandas>=1.4.2"
    ],
    zip_safe=False,
)
