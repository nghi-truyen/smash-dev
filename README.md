# SMASH - Spatially distributed Modelling and ASsimilation for Hydrology
[![Build Status](https://img.shields.io/badge/docs-public-brightgreen)](https://gitlab.irstea.fr/francois.colleoni/smash/)

## Compilation Instructions

1.  Clone the SMASH repository from GitLab.
    ```bash
    git clone https://gitlab.irstea.fr/francois.colleoni/smash.git
    ```
2.  Compile all programs, modules and libraries.
    ```bash
    make
    ```
    
# Developer notes:

## Adjoint Generation

1. Clean the code
  ```bash
  make clean
  ```
2. Generate tapaneade files (i.e. adjoint and tangent linear models)
  ```bash 
  make tap
  ```
3. Compile again
  ```bash
  make
  ```
  
## Develop in debug mode

Compile with debug mode.
  ```bash	
  make debug
  ```
