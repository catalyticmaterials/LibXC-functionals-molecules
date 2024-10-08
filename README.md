# LibXC-functionals-molecules
Example of using LibXC to evaluate energies for gas reactions

Step 1. Run optimization of H₂, O₂, H₂O and H₂O₂ using HSE06. Evaluate HSE06 energy for these molecules, H, and O. Evaluate non-self consistent energies for a list of density functionals. Write evertying to an ASE database.

Step 2. Run single point for a list of given functionals. Evaluate non-self consistent energies for a list of density functionals. Write evertying to an ASE database.

Step 3. Analyse self-consistent results taking the HSE06 results as the reference.

Step 4. Analyse non-self-consistent results for taking each functional as the reference.

The expected results looks like this

Functional-Correction: HSE06

Mean Deviation: -0.09 eV

Mean Absolute Deviation: 0.09 eV

------------------------------

Functional-Correction: GGA_XC_KT3

Mean Deviation: -0.07 eV

Mean Absolute Deviation: 0.14 eV

------------------------------

Functional-Correction: GGA_X_N12+GGA_C_N12

Mean Deviation: -0.15 eV

Mean Absolute Deviation: 0.15 eV

------------------------------

Functional-Correction: GGA_XC_VV10

Mean Deviation: -0.14 eV

Mean Absolute Deviation: 0.21 eV

------------------------------

Functional-Correction: RPBE

Mean Deviation: -0.08 eV

Mean Absolute Deviation: 0.25 eV

------------------------------

Functional-Correction: PBE

Mean Deviation: -0.28 eV

Mean Absolute Deviation: 0.29 eV

------------------------------
