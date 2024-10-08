from ase import db
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import read, Trajectory
from ase.optimize import QuasiNewton
from ase.parallel import world, parprint
from gpaw import GPAW
from gpaw import setup_paths
from gpaw.cluster import Cluster
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.utilities import h2gpts
import os
import shutil
import time

# Setup paths for GPAW setups
setup_paths.insert(0, './setups')

# Create output directory if it doesn't exist
OUTPUT_DIR = './output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def setup_calculator(atoms, functional, correction):
    """
    Setup the GPAW calculator based on the functional and correction method.

    Parameters:
    atoms (ase.Atoms): ASE atoms object.
    functional (str): The density functional to use for the calculation.
    correction (str): The correction method to apply ('SIC', 'U', or None).

    Returns:
    GPAW: A GPAW calculator instance configured with the provided parameters.
    """
    name = atoms.get_chemical_formula(mode='hill')

    # Set magnetic moments based on the formula
    magmom_map = {'O2': [1, 1], 'H': [1], 'O': [2]}
    if name in magmom_map:
        atoms.set_initial_magnetic_moments(magmoms=magmom_map[name])

    # Determine Hund's rule and spin polarization
    hund = True if name in ['H', 'O'] else False
    spinpol = True if name in ['O2', 'H', 'O'] else False
    nbands = -1 if name == 'H' else -4

    calc_config = {
        'xc': functional,
        'nbands': nbands,
        'basis': 'dzp',
        'mode': {'name': 'pw', 'ecut': 600, 'force_complex_dtype': True},
        'gpts': h2gpts(0.12, atoms.get_cell(), idiv=4),
        'parallel': {'augment_grids': True, 'sl_auto': True},
        'hund': hund,
        'spinpol': spinpol,
        'txt': f'{name}_{functional}_{correction}.txt'
    }

    return GPAW(**calc_config)

def run(functional, correction, molecule_name):
    """
    Run calculations for a given molecule with specified functional and correction.

    Parameters:
    functional (str): The density functional to use for the calculation.
    correction (str): The correction method to apply.
    molecule_name (str): The name of the molecule to run calculations on.

    Returns:
    None
    """
    start_time = time.time()

    # Read or build the molecule structure
    if os.path.exists(f'./output/{molecule_name}_{functional}_{correction}.traj'):
        atoms = read(f'./output/{molecule_name}_{functional}_{correction}.traj')
    else:
        atoms = Cluster(molecule(molecule_name))
        adjust_cell(atoms, 6, h=0.12)
        atoms.translate([0.01, 0.03, 0.05])
        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.pbc = (False, False, False)

    # Setup GPAW calculator
    calc = setup_calculator(atoms, functional, correction)
    atoms.calc = calc

    # Apply correction based on method
    if correction == 'SIC':
        calc_config.update({
            'occupations': {'name': 'fixed-uniform'},
            'eigensolver': FDPWETDM(localizationtype='PM_PZ',
                                    functional={'name':'PZ-SIC','scaling_factor': (1, 1)},
                                    grad_tol_pz_localization=1.0e-6,
                                   ),
            'mixer': {'backend': 'no-mixing'},
            'symmetry': 'off',
        })
    elif correction == 'U':
        calc_config.update({
            'setups': {'O':':p,6.8,0'},
        })

    # Run geometry optimization
    dyn = QuasiNewton(atoms, trajectory=None, maxstep=0.05)
    tra = Trajectory(f'{molecule_name}_{functional}_{correction}.traj', 'a', atoms)
    dyn.attach(tra.write, interval=1)
    dyn.run(fmax=0.01)

    # Get reference energy
    ref_energy = atoms.get_potential_energy()
    parprint(f'{functional} energy: {ref_energy}')

    # Move files to output directory
    if world.rank == 0:
        shutil.move(f'{molecule_name}_{functional}_{correction}.txt',
                    os.path.join(OUTPUT_DIR, f'{molecule_name}_{functional}_{correction}.txt'))
        shutil.move(f'{molecule_name}_{functional}_{correction}.traj',
                    os.path.join(OUTPUT_DIR, f'{molecule_name}_{functional}_{correction}.traj'))

    end_time = time.time()
    parprint(f'Time: {end_time - start_time:.2f} s')

    # List of GGA functionals for non-self-consistent energies
    df_list = [
        # GGA xc
        "GGA_XC_B97_GGA1",
        "GGA_XC_HCTH_407",
        "GGA_XC_HLE16",
        "GGA_XC_KT3",
        "GGA_XC_MPWLYP1W",
        "GGA_XC_NCAP",
        "GGA_XC_PBE1W",
        "GGA_XC_PBELYP1W",
        "GGA_XC_VV10",
        "GGA_XC_XLYP",
        # asymmetric xc
        "GGA_X_B88+GGA_C_ACGGAP",       #acGGA+
        "GGA_X_B88+GGA_C_LYP",          #BLYP
        "GGA_X_B88+GGA_C_OP_B88",       #BOP
        "GGA_X_B88+GGA_C_P86",          #BP86
        "GGA_X_B88+GGA_C_PBE",          #BPBE
        "GGA_X_B88+GGA_C_P86VWN",       #BP86VWN
        "GGA_X_CAP+GGA_C_P86",          #CAP
        "GGA_X_ITYH_PBE+GGA_C_PBE",     #ITYH-PBE
        "GGA_X_HTBS+GGA_C_PBE",         #HTBS-PBE
        "GGA_X_LSPBE+GGA_C_PBE",        #LSPBE
        "GGA_X_LSRPBE+GGA_C_PBE",       #LSRPBE
        "GGA_X_MPBE+GGA_C_PBE",         #mPBE
        "GGA_X_MPW91+GGA_C_PW91",       #mPW91
        "GGA_X_NCAP+GGA_C_P86",         #NCAP
        "GGA_X_NCAPR+GGA_C_P86",        #NCAPR
        "GGA_X_OPTX+GGA_C_LYP",         #OLYP
        "GGA_X_OPTX+GGA_C_PBE",         #OPBE
        "GGA_X_PBE+GGA_C_ACGGA",        #acGGA
        "GGA_X_PBE+GGA_C_OP_PBE",       #PBE-OP
        "GGA_X_RPBE+GGA_C_PBE",         #RPBE
        "GGA_X_PBEINT+GGA_C_ZVPBEINT",  #ZVPBEint
        "GGA_X_PBE_SOL+GGA_C_ZVPBESOL", #ZVPBEsol
        "GGA_X_SOGGA+GGA_C_PBE",        #SOGGA
        "GGA_X_SOGGA11+GGA_C_PBE",      #SOGGA11
        # symmetric xc
        "GGA_X_AM05+GGA_C_AM05",
        "GGA_X_APBE+GGA_C_APBE",
        "GGA_X_CHACHIYO+GGA_C_CHACHIYO",
        "GGA_X_GAM+GGA_C_GAM",
        "GGA_X_N12+GGA_C_N12",
        "GGA_X_PBE+GGA_C_PBE",
        "GGA_X_PBEFE+GGA_C_PBEFE",
        "GGA_X_PBEINT+GGA_C_PBEINT",
        "GGA_X_PBE_MOL+GGA_C_PBE_MOL",
        "GGA_X_PBE_SOL+GGA_C_PBE_SOL",
        "GGA_X_PW91+GGA_C_PW91",
        "GGA_X_Q2D+GGA_C_Q2D",
        "GGA_X_RGE2+GGA_C_RGE2",
        "GGA_X_SOGGA11+GGA_C_SOGGA11",
        "GGA_X_XPBE+GGA_C_XPBE",
        # MGGA xc
        "MGGA_XC_B97M_V",
        "MGGA_XC_CC06",
        "MGGA_XC_HLE17",
        "MGGA_XC_LP90",
        "MGGA_XC_TPSSLYP1W",
        "MGGA_X_M06_L+MGGA_C_M06_L",       #M06-L
        "MGGA_X_M11_L+MGGA_C_M11_L",       #M11-L
        "MGGA_X_MN12_L+MGGA_C_MN12_L",     #MN12-L
        "MGGA_X_MN15_L+MGGA_C_MN15_L",     #MN15-L
        "MGGA_X_PKZB+MGGA_C_PKZB",         #PKZB
        "MGGA_X_REVTPSS+MGGA_C_REVTPSS",   #revTPSS
        "MGGA_X_RTPSS+MGGA_C_TPSS",        #RTPSS
        "MGGA_X_R2SCAN+MGGA_C_R2SCAN",     #R2SCAN
        "MGGA_X_R2SCANL+MGGA_C_R2SCANL",   #R2SCANL
        "MGGA_X_REVM06_L+MGGA_C_REVM06_L", #REVM06-L
        "MGGA_X_REVTM+MGGA_C_REVTM",       #REVTM
        "MGGA_X_SCAN+MGGA_C_SCAN",         #SCAN
        "MGGA_X_SCANL+MGGA_C_SCANL",       #SCANL
        "MGGA_X_TM+MGGA_C_TM",             #TM
        "MGGA_X_TPSS+MGGA_C_TPSS",         #TPSS
        "MGGA_X_GVT4+MGGA_C_VSXC",         #VSXC
    ]

    nsc_energies = {}

    # Calculate non-self-consistent energies
    for df in df_list:
        try:
            gga_delta_energy = calc.get_xc_difference(df)
            parprint(f'{df} nsc energy: {ref_energy + gga_delta_energy}')
            nsc_energies[df] = ref_energy + gga_delta_energy
        except Exception as e:
            parprint(f'Error with {df}: {e}')
            continue

    # Write data to the database
    try:
        database = db.connect('DF-and-water-formation.db')
    except FileNotFoundError:
        database = db.connect('DF-and-water-formation.db')

    database.write(atoms, relaxed=True, density_functional=functional,
                   correction=correction, data={'nsc_energies': nsc_energies})

def main():
    molecules = ['H', 'O', 'O2', 'H2', 'H2O', 'H2O2']
    functionals = ['HSE06']  # Add other functionals as needed
    corrections = ['']  # Add other corrections as needed

    if not os.path.exists('output'):
        os.makedirs('output')

    # Run DFT calculations for each molecule, functional, and correction
    for molecule in molecules:
        for functional in functionals:
            for correction in corrections:
                if functional in ['PBE', 'HSE06'] and correction == '':
                    parprint(f'Running DFT for {functional} {correction} {molecule}')
                    run(functional, correction, molecule)
                else:
                    parprint('Check input values')

if __name__ == "__main__":
    main()
