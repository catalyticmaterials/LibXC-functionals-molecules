from ase import db
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.io import read, Trajectory
from ase.optimize import QuasiNewton
from ase.parallel import world, parprint
from gpaw import GPAW, PW
from gpaw import setup_paths
from gpaw.cluster import Cluster
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.utilities import h2gpts
import numpy as np
import os
import shutil
import time

setup_paths.insert(0, './setups')

def run_dft(functional, correction, molecule_name):
    start_time = time.time()

    # read variables
    name = molecule_name
    func = functional
    sufx = correction

    # set parameters
    hund = True if name in ['H', 'O'] else False
    spinpol = True if name in ['O2', 'H', 'O'] else False

    if os.path.exists(f'./output/{name}_{func}_{sufx}.traj'):
        atoms =  read(f'./output/{name}_{func}_{sufx}.traj')
    else:
        atoms = Cluster(molecule(f'{molecule_name}'))
        #atoms.minimal_box(border=8, h=0.12, multiple=4)
        adjust_cell(atoms, 8, h=0.12)
        atoms.translate([0.01, 0.03, 0.05])
        atoms.set_constraint(FixAtoms(indices=[0]))
        atoms.pbc = (False, False, False)

    # workaround for O2
    if name in ['O2']:
        atoms.set_initial_magnetic_moments(magmoms=[1,1])
    nbands = -1 if formula in ['H'] else -4

    calc_config = {
        'xc': f'{func}',
        'nbands': nbands,
        'basis': 'dzp',
        'mode': {'name':'pw','ecut':800,'force_complex_dtype':True},
        'gpts': h2gpts(0.12, atoms.get_cell(), idiv=4),
        'parallel': {'augment_grids': True, 'sl_auto': True},
        'hund': hund,
        'spinpol': spinpol,
        'symmetry': 'off',
        'txt': f'{name}_{func}_{sufx}.txt'
    }

    # set calculator
    calc  = GPAW(**calc_config)
    atoms.calc = calc

    # run geometry optimization
    dyn = QuasiNewton(atoms, trajectory=None, maxstep=0.05)
    tra = Trajectory(f'{name}_{func}_{sufx}.traj', 'a', atoms)
    dyn.attach(tra.write, interval=1)
    dyn.run(fmax=0.01)

    # get reference energy
    ref_energy = atoms.get_potential_energy()
    parprint(f'{func} energy: {ref_energy}')

    # move files
    if world.rank == 0:
        shutil.move(f'{name}_{func}_{sufx}.txt',  f'./output/{name}_{func}_{sufx}.txt')
        shutil.move(f'{name}_{func}_{sufx}.traj', f'./output/{name}_{func}_{sufx}.traj')

    end_time   = time.time()
    parprint(f'Time: {end_time - start_time} s')

    # list of GGA functionals
    GGA_dfs = [
        # xc
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
    ]

    nsc_energies = {}

    # get non-self consistent energies
    energy = atoms.get_potential_energy()
    for fd in GGA_dfs:
        try:
            gga_delta_energy = calc.get_xc_difference(fd)
            parprint(f'{fd} nsc energy: {ref_energy + gga_delta_energy}')
            nsc_energies[fd] = ref_energy + gga_delta_energy
        except Exception as e:
            parprint(f'{e}')
            continue

    # attempt to connect to the database
    try:
        database = db.connect('DF-and-water-formation.db')
    except FileNotFoundError:
        with db.connect('DF-and-water-formation.db') as database:
            pass  # The database will be created automatically

    # Write data to the database
    database.write(atoms, relaxed=True, density_functional=f'{func}', correction=f'{sufx}', data={'nsc_energies': f'{nsc_energies}'})

def main():
    molecules = ['H', 'O', 'O2','H2','H2O','H2O2']
    functionals = ['HSE06'] #PBE, PBE0
    corrections = [''] #, '', 'U', 'HF', 'SIC', 'SIC2']

    if not os.path.exists('output'):
        os.makedirs('output')

    for molecule in molecules:
        for functional in functionals:
            for correction in corrections:
                if ((functional in ['PBE', 'HSE06']) and (correction in [''])):
                    parprint(f'{functional} {correction} {molecule}')
                    run_dft(functional, correction, molecule)
                else:
                    parprint('Check input values')

if __name__ == "__main__":
    main()
