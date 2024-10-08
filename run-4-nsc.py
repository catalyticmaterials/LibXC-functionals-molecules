import numpy as np
import matplotlib.pyplot as plt
from ase.db import connect

def get_energy(db, formula, functional_correction):
    """
    Fetch the non-self-consistent energy for a given formula, functional, and correction.

    Parameters:
    db (ase.db.connect): ASE database connection object.
    formula (str): Chemical formula of the molecule.
    functional_correction (str): Functional and correction combination as functional@base_functional-correction.

    Returns:
    float: The non-self-consistent energy for the given formula.
    """
    functional, base_functional_correction = functional_correction.split('@')
    if '-' in base_functional_correction:
        base_functional, correction = base_functional_correction.split('-')
    else:
        base_functional = base_functional_correction
        correction = ''

    rows = list(db.select(formula=formula, density_functional=base_functional, correction=correction))
    if len(rows) == 0:
        raise ValueError(f"No data found for {formula} with base functional {base_functional} and correction {correction}")        

    nsc_energies = eval(rows[0].data['nsc_energies'])
    if functional in nsc_energies:
        return nsc_energies[functional]

    raise ValueError(f"No non-self-consistent energy found for {formula} with functional {functional} in base functional {base_functional} with correction {correction}")

def calculate_reaction_energy(db, product, coeff1, reactant1, coeff2, reactant2, functional_correction):
    """
    Calculate the reaction energy for a given reaction.

    Parameters:
    db (ase.db.connect): ASE database connection object.
    product (str): The product molecule of the reaction.
    coeff1 (float): Coefficient for the first reactant.
    reactant1 (str): The first reactant molecule.
    coeff2 (float): Coefficient for the second reactant.
    reactant2 (str): The second reactant molecule.
    functional_correction (str): Functional and correction combination.

    Returns:
    float: The reaction energy, rounded to two decimal places.
    """
    product_energy = get_energy(db, product, functional_correction)
    reactant1_energy = get_energy(db, reactant1, functional_correction)
    reactant2_energy = get_energy(db, reactant2, functional_correction)
    return round(product_energy - coeff1 * reactant1_energy - coeff2 * reactant2_energy, 2)

def limit_deviation(deviation, limit=0.5):
    """
    Limit the deviation to a specified range, defaulting to ±0.5 eV.

    Parameters:
    deviation (float): The original deviation value.
    limit (float): The maximum allowable absolute value of the deviation.

    Returns:
    float: The limited deviation value.
    """
    return max(min(deviation, limit), -limit)

def main():
    db = connect("DF-and-water-formation.db")

    base_functionals_corrections = [('HSE06', ''), ('PBE', ''), ('RPBE', ''), ('GGA_XC_KT3', ''), ('GGA_X_N12+GGA_C_N12', ''), ('GGA_XC_VV10', '')]

    for base_functional, correction in base_functionals_corrections:
        sample_row = db.get(density_functional=base_functional, correction=correction, formula='O')
        nsc_functionals = list(eval(sample_row.data['nsc_energies']).keys())

        deviations = {}

        for functional in nsc_functionals:
            # Calculate deviations and apply limit of ±0.5 eV
            E_diss_o2 = limit_deviation(-1 * calculate_reaction_energy(db, "O2", 1, "O", 1, "O", functional + "@" + base_functional + ('-' + correction if correction else '')) - 5.16)
            E_diss_h2 = limit_deviation(-1 * calculate_reaction_energy(db, "H2", 1, "H", 1, "H", functional + "@" + base_functional + ('-' + correction if correction else '')) - 4.52)
            E_form_h2o = limit_deviation(calculate_reaction_energy(db, "H2O", 1, "H2", 0.5, "O2", functional + "@" + base_functional + ('-' + correction if correction else '')) + 2.51)
            E_form_h2o2 = limit_deviation(calculate_reaction_energy(db, "H2O2", 1, "H2", 1, "O2", functional + "@" + base_functional + ('-' + correction if correction else '')) + 1.41)

            deviations[functional] = {
                "Dissociation of O2": E_diss_o2,
                "Dissociation of H2": E_diss_h2,
                "Formation of H2O": E_form_h2o,
                "Formation of H2O2": E_form_h2o2
            }

        # Save to a file
        output_filename = f"nsc-{base_functional + ('-' + correction if correction else '')}.txt"
        with open(output_filename, "w") as f:
            for functional, reactions in deviations.items():
                f.write(f"Functional: {functional}\n")
                for reaction, deviation in reactions.items():
                    f.write(f"{reaction}: {deviation:.2f} eV\n")
                f.write("\n")

        # Calculate mean absolute deviations (MAD)
        mad_values = {functional: np.mean([abs(dev) for dev in reactions.values()]) for functional, reactions in deviations.items()}

        # Print top 10 functionals with lowest MAD
        top_10_functionals = sorted(mad_values, key=mad_values.get)[:10]
        print(f"Top 10 functionals with lowest mean absolute deviation for {base_functional + ('-' + correction if correction else '')}:")
        for func in top_10_functionals:
            print(f"{func}: {mad_values[func]:.2f} eV")

        # Plotting
        for reaction, reference in [("Dissociation of O2", 5.16), ("Dissociation of H2", 4.52), ("Formation of H2O", -2.51), ("Formation of H2O2", -1.41)]:
            sorted_devs = sorted([(func, deviations[func][reaction]) for func in deviations], key=lambda x: x[1])
            functionals, dev_values = zip(*sorted_devs)

            plt.figure(figsize=(10, 10))
            plt.barh(functionals, dev_values, color='skyblue')
            plt.xlabel('Deviation from Reference (eV)')
            plt.title(f'Deviation for {reaction}')
            plt.tight_layout()
            plt.savefig(f"Deviation_{reaction.replace(' ', '_')}_{base_functional + ('-' + correction if correction else '')}.png")
            plt.show()

if __name__ == "__main__":
    main()
