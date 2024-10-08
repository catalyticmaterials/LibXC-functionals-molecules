from ase.db import connect
import matplotlib.pyplot as plt
import numpy as np

def get_energy(db, formula, density_functional, correction):
    """
    Fetch the energy of a given formula from the database for a specific functional and correction.

    Parameters:
    db (ase.db.connect): ASE database connection object.
    formula (str): Chemical formula of the molecule.
    density_functional (str): The density functional used for the calculation.
    correction (str): The correction method applied, if any.

    Returns:
    float: The energy value for the given molecule.
    """
    rows = list(db.select(formula=formula, density_functional=density_functional, correction=correction))
    if len(rows) == 0:
        raise ValueError(f"No data found for {formula} with functional {density_functional} and correction {correction}")
    return rows[0].energy

def calculate_reaction_energy(db, product, coeff1, reactant1, coeff2, reactant2, density_functional, correction):
    """
    Calculate the reaction energy for a given reaction.

    Parameters:
    db (ase.db.connect): ASE database connection object.
    product (str): The product molecule of the reaction.
    coeff1 (float): Coefficient for the first reactant.
    reactant1 (str): The first reactant molecule.
    coeff2 (float): Coefficient for the second reactant.
    reactant2 (str): The second reactant molecule.
    density_functional (str): The density functional used for the calculation.
    correction (str): The correction method applied, if any.

    Returns:
    float: The reaction energy.
    """
    product_energy = get_energy(db, product, density_functional, correction)
    reactant1_energy = get_energy(db, reactant1, density_functional, correction)
    reactant2_energy = get_energy(db, reactant2, density_functional, correction)
    return product_energy - coeff1 * reactant1_energy - coeff2 * reactant2_energy

def calculate_deviations(db):
    """
    Calculate energy deviations for a series of reactions compared to reference values.

    Parameters:
    db (ase.db.connect): ASE database connection object.

    Returns:
    dict: Dictionary containing reaction deviations for each functional-correction combination.
    """
    reactions = {
        "Dissociation of H2": {"product": "H2", "coeff1": 1, "reactant1": "H", "coeff2": 1, "reactant2": "H", "reference": -4.52}, 
        "Dissociation of O2": {"product": "O2", "coeff1": 1, "reactant1": "O", "coeff2": 1, "reactant2": "O", "reference": -5.16}, 
        "Formation of H2O": {"product": "H2O", "coeff1": 1, "reactant1": "H2", "coeff2": 0.5, "reactant2": "O2", "reference": -2.51},
        "Formation of H2O2": {"product": "H2O2", "coeff1": 1, "reactant1": "H2", "coeff2": 1, "reactant2": "O2", "reference": -1.41},
    }

    deviations = {}

    for reaction, details in reactions.items():
        for row in db.select():
            functional = row.density_functional
            correction = row.correction
            key = functional if not correction else f'{functional}-{correction}'

            try:
                energy = calculate_reaction_energy(db, details["product"], details["coeff1"], details["reactant1"], details["coeff2"], details["reactant2"], functional, correction)
                deviation = energy - details["reference"]
                deviations.setdefault(reaction, {}).setdefault(key, deviation)
            except ValueError as e:
                print(f"Skipping reaction {reaction} for {key} due to error: {e}")
                continue

    return deviations

def plot_deviations(deviations):
    """
    Plot the deviations for each reaction.

    Parameters:
    deviations (dict): Dictionary containing the deviations of each reaction for functional-correction combinations.
    """
    for reaction, data in deviations.items():
        sorted_data = dict(sorted(data.items(), key=lambda item: abs(item[1])))
        plt.bar(sorted_data.keys(), sorted_data.values())
        plt.title(reaction)
        plt.ylabel("Deviation (eV)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

def print_statistics(deviations):
    """
    Print statistical data (mean deviation and mean absolute deviation) for each functional-correction combination.

    Parameters:
    deviations (dict): Dictionary containing the deviations of each reaction for functional-correction combinations.
    """
    stats = {}

    for functional_correction in deviations[next(iter(deviations))].keys():
        try:
            deviation_values = [deviations[reaction][functional_correction] for reaction in deviations]
            mean_deviation = np.mean(deviation_values)
            mean_absolute_deviation = np.mean(np.abs(deviation_values))
            stats[functional_correction] = {"mean_deviation": mean_deviation, "mean_absolute_deviation": mean_absolute_deviation}  
        except KeyError:
            print(f"Error with functional-correction: {functional_correction}")
            raise

    sorted_stats = dict(sorted(stats.items(), key=lambda item: item[1]["mean_absolute_deviation"]))

    for functional_correction, data in sorted_stats.items():
        print(f"Functional-Correction: {functional_correction}")
        print(f"Mean Deviation: {data['mean_deviation']:.2f} eV")
        print(f"Mean Absolute Deviation: {data['mean_absolute_deviation']:.2f} eV")
        print("------------------------------")

def main():
    """
    Main function to connect to the database, calculate deviations, and print statistics.
    """
    db = connect("DF-and-water-formation.db")
    deviations = calculate_deviations(db)
    print_statistics(deviations)
    plot_deviations(deviations)

if __name__ == "__main__":
    main()
