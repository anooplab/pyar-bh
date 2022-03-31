import os
import subprocess as subp
import sys

import numpy as np
from scipy.optimize import basinhopping as bh

bohr2angstrom = 0.52917726


def which(program):
    import os

    def is_exe(exec_path):
        return os.path.isfile(exec_path) and os.access(exec_path, os.X_OK)

    file_path, file_name = os.path.split(program)
    if file_path:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    print("nothing found")
    sys.exit()


def create_orca_input(symbols, coordinates, charge, mult, keywords,
                      extra_keywords, nprocs):
    name = "molecule"

    with open(f'{name}.inp', "w+") as file:
        if 'engrad' not in keywords.lower():
            keywords += ' ENGRAD'
        file.write(f"{keywords}\n")
        if nprocs:
            # noinspection SpellCheckingInspection
            file.write(f"%pal nprocs {nprocs} end\n")
        if extra_keywords:
            file.write(f"{extra_keywords}\n")
        file.write(f"* xyz {charge} {mult}\n")
        for z, c in zip(symbols, coordinates):
            coordinate_line = f"{z:2s}  " \
                              f"{c[0]:12.8f}  " \
                              f"{c[1]:12.8f}  " \
                              f"{c[2]:12.8f}\n"
            file.write(coordinate_line)
        file.write('*\n')

        file.write("\n")
    return name


def create_gaussian_input(symbols, coordinates, charge, mult, keywords,
                          extra_keywords, nprocs):
    name = "molecule"

    with open(f'{name}.gjf', "w+") as file:
        file.write("%nosave\n")
        if nprocs:
            file.write(f"%nprocs={nprocs}\n")
        if extra_keywords:
            file.write(f"{extra_keywords}\n")
        if 'force' not in keywords.lower():
            keywords += ' Force'
        file.write(f"{keywords}\n\n")
        file.write(f"{''.join(symbols)} {len(coordinates)} cluster\n\n")
        file.write(str(charge) + " " + str(mult) + "\n")
        for z, c in zip(symbols, coordinates):
            coordinate_line = f"{z:2s}  " \
                              f"{c[0]:12.8f}  " \
                              f"{c[1]:12.8f}  " \
                              f"{c[2]:12.8f}\n"
            file.write(coordinate_line)

        file.write("\n")
    return name


def read_gaussian_energy(name):
    file_to_open = f'{name}.log'
    with open(file_to_open, 'r+') as fp:
        return next(
            (
                float(line.split()[4])
                for line in fp.readlines()[::-1]
                if 'SCF Done' in line
            ),
            1e10,
        )


def read_orca_energy_and_gradients(name):
    file_to_open = f'{name}.engrad'
    with open(file_to_open, 'r+') as fp:
        cs = fp.readlines()
        energy = float(cs[7])
        gradients = [float(x) for x in cs[11:11 + int(cs[3]) * 3]]
        return energy, gradients


def run_gaussian(name):
    with open(f'{name}.log', 'w') as out_n_err:
        out = subp.Popen(["g16", f'{name}.gjf'], stdout=out_n_err,
                         stderr=out_n_err)
        out.communicate()
        out.poll()
        exit_status = out.returncode
    return exit_status


def run_orca(name):
    # noinspection SpellCheckingInspection
    exe = which('orca')
    with open(f'{name}.out', 'w') as out_n_err:
        out = subp.Popen([exe, f'{name}.inp'], stdout=out_n_err,
                         stderr=out_n_err)
        out.communicate()
        out.poll()
        exit_status = out.returncode
    return exit_status


def write_xyz(coordinates, symbols, name, xyz_file_name):
    xyz_coordinate = coordinates.reshape((-1, 3))
    with open(xyz_file_name, 'a+') as fp:
        fp.writelines(f"{len(xyz_coordinate)}\n")
        fp.writelines(f"{name}\n")
        for symbol, line in zip(symbols, xyz_coordinate):
            fp.writelines(
                f"{symbol:2s} "
                f"{line[0]:12.8f} "
                f"{line[1]:12.8f} "
                f"{line[2]:12.8f}\n")


def calculate_g16_energy(coordinates, atoms, charge, multiplicity, keywords,
                         extra_keywords, nprocs):
    coordinate_i = coordinates.reshape((-1, 3))
    name = create_gaussian_input(atoms, coordinate_i, charge, multiplicity,
                                 keywords, extra_keywords, nprocs)
    exit_status = run_gaussian(name)
    return read_gaussian_energy(name) if exit_status == 0 else 1e10


def calculate_orca_energy_and_gradients(coordinates, atoms, charge, multiplicity, keywords,
                                        extra_keywords, nprocs):
    coordinate_i = coordinates.reshape((-1, 3))
    name = create_orca_input(atoms, coordinate_i, charge, multiplicity,
                             keywords, extra_keywords, nprocs)
    exit_status = run_orca(name)
    return read_orca_energy_and_gradients(name) if exit_status == 0 else 1e10


def read_xyz(filename):
    with open(filename) as fp:
        f = fp.readlines()
    try:
        number_of_atoms = int(f[0])
    except ValueError as e:
        sys.exit(f"First line should be number of atoms in the file {filename}\n{e}")
    try:
        geometry_section = [each_line.split() for each_line in f[2:] if
                            len(each_line) >= 4]
    except ValueError as e:
        sys.exit(
            f"Something wrong with reading the geometry section\n{e}")
    if len(geometry_section) != number_of_atoms:
        sys.exit('Error in reading %s' % filename)
    atoms_list = []
    coordinates = []
    for c in geometry_section:
        try:
            symbol = c[0].capitalize()
            x_coord = float(c[1])
            y_coord = float(c[2])
            z_coord = float(c[3])
        except ValueError as e:
            sys.exit(f'Error in reading {filename}\n{e}')
        atoms_list.append(symbol)
        coordinates.append([x_coord, y_coord, z_coord])

    mol_coordinates = np.array(coordinates)
    mol_name = filename[:-4]
    return atoms_list, mol_coordinates, mol_name


def run_bh(cla):
    xyz_file = cla['input_file']
    atoms, coordinates, name = read_xyz(xyz_file)
    software = cla['software']
    cla.pop('software')
    if software == 'orca':
        objective_function = calculate_orca_energy_and_gradients
    elif software == 'gaussian':
        objective_function = calculate_g16_energy
    else:
        objective_function = None

    arguments = (atoms, cla['charge'], cla["multiplicity"],
                 cla["keywords"], cla["extra_keywords"], cla["nprocs"])

    if os.path.exists('best_trj.xyz'):
        os.remove('best_trj.xyz')

    if os.path.exists('best.xyz'):
        os.remove('best.xyz')

    def coordinate_update(x):
        write_xyz(x, cla['atoms'], f"intermediates stage",
                  'best_trj.xyz')

    coordinates /= bohr2angstrom
    coordinates = coordinates.reshape(-1)
    result = bh(objective_function, coordinates,
                minimizer_kwargs={'method': 'BFGS', 'jac': True, 'args': arguments}, T=400,
                disp=True, stepsize=0.5, callback=coordinate_update)
    c = result.x * bohr2angstrom
    c = c.reshape((-1, 3))
    print(result)

    print(result.message)
    final_energy = result.fun
    final_coordinates = result.x
    write_xyz(c, atoms, f"Energy: {final_energy}", 'best.xyz')
    print("Global Best")
    print(f"Energy: {final_energy}")
    print(f"Coordinates\n{final_coordinates.reshape(-1, 3)}")
    return final_energy, final_coordinates


def main():
    pass


if __name__ == "__main__":
    main()
