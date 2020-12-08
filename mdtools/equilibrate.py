import shutil
from pathlib import Path
import simtk.openmm.app as app
from mdtools.openmm_utils import configure_simulation


def equilibrate(
    pdb_file: str,
    top_file: str,
    output_pdb: str,
    solvent_type: str,
    gpu_index: int,
    dt_ps: float,
    temperature_kelvin: float,
    heat_bath_friction_coef: float,
):
    """Run equilibration and output PDB file at the end"""
    sim = configure_simulation(
        pdb_file,
        top_file,
        solvent_type,
        gpu_index,
        dt_ps,
        temperature_kelvin,
        heat_bath_friction_coef,
    )

    # Report a PDB file at the end of equilibration
    sim.reporters.append(app.PDBReporter(output_pdb, nsteps))

    # Run equilibration
    sim.step(nsteps)


if __name__ == "__main__":
    # Directory containing subdirectories with PDB/TOP file pairs
    input_path = Path("")
    # Directory to write equilibrated PDB/TOPs to
    output_path = Path("")
    # Solvent type; explicit or implicit
    solvent_type = "explicit"
    # GPU index (always 0)
    gpu_index = 0
    # Integration step
    dt_ps = 0.0001
    # Temperature to run equilibration
    temperature_kelvin = 300
    # Heat bath friction coefficient
    heat_bath_friction_coef = 1.0
    # Length of equilibration
    simulation_length_ns = 2

    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    for system_dir in input_path.iterdir():
        if not system_dir.is_dir():
            continue

        # Create system output directory
        output_dir = output_path.joinpath(system_dir.name)
        output_dir.mkdir()

        top_file = next(system_dir.glob("*.top"))
        pdb_file = next(system_dir.glob("*.pdb"))

        # Copy topology file to output directory
        shutil.copy(top_file, output_dir)
        output_pdb = output_dir.joinpath(pdb_file.name)

        equilibrate(
            pdb_file.as_posix(),
            top_file.as_posix(),
            output_pdb.as_posix(),
            solvent_type,
            gpu_index,
            dt_ps,
            temperature_kelvin,
            heat_bath_friction_coef,
        )
