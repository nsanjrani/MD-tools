from tqdm import tqdm
import shutil
from pathlib import Path
import simtk.unit as u
import simtk.openmm.app as app
from mdtools.openmm_utils import configure_simulation


def equilibrate(
    pdb_file: str,
    top_file: str,
    log_file: str,
    output_pdb: str,
    solvent_type: str,
    gpu_index: int,
    log_steps: int,
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

    # Configure simulation output log
    sim.reporters.append(
        app.StateDataReporter(
            log_file,
            log_steps,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )

    print("starting sim")
    # Run equilibration
    sim.step(nsteps)


if __name__ == "__main__":
    # Directory containing subdirectories with PDB/TOP file pairs
    input_path = Path("/homes/abrace/tmp/plpro_outliers_top80")
    # Directory to write equilibrated PDB/TOPs to
    output_path = Path("/homes/abrace/tmp/plpro_outliers_top80_equil")
    # Solvent type; explicit or implicit
    solvent_type = "explicit"
    # GPU index (always 0)
    gpu_index = 0
    # Integration step
    dt_ps = 0.0001 * u.picosecond
    # Temperature to run equilibration
    temperature_kelvin = 300
    # Heat bath friction coefficient
    heat_bath_friction_coef = 1.0
    # Length of equilibration
    simulation_length_ns = 2 * u.nanosecond
    # Log to report every log_interval_ps picoseconds
    log_interval_ps = 50 * u.picosecond

    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)
    log_steps = int(simulation_length_ns / log_interval_ps)

    for system_dir in tqdm(input_path.iterdir()):
        if not system_dir.is_dir():
            continue

        # Create system output directory
        output_dir = output_path.joinpath(system_dir.name)
        output_dir.mkdir()

        top_file = next(system_dir.glob("*.prmtop"))
        pdb_file = next(system_dir.glob("*.pdb"))

        # Copy topology file to output directory
        shutil.copy(top_file, output_dir)
        output_pdb = output_dir.joinpath(pdb_file.name)
        log_file = output_dir.joinpath("output.log")

        print(top_file)
        print(pdb_file)
        print(output_dir)

        equilibrate(
            pdb_file.as_posix(),
            top_file.as_posix(),
            log_file.as_posix(),
            output_pdb.as_posix(),
            solvent_type,
            gpu_index,
            log_steps,
            dt_ps,
            temperature_kelvin,
            heat_bath_friction_coef,
        )
