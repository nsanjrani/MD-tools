import h5py
import tempfile
from typing import List
from simtk.openmm.app import PDBFile
import MDAnalysis
from MDAnalysis.analysis import rms
from concurrent.futures import ProcessPoolExecutor


class ReporterComputation:
    def __init__(self):
        pass

    def write(self, h5_file: h5py.File):
        raise NotImplementedError("Base class must implement.")

    def run(self, pdb_file: str):
        raise NotImplementedError("Base class must implement.")


class OfflineReporter:
    def __init__(
        self,
        file: str,
        reportInterval: int,
        frames_per_h5: int = 0,
        computations: List[ReporterComputation] = [],
        particle_positions: bool = True,
        particle_velocities: bool = False,
        forces: bool = False,
        energies: bool = False,
    ):
        self._frame_count = 0  # Keep track of number of frames seen per HDF5 file
        self._h5_file_count = 0  # Keep track of number of HDF5 files written
        self._base_name = file
        self._report_interval = reportInterval
        self._frames_per_h5 = frames_per_h5
        self._computations = computations
        self._particle_positions = particle_positions
        self._particle_velocities = particle_velocities
        self._forces = forces
        self._energies = energies

        # Need for async computations
        self._executor = ProcessPoolExecutor(max_workers=len(self._computations) + 1)
        self._futures = []

    def _write_report(self):
        h5_fname = f"{self._base_name}_{self._h5_file_count:04}.h5"
        with h5py.File(h5_fname, "w", swmr=False) as f:
            for computation in self._computations:
                computation.write(f)

    def _block_on_report(self):
        # Wait for previous report to finish
        for future in self._futures:
            future.result()  # Could raise exceptions
        self._futures = []

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (
            steps,
            self._particle_positions,
            self._particle_velocities,
            self._forces,
            self._energies,
            None,
        )

    def report(self, simulation, state):

        with tempfile.NamedTemporaryFile(suffix=".pdb") as file:
            PDBFile.writeFile(
                simulation.topology,
                state.getPositions(),
                file=file,
                keepIds=True,
                extraParticleIdentifier="EP",
            )

            self._block_on_report()

            self._futures = [
                self._executor.submit(computation.run, file.name)
                for computation in self._computations
            ]

            self._frame_count += 1

        # Write HDF5 file
        if self._frame_count == self._frames_per_h5:
            self._frame_count = 0
            self._block_on_report()
            self._write_report()


class RMSDReporter(ReporterComputation):
    def __init__(self, reference_pdb_file: str, selection: str):
        super().__init__()

        self._rmsds = []  # For storing RMSD values in between writes
        self.selection = selection

        self._reference_positions = (
            MDAnalysis.Universe(reference_pdb_file)
            .select_atoms(self.selection)
            .positions.copy()
        )

    def write(self, h5_file: h5py.File):
        h5_file.create_dataset(
            "rmsd", data=self._rmsds, dtype="float16", fletcher32=True, chunks=(1,)
        )
        self._rmsds = []

    def report(self, pdb_file: str):
        positions = (
            MDAnalysis.Universe(pdb_file).select_atoms(self.selection).positions()
        )
        rmsd = rms.rmsd(positions, self._reference_positions, superposition=True)
        self._rmsds.append(rmsd)
