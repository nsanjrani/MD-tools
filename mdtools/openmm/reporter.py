import h5py
import numpy as np
from typing import Optional, List
import simtk.unit as u
import MDAnalysis
from MDAnalysis.analysis import distances, rms, contacts
from mdtools.analysis.order_parameters import fraction_of_contacts
from mdtools.writers import (
    write_contact_map,
    write_heavy_atom_contacts,
    write_point_cloud,
    write_fraction_of_contacts,
    write_rmsd,
)


def wrap(atoms):
    def wrap_nsp10_16(positions):
        # update the positions
        atoms.positions = positions
        # only porting CA into nsp16
        # TODO: only selecting the protein, does this need to be changed all it does is center the protein https://userguide.mdanalysis.org/stable/examples/transformations/center_protein_in_box.html
        nsp16 = atoms.segments[0].atoms
        # wrapping atoms into continous frame pbc box
        box_edge = nsp16.dimensions[0]
        box_center = box_edge / 2
        trans_vec = box_center - np.array(nsp16.center_of_mass())
        atoms.translate(trans_vec).wrap()
        trans_vec = box_center - np.array(atoms.center_of_mass())
        atoms.translate(trans_vec).wrap()

        return atoms.positions

    return wrap_nsp10_16


class OfflineReporter:
    def __init__(
        self,
        file: str,
        reportInterval: int,
        frames_per_h5: int = 0,
        wrap_pdb_file: Optional[str] = None,
        reference_pdb_file: Optional[str] = None,
        # change to CA or ligand too?
        openmm_selection: List[str] = ["CA"],
        # TODO: change this to just name CA and ligand ie not protein? protein and ligand?
        mda_selection: str = "protein and name CA",
        mda_lig_selection: str = "resname BTN and not name H*",
        threshold: float = 8.0, # number of angstroms that defines a contact
        contact_map: bool = False,
        point_cloud: bool = False,
        fraction_of_contacts: bool = False,
        heavy_atom_contacts: bool = True,
    ):

        if fraction_of_contacts and reference_pdb_file is None:
            raise ValueError(
                "Computing `fraction_of_contacts` requires `reference_pdb_file`."
            )
        if contact_map and reference_pdb_file is None:
            raise ValueError("Computing `contact_map` requires `reference_pdb_file`.")

        self._file_idx = 0
        self._base_name = file
        self._report_interval = reportInterval
        self._wrap_pdb_file = wrap_pdb_file
        self._reference_pdb_file = reference_pdb_file
        self._openmm_selection = openmm_selection
        self._mda_selection = mda_selection
        self._mda_lig_selection = mda_lig_selection
        self._threshold = threshold
        self._frames_per_h5 = frames_per_h5
        self._contact_map = contact_map
        self._point_cloud = point_cloud
        self._fraction_of_contacts = fraction_of_contacts
        self._heavy_atom_contacts = heavy_atom_contacts

        self._init_batch()
        self._init_reference_positions()
        self._init_ligand_reference_positions()
        self._init_reference_contact_map()
        self._init_wrap()

    def _init_reference_positions(self):
        # Set up for reporting optional RMSD to reference state
        if self._reference_pdb_file is None:
            self._reference_positions = None
            return

        mda_u = MDAnalysis.Universe(self._reference_pdb_file)

        self._reference_positions = mda_u.select_atoms(
            self._mda_selection
        ).positions.copy()

    def _init_ligand_reference_positions(self):
        u = MDAnalysis.Universe(self._reference_pdb_file)
        self._ref_lig_positions = u.select_atoms(self._mda_lig_selection).indices.copy()

    def _init_reference_contact_map(self):
        if not self._fraction_of_contacts:
            return

        assert self._reference_pdb_file is not None
        mda_u = MDAnalysis.Universe(self._reference_pdb_file)
        reference_positions = mda_u.select_atoms(self._mda_selection).positions.copy()
        self._reference_contact_map = self._compute_contact_map(reference_positions)

    def _init_wrap(self):
        if self._wrap_pdb_file is None:
            self.wrap = None
            return

        mda_u = MDAnalysis.Universe(self._wrap_pdb_file)
        atoms = mda_u.select_atoms(self._mda_selection)
        self.wrap = wrap(atoms)

    def _init_batch(self):
        # Frame counter for writing batches to HDF5
        self._num_frames = 0

        # Row, Column indices for contact matrix in COO format
        if self._contact_map:
            self._rows, self._cols = [], []

        if self._reference_pdb_file is not None:
            self._rmsd = []

        if self._fraction_of_contacts:
            self._fraction_of_contacts_data = []

        if self._point_cloud:
            self._point_cloud_data = []

        if self._heavy_atom_contacts:
            self._heavy_atom_contacts_data = []

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _collect_rmsd(self, positions):
        if self.wrap is not None:
            positions = self.wrap(positions)

        rmsd = rms.rmsd(positions, self._reference_positions, superposition=True)
        self._rmsd.append(rmsd)
    
    # TODO: check this function, is the wrapping of these positions needed or would it put them on top of each other?
    def _compute_heavy_atom_contacts(self, positions_lig, positions_prot):
        # TODO: do we need to change the threshold? What counts as a contact?
        distance = distances.distance_array(positions_prot, positions_lig)
        heavy_contacts = contacts.hard_cut_q(distance, self._threshold)

        self._heavy_atom_contacts_data.append(heavy_contacts)

    def _compute_contact_map(self, positions):
        contact_map = distances.contact_matrix(
            positions, self._threshold, returntype="sparse"
        )
        return contact_map

    def _collect_contact_map(self, contact_map):
        # Represent contact map in COO sparse format
        coo = contact_map.tocoo()
        self._rows.append(coo.row.astype("int16"))
        self._cols.append(coo.col.astype("int16"))

    def _collect_fraction_of_contacts(self, contact_map):
        self._fraction_of_contacts_data.append(
            fraction_of_contacts(contact_map, self._reference_contact_map)
        )

    def _collect_point_cloud(self, positions):
        self._point_cloud_data.append(positions)

    def report(self, simulation, state):
        # all atom indices
        atom_indices = [
            a.index
            for a in simulation.topology.atoms()
            if a.name in self._openmm_selection
        ]
        all_positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions = all_positions[atom_indices].astype(np.float32)

        if self._heavy_atom_contacts:
            # protein atom indices
            protein_positions = positions

            # ligand atom indices
            ligand_positions = all_positions[self._ref_lig_positions]

            self._compute_heavy_atom_contacts(ligand_positions, protein_positions)

        if self._contact_map or self._fraction_of_contacts:
            contact_map = self._compute_contact_map(positions)
            if self._contact_map:
                self._collect_contact_map(contact_map)
            if self._fraction_of_contacts:
                self._collect_fraction_of_contacts(contact_map)

        if self._point_cloud:
            self._collect_point_cloud(positions)

        if self._reference_positions is not None:
            self._collect_rmsd(positions)

        self._num_frames += 1

        if self._num_frames == self._frames_per_h5:
            file_name = f"{self._base_name}.h5"

            with h5py.File(file_name, "w", swmr=False) as h5_file:

                if self._contact_map:
                    write_contact_map(h5_file, self._rows, self._cols)

                if self._point_cloud:
                    self._point_cloud_data = np.transpose(
                        self._point_cloud_data, [0, 2, 1]
                    )
                    write_point_cloud(h5_file, self._point_cloud_data)

                # Optionally, write rmsd to the reference state
                if self._reference_positions is not None:
                    write_rmsd(h5_file, self._rmsd)

                if self._fraction_of_contacts:
                    write_fraction_of_contacts(h5_file, self._fraction_of_contacts_data)

                if self._heavy_atom_contacts:
                    write_heavy_atom_contacts(h5_file, self._heavy_atom_contacts_data)

            self._init_batch()
