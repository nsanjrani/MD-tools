import h5py
import numpy as np


def write_contact_map(h5_file: h5py.File, rows, cols):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype("int16"))

    # list of np arrays of shape (2 * X) where X varies
    data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
    h5_file.create_dataset(
        "contact_map",
        data=data,
        dtype=dt,
        fletcher32=True,
        chunks=(1,) + data.shape[1:],
    )


def write_point_cloud(h5_file: h5py.File, point_cloud: np.ndarray):
    h5_file.create_dataset(
        "point_cloud",
        data=point_cloud,
        dtype="float32",
        fletcher32=True,
        chunks=(1,) + point_cloud.shape[1:],
    )


def write_rmsd(h5_file: h5py.File, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def write_fraction_of_contacts(h5_file: h5py.File, fnc):
    h5_file.create_dataset(
        "fnc", data=fnc, dtype="float16", fletcher32=True, chunks=(1,)
    )
