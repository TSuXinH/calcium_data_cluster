import h5py


def unzip_single_session(path):
    file = h5py.File(path)
    keys = list(file.keys())
    if 'CaA0' in keys:
        data = file['CaA0']