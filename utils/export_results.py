import pickle


def pkl_export(predictions, filename):
    """ Exports a given array into binary pkl format

    Args:
        predictions: array to save
        filename: string with file path: for instance preds.pkl
    """
    with open(filename, 'wb') as outfile:
        pickle.dump(predictions, outfile, protocol=2)


def pkl_import(filename):
    """ Imports a binary pickled file

    Args:
        filename: path to the file to import

    Returns: the file content.

    """
    with open(filename, 'rb') as infile:
        return pickle.load(infile, encoding="bytes")


# Test
if __name__ == "__main__":
    import numpy as np

    a = np.arange(10, dtype=float)
    pkl_export(predictions=a, filename="tmp.pkl")
    assert ((np.array(pkl_import("tmp.pkl")) == a[:]).all())
    print("Test passed")
