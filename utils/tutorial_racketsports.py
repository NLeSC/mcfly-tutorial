"""Data precprocessing or loading for RacketSports dataset.

Summary:
Contains script to preprocess RacketSports dataset and function to load the
already preprocessed dataset.

This dataset is rather simple which makes it well suited for quick training
of mcfly models.
"""
import numpy as np
import os.path
import zipfile
import six.moves.urllib as urllib


def download_preprocessed_data(directory_to_extract_to):
    """Load already preprocessed data from zenodo.

    Args:
    ----
    directory_to_extract_to: str
        Define directory to extract dataset to (if not yet present).
    """
    data_path = os.path.join(directory_to_extract_to,
                             'RacketSports', 'preprocessed')

    if not os.path.isdir(data_path):
        path_to_zip_file = os.path.join(directory_to_extract_to, 'RacketSports.zip')

        # Download zip file with data
        if not os.path.isfile(path_to_zip_file):
            print("Downloading data...")
            local_fn, headers = urllib.request.urlretrieve(
                'https://zenodo.org/record/3743603/files/RacketSports.zip',
                filename=path_to_zip_file)
        else:
            print("Data already downloaded")
        # Extract the zip file
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            print("Extracting data...")
            zip_ref.extractall(directory_to_extract_to)
        print("Done")
    else:
        print("Data already downloaded and extracted.")

    return data_path


def fetch_and_preprocess(directory_to_extract_to,
                         output_dir='preprocessed'):
    """High level function to fetch_and_preprocess the RacketSports dataset.

    Parameters
    ----------
    directory_to_extract_to : str
        the directory where the data will be stored
    ouptput_dir : str
        name of the directory to write the outputdata to

    Returns
    -------
    outdatapath: str
        The directory in which the numpy files are stored
    """
    targetdir = fetch_data(directory_to_extract_to)
    outdatapath = os.path.join(targetdir, output_dir)
    if not os.path.exists(outdatapath):
        os.makedirs(outdatapath)
    if os.path.isfile(os.path.join(outdatapath, 'X_train.npy')):
        print('Data previously pre-processed and np-files saved to ' +
              outdatapath)
    else:
        preprocess(targetdir, outdatapath)
    return outdatapath


def preprocess(targetdir, outdatapath):
    """ Function to preprocess the RacketSports data after it is fetched

    Parameters
    ----------
    targetdir : str
        subdirectory of directory_to_extract_to, targetdir
        is defined by function fetch_data
    outdatapath : str
        a subdirectory of directory_to_extract_to, outdatapath
        is the direcotry where the Numpy output will be stored.

    Returns
    -------
    None
    """
    datadir = os.path.join(targetdir)  #, 'RacketSports')
    filenames = os.listdir(datadir)
    filenames.sort()
    print('Start pre-processing all ' + str(len(filenames)) + ' files...')

    # Load ans split data
    file_train = os.path.join(datadir, 'RacketSports_TRAIN.arff')
    file_test = os.path.join(datadir, 'RacketSports_TEST.arff')
    X_train, y_train = load_racket_arff(file_train)
    X_test, X_val, y_test, y_val = load_and_split(file_test, random_seed=1)

    store_data(X_train, y_train, X_name='X_train', y_name='y_train',
               outdatapath=outdatapath, shuffle=True)
    store_data(X_val, y_val, X_name='X_val', y_name='y_val',
               outdatapath=outdatapath, shuffle=False)
    store_data(X_test, y_test, X_name='X_test', y_name='y_test',
               outdatapath=outdatapath, shuffle=False)

    print('Processed data succesfully stored in ' + outdatapath)
    return None


def fetch_data(directory_to_extract_to):
    """
    Fetch the data and extract the contents of the zip file
    to the directory_to_extract_to.
    First check whether this was done before, if yes, then skip

    Parameters
    ----------
    directory_to_extract_to : str
        directory to create subfolder 'PAMAP2'

    Returns
    -------
    targetdir: str
        directory where the data is extracted
    """
    targetdir = os.path.join(directory_to_extract_to, "RacketSports")
    if os.path.exists(targetdir):
        print('Data previously downloaded and stored in ' + targetdir)
    else:
        os.makedirs(targetdir)  # create target directory
        # Download the PAMAP2 data, this is 688 Mb
        path_to_zip_file = os.path.join(directory_to_extract_to, 'RacketSports.zip')
        test_file_exist = os.path.isfile(path_to_zip_file)
        if test_file_exist is False:
            url = str('http://www.timeseriesclassification.com/' +
                      'Downloads/RacketSports.zip')
            # retrieve data from url
            local_fn, headers = urllib.request.urlretrieve(url,
                                                           filename=path_to_zip_file)
            print('Download complete and stored in: ' + path_to_zip_file)
        else:
            print('The data was previously downloaded and stored in ' +
                  path_to_zip_file)
        # unzip

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(targetdir)
        os.remove(path_to_zip_file)
    return targetdir


def load_racket_arff(filename):
    """Load data from arff file."""
    start = 0
    data = []
    labels = []
    with open(filename) as fp:
        line = fp.readline()
        count = 0
        while line:
            if start == 1:
                lines = line.split('\\n')
                data_line = []
                for l in lines:
                    data_line_sub = []
                    for entry in l.split(','):
                        if entry.startswith('B') or entry.startswith('S'):
                            labels.append(entry.replace("'", "").replace('\n', ''))
                        else:
                            data_line_sub.append(float(entry.replace("'", "")))
                    data_line.append(data_line_sub)
                data.append(data_line)

            if line.startswith('@data'):
                start = 1

            line = fp.readline()
            count += 1

    return np.swapaxes(np.array(data), 1, 2), labels


def load_and_split(file_test, random_seed=1):
    """Load data and split into train, test, validation."""
    # Load data from arff files
    X_test0, y_test0 = load_racket_arff(file_test)

    # Split dataset
    np.random.seed(random_seed)
    y_val = []
    y_test = []
    IDs_val = []
    IDs_test = []

    for label in list(set(y_test0)):
        idx = np.where(np.array(y_test0) == label)[0]
        idx1 = np.random.choice(idx, len(idx)//2, replace=False)
        idx2 = list(set(idx) - set(idx1))
        IDs_val.extend(idx1)
        IDs_test.extend(idx2)
        y_val.extend(len(idx1) * [label])
        y_test.extend(len(idx2) * [label])

        print(label, y_test0.count(label))

    X_test = X_test0[IDs_test, :, :]
    X_val = X_test0[IDs_val, :, :]
    return X_test, X_val, y_test, y_val


def store_data(X, y, X_name, y_name, outdatapath, shuffle=False):
    """
    Converts python lists x 3D and y 1D into numpy arrays
    and stores the numpy array in directory outdatapath
    shuffle is optional and shuffles the samples

    Parameters
    ----------
    X : list
        list with data
    y : list
        list with data
    X_name : str
        name to store the x arrays
    y_name : str
        name to store the y arrays
    outdatapath : str
        path to the directory to store the data
    shuffle : bool
        whether to shuffle the data before storing
    """
    X = np.array(X)
    y = np.array(y)
    # Shuffle the train set
    if shuffle:
        np.random.seed(123)
        neworder = np.random.permutation(X.shape[0])
        X = X[neworder, :, :]
        y = y[neworder]
    # Save binary file
    xpath = os.path.join(outdatapath, X_name)
    ypath = os.path.join(outdatapath, y_name)
    np.save(xpath, X)
    np.save(ypath, y)
    print('Stored ' + xpath, y_name)


def load_data(outputpath):
    """Load the numpy data as stored in directory outputpath.

    Parameters
    ----------
    outputpath : str
        directory where the numpy files are stored

    Returns
    -------
    x_train
    y_train_binary
    x_val
    y_val_binary
    x_test
    y_test_binary
    """
    ext = '.npy'
    X_train = np.load(os.path.join(outputpath, 'X_train' + ext))
    y_train = np.load(os.path.join(outputpath, 'y_train' + ext))
    X_val = np.load(os.path.join(outputpath, 'X_val' + ext))
    y_val = np.load(os.path.join(outputpath,  'y_val' + ext))
    X_test = np.load(os.path.join(outputpath, 'X_test' + ext))
    y_test = np.load(os.path.join(outputpath,  'y_test' + ext))
    return X_train, y_train, X_val, y_val, X_test, y_test
