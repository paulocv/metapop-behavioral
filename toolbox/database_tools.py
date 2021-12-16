"""
A module with functionalities to work with massive results from simulations,
typically with several varying parameters.
"""

import pandas as pd
import numpy as np
import itertools
from toolbox.file_tools import read_file_header, read_config_strlist, read_csv_names, HEADER_END
from toolbox.paramvar_tools import get_varparams_with_zip, ziplist_to_flat


def read_complete_output_file(filename,
                              entry_char=">", attribution_char="=", comment_char="#",
                              header_end=HEADER_END, varparam_key="vary_parameters",
                              zip_key="zip_parameters", unpack_zips=True, delimiter='\t',
                              names=None):
    """
    Imports dataframe from an output summary file based on its HEADER.
    I.e., looks for the varying parameters in 'vary_parameters' entry, then
    interpret the lists of values.
    The DataFrame index is built as the cartesian product of all varying
    parameters. Yet (update 11/2019) it may accept also zipped
    tuples of parameters that were varied together.

    Parameters
    ------
    filename : str
        Name of the input file
    entry_char : str
        Character used to start an input line in the file header.
    attribution_char : str
        Character that separates the key name from its value (header only).
    comment_char : str
        Character that indicates a comment. Used both for header and data.
    header_end : str
        String that defines the separation between the header and the main data.
        It must contain '\n' at the end.
    varparam_key : str
        Keyword for the varying parameter names list in the header. Ex:
        > vary_parameters = rho, kappa, pi, beta_ep
    zip_key : str
        Keyword for the zipped sets of parameters. Ex:
        > zip_parameters = (rho, kappa)
    unpack_zips : bool
        If True (default), the zipped sets of parameters are flattened
        at the multiindex of the resulting dataframe, meaning that each
        zipped parameter will have its own level of hierarchy. If False,
        zipped parameters are grouped as tuples in a single level.
    delimiter : str
        horizontal delimiter between column entries
    names : list
        List of column names for the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Resulting data frame read from file.
    var_param_dict : dict
        Dictionary with the varying parameters names and their lists of values.
        This may be useful to unambiguous .loc indexing of the DataFrame rows,
        as the indexes are often floats.
    """

    # Reads the file header and interprets its valid entries (including varying
    #  parameters).
    file_header = read_file_header(filename, header_end)
    header_size = len(file_header) + 1
    input_dict = read_config_strlist(file_header, entry_char, attribution_char,
                                     comment_char)
    names_list, values_list = get_varparams_with_zip(input_dict, varparam_key,
                                                     zip_key)
    flat_names_list = ziplist_to_flat(names_list)

    # Constructs the MultiIndex object as the cartesian product of all varying
    # parameters (handling zipped ones correctly).
    if unpack_zips:
        # This loop constructs the flattened/unpacked list of parameter sets,
        # in the way they are expected to be at the file.
        index_tuples = []
        for i_sim, parameter_values in enumerate(itertools.product(*values_list)):
            # Flattens the current set of parameters
            flat_parameter_values = ziplist_to_flat(parameter_values, return_type=tuple)
            index_tuples.append(flat_parameter_values)

        # Constructs the multi index with the unpacked version of the param list.
        index = pd.MultiIndex.from_tuples(index_tuples, names=flat_names_list)

    else:
        # In this case, constructs the index using tuples for the zipped params.
        index = pd.MultiIndex.from_product(values_list,
                                           names=names_list)

    # Reads the actual numerical data from file, interpreting it as a
    # pandas DataFrame with a multi-index.
    # Deprecated from pandas 0.25: pd.read_table
    df = pd.read_csv(filename, sep=delimiter,
                     skiprows=header_size, header=None,
                     comment=comment_char, na_values=["NAN"], keep_default_na=True)

    # Eliminates a NAN column, which happens in case the line has another
    # delimiter at the end
    df = df.dropna(axis=1)

    # Sets the index and restrict the data frame to the data columns (removes index cols)
    df.index = index
    df = df.loc[:, len(flat_names_list):]  # Removes the indexing columns

    # Sets the default value of the column 'names'
    # If the file header has an "outputs" topic, uses as column names
    # Else, uses integer indexes (starting from 0).
    if names is None:
        try:
            names = read_csv_names(input_dict["outputs"])
        except KeyError:
            names = list(range(len(df.columns)))

    df.columns = names  # Renames the columns

    var_param_dict = dict([(name, value)
                          for name, value in zip(names_list, values_list)])
    return df, var_param_dict


def read_mixed_output_file(filename, decimal_places=9,
                           entry_char=">", attribution_char="=", comment_char="#",
                           header_end=HEADER_END, varparams_names=None,
                           varparams_key="vary_parameters",
                           outputs_key="outputs", delimiter='\t',
                           names=None):
    """
    Reads an output summary file as a dataframe with a multiindex.
    Differently from 'read_complete_output_file', this function reads
    each parameter set from the first columns of the data, being more
    versatile (can be used with non-regularly distributed parameter
    sets.

    As float indexes may be used, the values are rounded to 'decimal_places',
    therefore numbers that differ bellow such precision will be considered
    as the same index values.

    Parameters
    ------
    filename : str
        Name of the input file
    decimal_places : int
        Number of decimal places bellow which floats are considered
        as equal indexes. This does not affect the data, only the index
        values.
    entry_char : str
        Character used to start an input line in the file header.
    attribution_char : str
        Character that separates the key name from its value (header only).
    comment_char : str
        Character that indicates a comment. Used both for header and data.
    header_end : str
        String that defines the separation between the header and the main data.
        It must contain '\n' at the end.
    varparams_names : list
        List of names of the parameters. If not informed, it is seached
        at the file header with the key specified as'varparams_key'.
    varparams_key : str
        Keyword for the varying parameter names list in the header. Ex:
        > vary_parameters = rho, kappa, pi, beta_ep
    outputs_key : str
        Keyword for the output topics, read from the file header.
    delimiter : str
        horizontal delimiter between column entries
    names : list
        List of column names for the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Resulting data frame read from file.
    """

    # Reads the file header and interprets its valid entries (including varying
    #  parameters).
    file_header = read_file_header(filename, header_end)
    header_size = len(file_header) + 1
    input_dict = read_config_strlist(file_header, entry_char, attribution_char,
                                     comment_char)

    # Tries to read the varying parameter names, if they are not informed
    # as 'varparams_names' argument
    if varparams_names is None:
        try:
            varparams_names = read_csv_names(input_dict[varparams_key])
        except KeyError:
            raise KeyError("Hey, keyword '{}' was not found in"
                           "the input file. If the file does not inform"
                           " the vary parameter names, this can be "
                           "informed to function {} as 'varparams_names'."
                           "".format(varparams_key, "read_mixed_output_file"))

    # Reads the database from file and removes inexistent entries
    # deprecated in pandas 0.25: pd.read_table
    df = pd.read_csv(filename, sep=delimiter,
                       skiprows=header_size, header=None,
                       comment=comment_char)
    df = df.dropna(axis=1)

    # Separates the dataframe between index (parameter values) and actual data.
    indexing_df = df.loc[:, :len(varparams_names)-1]  # Index (parameters) columns
    values_df = df.loc[:, len(varparams_names):]  # Values (data) column

    # Sets the default value of the column 'names'.
    # If the file header has an "outputs" topic, uses as column names
    # Else, uses integer indexes (starting from 0).
    if names is None:
        try:
            names = read_csv_names(input_dict[outputs_key])
        except KeyError:
            names = list(range(len(values_df.columns)))

    # Renames the columns of the index and values dataframes
    indexing_df.columns = varparams_names
    values_df.columns = names

    # ------------------
    # Construction of the index object

    # Gets the data types of each column (using the FIRST LINE)
    dtypes = [type(x) for x in indexing_df.loc[0]]

    # Function that converts an str type into 'object'.
    def conv(dtype):
        if dtype in [str]:
            return object
        else:
            return dtype

    df_size = len(indexing_df)

    # Initializes the index object as a list of np arrays.
    # Uses the function 'conv' to convert "str" into "object", so the strings
    # are correctly stored (otherwise, stores only the first char).
    index = [np.empty(df_size, dtype=conv(tp)) for tp in dtypes]  # New: list of arrays
    # index = np.empty((len(varparams_names), df_size))  # Old: 2D array

    # Makes a list of booleans that tells if the value can be rounded.
    # Eg: a string cannot be rounded, therefore comparison is exact.
    is_not_roundable = [x in [str] for x in dtypes]

    # For each line of the index df
    for i in range(df_size):
        line = indexing_df.loc[i]
        # For each parameter value of the line
        for j, (value, nroundable) in enumerate(zip(line, is_not_roundable)):
            # Stores the current value. Checks if the value should be rounded.
            if nroundable:
                # Non-roundable value.
                index[j][i] = value
            else:
                # Stores the rounded values of the parameter values
                index[j][i] = round(value, decimal_places)

    # Converts the arrays into Pandas MultiIndex object and sets to df
    values_df.index = pd.MultiIndex.from_arrays(index, names=varparams_names)

    return values_df


def read_simulation_file(filename,
                         entry_char=">", attribution_char="=", comment_char="#",
                         header_end=HEADER_END, delimiter='\t', outputs_key="outputs",
                         names=None):
    """ Reads a simulation file and returns as a dataframe. The file is assumed
    to have an index as the first column.
    Additionally, in the file header, the key "> outputs = " (if found)
    determines the names of the dataframe columns.

    Parameters
    ------
    filename : str
        Name of the input file
    entry_char : str
        Character used to start an input line in the file header.
    attribution_char : str
        Character that separates the key name from its value (header only).
    comment_char : str
        Character that indicates a comment. Used both for header and data.
    header_end : str
        String that defines the separation between the header and the main data.
        It must contain '\n' at the end.
    outputs_key : str
        Keyword for the output topics, read from the file header.
    delimiter : str
        Horizontal delimiter between column entries
    names : list
        List of column names for the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Resulting data frame read from file.
    """

    # Reads the file header and interprets its valid entries.
    file_header = read_file_header(filename, header_end)
    header_size = len(file_header) + 1
    input_dict = read_config_strlist(file_header, entry_char, attribution_char,
                                     comment_char)

    # Reads the database from file and removes inexistent entries
    # deprecated from pandas 0.25: pd.read_table
    df = pd.read_csv(filename, sep=delimiter,
                       skiprows=header_size, header=None,
                       comment=comment_char, index_col=0)
    df = df.dropna(axis=1)

    # Sets the default value of the column 'names'.
    # If the file header has an "outputs" topic, uses as column names
    # Else, uses integer indexes (starting from 0).
    if names is None:
        try:
            names = read_csv_names(input_dict[outputs_key])
        except KeyError:
            names = list(range(len(df.columns)))

    df.columns = names  # Renames the columns

    return df


def xs_by_dict(df, param_dict, column_names=None):
    """
    Makes a cross section of a pandas multiindex dataframe by
    taking a dictionary of the parameters (index) names and their
    values.

    Parameters
    ----------
    df : pandas.DataFrame
        Multi-index data frame.
    param_dict : dict
        Dictionary with the level names (as keys) and their values to
        be accessed (as values).
    column_names : str, list
        Column name to be used. If not informed, all columns are used.
    """
    if column_names is None:
        return df.xs(tuple(param_dict.values()),
                     level=tuple(param_dict.keys()))
    else:
        return df.xs(tuple(param_dict.values()),
                     level=tuple(param_dict.keys()))[column_names]
