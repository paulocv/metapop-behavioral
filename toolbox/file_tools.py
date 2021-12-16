""" 
Version: 2.0 - First version to reveive a version number. :p
(1.x will be left for older non-registered versions).
"""
import json
import os
import sys
import numpy as np

SEP = "/"  # Use '/' for Linux and '\\' for Windows.
HEADER_END = "-----\n"

# --------------------------------
# FOLDER, FILE AND ARGV OPERATIONS
# --------------------------------


def file_overwrite_check(file_path):

    # Checks if the file already exists and prompt an action to the user.
    if os.path.exists(file_path):
        answer = input("\nWarning: file '{}' already exists, meaning that "
                       "you may be about to overwrite an older simulation.\n"
                       "Do you want to stop ('s'), rename ('r') or overwrite it"
                       "anyway ('o')?\n".format(file_path))
        # If the user wants to overwrite it anyway:
        if answer == "o":
            return file_path
        # If the user wants to rename the file (uses the same folder)
        elif answer == "r":
            file_path = os.path.dirname(file_path) + SEP
            file_path += input(file_path)
            return file_path
        # If the user wants to stop
        elif answer == "s":
            print("Quitting now...")
            quit()
        else:
            print("What did you type dude!?? I'll quit anyway, dumb...")
            quit()

    # If file does not exist, return its path anyway
    return file_path


def read_argv_optional(argi, dtype=None, default=None):
    """Reads an option from argv[argi]. Returns a default value
     if the argument is not present. Also converts to dtype otherwise.
    """
    try:
        if dtype is None:
            res = sys.argv[argi]
        else:
            res = dtype(sys.argv[argi])
    except IndexError:
        res = default

    return res


def read_flag_argument(argv, flag, optional=True, default=None):
    """
    Tries to read a flagged option from a list of strings - typically argv.
    OBS: python has the argparse module, which shall do the same job and even better.
    But this function may be easier for simple uses.

    Example
    -------
    "tmux a -t my_session"  --> argv = ["tmux", "a", "-t", "my_session"]
    Calling read_flag_options(argv, "-t") should return "my_session".

    Parameters
    ----------
    argv : list
        List of strings from which the argument is extracted.
    flag : str
        String that flags the desired option as the next entry. Must contain trailing "-" or "--".
    optional : bool
        If True, returns None for not-found option.
        If False, raises an error
    default : str
        If optional is True, specifies the default return value.
    """

    try:
        # Finds the flag in list. If not found, catches the ValueError.
        flag_index = argv.index(flag)
    except ValueError:
        if optional:
            return default
        else:
            raise ValueError("Hey, the required flag '{}' was not found.".format(flag))

    # Checks if given is was not the last position in list.
    if flag_index >= len(argv) - 1:
        raise ValueError("Hey, the flag '{}' is the last element in the given list, but I expected an "
                         "option to be passed after it.".format(flag))
    else:
        return argv[flag_index + 1]


def get_folder_name_from_argv(argi=2, root_folder="", argi_check=True):
    """Reads a folder path from argv (argv[2] by standard).
    Adds the separator character (/, \\) if it was forgotten.
    Checks if the folder exists, and creates it otherwise.

    If the corresponding position in argv is not informed, asks for the
    user the path of the folder, starting from a given root folder.
    """
    # First tries to read the output folder name from argv[2]
    try:
        output_folder = sys.argv[argi]
    except IndexError:
        if argi_check:
            # If argv[argi] was not passed, asks the user for the output folder.
            output_folder = root_folder
            output_folder += input("Output folder path was not informed. Please inform:\n"
                                   "{}".format(root_folder))
        else:
            raise IOError("Hey, argv[{}] was not found!".format(argi))

    # Adds the SEP (/ or \\) character to the end of the folder name.
    if output_folder[-len(SEP):] != SEP:
        output_folder += SEP

    # Checks if the folder does not exist. Creates it, in this case.
    if not os.path.exists(output_folder):
        os.system("mkdir -p '{}'".format(output_folder))

    return output_folder


def make_folder(folder_path, silent=True):
    """Creates given directory if non-existent"""
    if not os.path.exists(folder_path):
        os.system("mkdir -p '{}'".format(folder_path))
    elif not silent:
        print("Folder '{}' already exists.".format(folder_path))


def opt_make_folder_and_add_sep(folder_path, silent=True):
    """
    Creates given directory if non-existent. Adds the SEP character if not there at the ent. Returns new folder name.
    """
    if folder_path[-len(SEP):] != SEP:
        folder_path += SEP
    if not os.path.exists(folder_path):
        os.system("mkdir -p '{}'".format(folder_path))
    elif not silent:
        print("Folder '{}' already exists.".format(folder_path))

    return folder_path


# --------------------------------------------------------------------
# USEFUL STRING OPERATIONS
# --------------------------------------------------------------------


def remove_border_spaces(string):
    """ Strips the whitespace borders of a string.
    Used inside 'read_config_file' function.
    """
    if type(string) != str:
        raise TypeError(
            "Hey, a non-string object was passed to function "
            "'Remove_border_spaces'!")

    if string == '':
        return string

    while string[0] == ' ' or string[0] == '\t':
        string = string[1:]
        if string == '':
            return string

    while string[-1] == ' ' or string[-1] == '\t':
        string = string[:-1]
        if string == '':
            return string

    return string


def add_spaces_to_fill(text, places):
    """Adds blank spaces to a string so that the length of the final
    string is given by 'places'.
    """
    return text + (places-len(text))*" "


def str_to_list_json(s):
    """Converts a string to a python list using json.loads().
    Will only work for simple lists with objects supported by json.

    s : str
    Returns : list
    """
    s.replace("'", '"')
    return json.loads(s)


def str_to_list(string, key_name=""):
    """Evaluates a string as a list. Checks if border characters are
    '[' and ']', to avoid bad typing.

    key_name : string (optional)
        Name of the parameter. Useful for error message.
    """
    if string[0] == '[' and string[-1] == ']':
        return list(eval(string))
    else:
        raise ValueError("Hey, bad parameter or list of parameters"
                         " in {} = {}".format(key_name,
                                              string))


def str_to_tuple(string, key_name=""):
    """Evaluates a string as a tuple. Checks if border characters are
    '(' and ')', to avoid bad typing.

    key_name : string (optional)
        Name of the parameter. Useful for error message.
    """
    if string[0] == '(' and string[-1] == ')':
        return tuple(eval(string))
    else:
        raise ValueError("Hey, bad parameter or tuple of parameters"
                         " in {} = {}".format(key_name,
                                              string))


def str_to_bool(string, truelist=None):
    """Returns a boolean according to a string. True if the string
    belongs to 'truelist', false otherwise.
    By default, truelist has "True" only.
    """
    if truelist is None:
        truelist = ["True"]
    return string in truelist


def str_to_bool_safe(s, truelist=("True", "true", "T"), falselist=("False", "false", "F")):
    """
    Converts a boolean codified as a string. Instead of using 'eval', compares with lists of accepted strings for
    both true and false bools, and raises an error if the string does not match any case.

    Parameters
    ----------
    s : str
        The string to be read from
    truelist : tuple or list
        Tuple or list of strings interpreted as True.
    falselist : tuple or list
        Tuple or list of strings interpreted as False.

    Returns
    -------
    res : bool
    """
    if s in truelist:
        return True
    elif s in falselist:
        return False
    elif isinstance(s, bool):
        # In case the input is already a boolean.
        return s
    else:
        raise ValueError("Hey, the string '{}' could not be understood as a boolean.".format(s))


def str_to_dict(string, key_name=""):
    """Evaluates a string as a dict. Checks if border characters are
    '{' and '}', to avoid bad typing.

    key_name : string (optional)
        Name of the parameter. Useful for error message.

    Future idea: Actually split the string and eval according to types.
    """
    if string[0] == '{' and string[-1] == '}':
        d = eval(string)
        if type(d) is not dict:
            raise TypeError("Hey, could not interpret this as a dict: \n"
                            "'{}'".format(d))
    else:
        raise ValueError("Hey, bad parameter or tuple of parameters"
                         " in {} = {}".format(key_name,
                                              string))
    return d


# Float from strings are accepted as fractions, like 1 / 5.1
def float_as_frac(s):
    try:
        res = float(s)
    except ValueError:
        res = s.split("/")
        res = float(res[0]) / float(res[1])
    return res


def get_bool_from_dict(input_dict, key, truelist=None, raise_keyerror=False,
                       std_value=False):
    """Returns a boolean read from a string at an input dictionary.
    True if the string belongs to 'truelist', false otherwise.
    By default, truelist has "True" only.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.
    key : any hashable
        Keyword for the boolean to be read.
    truelist : list
        List of strings that are considered as True value.
    raise_keyerror : bool
        Defines if a key error is raised if the required key is not
        found on input_dict. Default is False, meaning no key error.
        In this case, the function returns std_value.
    std_value : bool
        Value to return if the key is not found and raise_keyerror=False.
    """
    if truelist is None:
        truelist = ["True"]

    try:
        return input_dict[key] in truelist
    except KeyError:
        if raise_keyerror:
            raise KeyError("Hey, key '{}' was not found on input dict."
                           "".format(key))
        else:
            return std_value


def seconds_to_hhmmss(time_s):
    """Converts a time interval given in seconds to hh:mm:ss.
    Input can either be a string or floating point.
    'ss' is rounded to the closest integer.

    Returns a string: []h[]m[]s
    """
    time_s = float(time_s)

    hh = int(time_s/3600)
    time_s = time_s % 3600
    mm = int(time_s/60)
    ss = round(time_s % 60)

    return "{}h{}m{}s".format(hh, mm, ss)


def list_to_csv(parlist, sep=", "):
    """Returns a csv string with elements from a list."""
    result_str = ""
    for par in parlist:
        result_str += str(par) + sep
    return result_str[:-len(sep)]


def read_csv_names(string, sep=","):
    """Reads multiple strings separated by commas and removes border spaces.
    Example:
        "beta, pj ,  num_steps" --> ['beta', 'pj', 'num_steps']
    """
    return [remove_border_spaces(name) for name in string.split(sep)]


def cast_to_export(value, float_fmt="{:12.6f}", int_fmt="{:12d}"):
    """
    Converts a given variable to a string in an adequate format for tabular files.
    """
    if isinstance(value, (float, np.floating)):
        out = float_fmt.format(value)
    elif isinstance(value, (int, np.integer)):
        out = int_fmt.format(value)
    else:
        out = str(value)
    return out


def cast_to_export_list(values, float_fmt="{:12.6f}", int_fmt="{:12d}", sep="\t") -> str:
    """
    Converts a list of values into strings to be exported as a fixed width string, using cast_to_export in each value.
    Adds a sep character after each entry.
    """
    out_str = ""
    for value in values:
        out_str += cast_to_export(value, float_fmt, int_fmt) + sep
    return out_str


# -------------------------------------------------------------------
# CONFIGURATION FILE AND INPUT DICIONARY common operations
# -------------------------------------------------------------------

def read_optional_from_dict(input_dict, key, standard_val=None,
                            typecast=None):
    """Tries to read an option from a dictionary. If not found, a
    standard value is returned instead. If no standard value is
    informed, None is returned. Data can also be converted by
    a type cast.
    The given standard value is not converted by typecast.

    WITH typecast=None, THIS CAN BE REPLACED BY dict.get() METHOD!
    """
    try:
        val = input_dict[key]
    except KeyError:
        # if standard_val is None:
        #     raise KeyError("Hey, parameter '{}' was not found on dict."
        #                    "".format(key))
        # else:
        return standard_val
    # Data conversion.
    if typecast is None:
        return val
    else:
        return typecast(val)


def read_kwargs_from_dict(input_dict, type_dict):
    """ Reads, from an input dictionary, a set of keyword arguments.
    This function is useful to gather optional arguments for a function call
    from a greater, more complete, input dictionary. For this purpose,
    keys that are not found at input_dict are simply ignored.

    The desired keywords mus be the keys of type_dict, whose values
    must be the type cast callable. None can be passed as a type cast
    for no conversion.

    Parameters
    ------
    input_dict : dict
        Dictionary with the input keywords. Can contain more than the
        necessary keywords.
    type_dict : dict
        Dictionary with the desired names as keys and their type casts
        as values. Type can be None for no conversion.
    """
    kwargs = dict()

    for key, cast in type_dict.items():
        try:
            if cast is None:
                kwargs[key] = input_dict[key]
            else:
                kwargs[key] = cast(input_dict[key])
        except KeyError:
            pass

    return kwargs


def read_args_from_dict(input_dict, arg_names, arg_types):
    """Reads *mandatory* arguments from an input dictionary.
    Useful for reading positional arguments for a function.

    All the names in arg_names must be present as keys of
    input_dict, and the number of elements of arg_names and
    arg_types must match. However, an item of arg_types can
    be passed as None to avoid its type cast.

    Returns a tuple with the gathered and casted parameters,
    sorted according to arg_names.

    Parameters
    ----------
    input_dict : dict
    arg_names : list, tuple
    arg_types : list, tuple
    """
    if len(arg_names) != len(arg_types):
        raise ValueError("Hey, the number of argument names ({}) and the "
                         "number of argument types ({}) do not match!"
                         "".format(len(arg_names), len(arg_types)))

    args = []

    for name, cast in zip(arg_names, arg_types):
        if cast is None:
            args.append(input_dict[name])
        else:
            args.append(cast(input_dict[name]))

    return tuple(args)


def read_config_file(file_path, entry_char='>', attribution_char='=',
                     comment_char='#', endline=None):
    """Function that reads 'markup' files.
     It opens the file and looks for lines with 'entry_char'. Example:
        > option_name: value  #This is a comment

    The ':' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    file_path : str
        Name of the file to be read.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    endline : str
        Termination line string. If this line is found on the file,
        the reading is terminated and the function returns the results
        gathered until this point.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file.
    """
    # File opening and storing
    fp = open(file_path, 'r')
    file_strlist = fp.read().splitlines()
    fp.close()

    return read_config_strlist(file_strlist, entry_char, attribution_char,
                               comment_char, endline)


def read_config_strlist(strlist, entry_char='>', attribution_char='=',
                        comment_char='#', endline=None):
    """ Similar to 'read_config_file', but works at a string or a list
    of strings instead.
    Function that reads 'markup' content from a list of strings.
    It looks for lines with 'entry_char'. Example:
        > option_name = value  #This is a comment

    The '=' can be replaced by any combination of characters specified
    as 'attribution_char' keyword argument.

    Inputs
    ----------
    strlist : list, str
        String list to be read. Optionally, if a single string is passed,
        it is transformed into a string list by spliting on '\n' chars.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    endline : str
        Termination line string. If this line is found on the string,
        the reading is terminated and the function returns the results
        gathered until this point.
        Obs: may or not contain the '\n' character at the end. Both
        cases work.

    Returns
    ----------
    result_dictio : dict
        Dictionary with all the options read from file.
    """

    # Checks if the input is a single string, and then splits it by \n.
    if type(strlist) == str:
        strlist = strlist.splitlines()

    # Size of attribution and entry chars (strings)
    entry_char_len = len(entry_char)
    attr_char_len = len(attribution_char)

    # Removes the '\n' at the end of 'endline'
    if endline is not None and len(endline) != 0:
        if endline[-1] == '\n':
            endline = endline[:-1]

    # Main loop over the list of strings
    result_dictio = {}
    for line in strlist:

        # Stops the loop if an 'endline' is found.
        if line == endline:
            break

        # Gets only lines which have the entry character at the start
        if line[0:entry_char_len] != entry_char:
            continue

        # Line text processing
        # Ignores everything after a comment character
        line = line.split(comment_char)[0]
        # Eliminates the initial (entry) character
        line = line[entry_char_len:]

        # Separation between key and value
        # Finds where is the attribution char, which separates key from
        # value.
        attr_index = line.find(attribution_char)
        # If no attribution char is found, raises an exception.
        if attr_index == -1:
            raise ValueError(
                "Heyy, the attribution character '" + attribution_char +
                "' was not found in line: '" + line + "'")
        key = remove_border_spaces(line[:attr_index])
        value = remove_border_spaces(line[attr_index + attr_char_len:])

        # Finally adds the entry to the dictionary
        result_dictio[key] = value

    return result_dictio


def entry_string(key, value, entry_char=">", attribution_char="=",
                 end_char="\n"):
    """Converts a keyword and a value to an accepted input string for
    'read_config_file.

    Inputs
    ----------
    key : str
        Keyword (name of the option/parameter). If not string, a
        conversion is attempted.
    value : str
        Value of the option/parameter. If not a string, a conversion
        is attempted.
    entry_char : str
        Character to start the line.
    attribution_char : str
        Character that separates the key from the value.
    end_char : str
        Character inserted at the end of the string.

    Returns
    ----------
    result_str : str
        String with an input line containing '> key = value'.
    """
    result_str = entry_char + " "
    result_str += str(key)
    result_str += " " + attribution_char + " "
    result_str += str(value)
    result_str += end_char
    return result_str


def write_config_string(input_dict, entry_char='>', attribution_char='=',
                        usekeys=None):
    """ Exports a dictionary of inputs to a string.

    Inputs
    ----------
    input_dict : dict
        Dictionary with the inputs to be exported to a string.
    entry_char : str
        Character used to start an input line in the config file.
    attrbution_char : str
        Character that separates the key name from its value.
    comment_char : str
        Character that indicates a commentary. Everything after this
        character is ignored.
    usekeys : list
        Use that input to select the input_dict entries that you want
        to export.
        Inform a list of the desired keys. Default is the whole dict.

    Returns
    -------
    result_str : str
        String with the formated inputs that were read from the input_dict.
    """
    # Selects the desired entries of the input_dict
    if usekeys is not None:
        input_dict = {key: input_dict[key] for key in usekeys}

    result_str = ""

    for key, value in input_dict.items():
        result_str += entry_string(key, value, entry_char, attribution_char)

    return result_str


def count_header_lines(file_name, header_end=HEADER_END):
    """Opens and reads a file until it finds a 'header finalization' line.
    By standard, such line is '-----\n' (five -).
    Returns only the number of lines in the header.
    After that operation, the file is closed.

    Parameters
    ----------
    file_name : str
        Path for the (closed) file.
    header_end : str
        String that marks the end of a header section.
        Must contain the '\n' at the end.
    """
    fp = open(file_name, "r")

    for i, line in enumerate(fp):  # Reads file until eof
        # Checks if the line is a header end
        if line == header_end:
            fp.close()
            return i + 1

    # If EOF is reached without finding the header end, assumes there is no header
    print("Hey, warning: no 'header_end' line was found in file."
          "It will be assumed that the file has no header (i.e., 0 "
          "header lines).\n"
          "File path: '{}'\n"
          "The expected header end was: '{}'".format(file_name, header_end)
          )
    fp.close()
    return 0


def read_file_header(filename, header_end=HEADER_END):
    """Reads a file until it finds a 'header finalization' line, and
    returns the read content.

    Parameters
    ----------
    filename : str
        Path for the file
    header_end : str
        String that marks the end of a header section.
        Must contain the '\n' at the end.

    Returns
    -------
    output : list
        A list with each line from the header. Does not contain the "header_end"
        line.
    """
    fp = open(filename, 'r')
    output = []

    # Reads file line once.
    line = fp.readline()

    while line:  # Reads until eof
        # Checks if the line is a header end
        if line == header_end:
            fp.close()
            return output
        output += [line[:-1]]  # Stores the line without the \n character
        line = fp.readline()

    # If EOF is reached without finding the header end, an error is raised.
    fp.close()
    raise EOFError("Hey, I did not find the header ending string on file:\n"
                   "File: '{}'\n"
                   "Ending str:'{}'\n".format(fp.name, header_end))

# -------------------------------
# ZIP/UNZIP FUNCTIONS
# -------------------------------


def zip_file(fname, remove_orig=True, level=5, verbose=False, ask_overwrite=False):
    """Compresses a single file using gzip.
    The compressed file name is [fname].gz
    """
    flags = ""
    if verbose:
        flags += "v"  # "verbose"
    if not ask_overwrite:
        flags += "f"  # "force"
    if not remove_orig:
        flags += "k"  # "keep"

    os.system("gzip -" + flags + " -{:d} ".format(level) + "-f " + fname)


def unzip_file(fname, remove_orig=True, verbose=False, ask_overwrite=False):
    """Decompresses a single file using gzip.
    The decompressed file name is [fname] minus the .gz suffix.
    """
    flags = "d"  # "decompress"
    if verbose:
        flags += "v"  # "verbose"
    if not ask_overwrite:
        flags += "f"  # "force"
    if not remove_orig:
        flags += "k"  # "keep"

    os.system("gzip -" + flags + " -f " + fname)


def tar_zip_folder(dirname, tarname=None, remove_orig=True, level=5, verbose=False):
    """Compresses a folder to a tar.gz file.

    Parameters
    ----------
    dirname : str
        Path of the folder to be compressed.
    tarname : str
        Path for the tar.gz file. Optional. If not informed, uses the
        name of the folder.
    remove_orig : bool
        Bool to remove the original folder after zipping.
        By default, it removes (True).
    level : int
        Level of compression, being 1 the fastest and 9 the best.
        Default is 5.
    """

    # Dirname here will not have the SEP character at the end
    if dirname[-1] == SEP:
        dirname = dirname[:-1]

    # If tar_name is not informed, uses directory name
    if tarname is None:
        tarname = dirname + ".tar"

    flags = "cz"  # -c = Create tar file. -z = zip (compress with gzip)
    if verbose:
        flags += "v"

    # This command tars and compresses the file at once
    os.system("GZIP=-{:d} ".format(level) + "tar -" + flags + " -f " + tarname + " " + dirname)
    # os.system("gzip -{:d} -f ".format(level) + tarname)  # -z flag

    # Removes the original folder
    if remove_orig:
        os.system("rm -r " + dirname)


def tar_unzip_folder(tarname, remove_orig=True, verbose=False):
    """Unzips and turns a .tar.gz file into a folder again."""

    flags = "xz"
    if verbose:
        flags += "v"

    # Unzips and untars
    os.system("tar -" + flags + " -f" + tarname)

    # Removes the original tar
    if remove_orig:
        # os.system("rm " + dirname + SEP + "*")
        os.system("rm " + tarname)
        

def tar_folder(dirname, tarname=None, remove_orig=True, verbose=False):
    """Produces a tarball from a folder, without compressing its contents"""
    # Dirname here will not have the SEP character at the end
    if dirname[-1] == SEP:
        dirname = dirname[:-1]

    parent_dir = os.path.dirname(dirname)  # Used to change (-C) to dir before creating tar
    basename = os.path.basename(dirname)

    # If tar_name is not informed, uses directory name
    if tarname is None:
        tarname = dirname + ".tar"

    flags = "c"  # -c = Create tar file.
    if verbose:
        flags += "v"

    # Command
    #  -C = changes to directory during tar.
    os.system("tar -C {} -{} -f {} {}".format(parent_dir, flags, tarname, basename))

    # Removes the original folder
    if remove_orig:
        os.system("rm -r " + dirname)


def untar_folder(tarname, remove_orig=True, verbose=False):
    """Reverses a tar object into a directory again, without unzipping it."""

    flags = "x"  # -x = extract.
    if verbose:
        flags += "v"

    parent_dir = os.path.dirname(tarname)

    # Command
    #  -C = changes to directory during tar.
    os.system("tar -C {} -{} -f {}".format(parent_dir, flags, tarname))

    # Removes the original tar
    if remove_orig:
        # os.system("rm " + dirname + SEP + "*")
        os.system("rm " + tarname)


def possibly_unzip_file(fname, zip_suffixes=(".gz", ".zip"), raise_error=True):
    """
    For a "regular" path fname, checks if it is actually ziped (i.e., fname + zip_suffix).

    Parameters
    ----------
    fname : str
        Path of the file without any zip suffix (such as .gz), regardless of existence.
    zip_suffixes : tuple
        Possible suffixes of the compressed file, such as .gz, .zip, .xz, .7z.
    raise_error : bool
        If True, raises an error if neither unzipped nor zipped versions of the file are found.

    Returns
    -------
    unzip_occurs : bool
        Returns true only if the decompression is executed.
    """
    # First checks if normal path exists
    if os.path.exists(fname):
        return False

    # Otherwise, checks if zipped file exists
    for s in zip_suffixes:
        if os.path.exists(fname + s):
            unzip_file(fname + s, remove_orig=False)
            return True

    # At this point, no version of the file was found. Either raises an error or leaves silently.
    if raise_error:
        raise FileNotFoundError("Hey, file '{}' was not found neither as it is, nor in a zipped format.\n"
                                "I've tried these suffixes: {}".format(fname, zip_suffixes))
    return False


def remove_file(fname):
    """Shorthand to remove file. Useful for modules that do not import 'os'."""
    return os.system("rm " + fname)
