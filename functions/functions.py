# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:20:38 2017
@author: Dimitrios Zacharatos
"""
##########################################################################################
# LOAD
##########################################################################################
import numpy as np
import pandas as pd
import string as st
import random as rnd
from scipy.linalg import eigh, cholesky
from scipy.stats import norm
##########################################################################################
# POPULATE DATAFRAME
##########################################################################################
def populate_dataframe(ncols=5,nrows=5,value=np.nan):
    """
    Generate a DataFrame with specified number of rows and columns, initialized with a specified value.

    Parameters
    ----------
    ncols : int, optional, default=5
        The number of columns in the DataFrame. The column names will be integers from 1 to `ncols`.

    nrows : int, optional, default=5
        The number of rows in the DataFrame. Rows will be indexed from 1 to `nrows`.

    value : scalar, optional, default=np.nan
        The value with which to populate the DataFrame. All cells will be initialized to this value.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the specified shape (`nrows` by `ncols`) filled with the specified `value`.

    Examples
    --------
    >>> populate_dataframe(ncols=3, nrows=4, value=0)
       1  2  3
    1  0  0  0
    2  0  0  0
    3  0  0  0
    4  0  0  0

    >>> populate_dataframe(ncols=2, nrows=3, value="empty")
         1      2
    1  empty  empty
    2  empty  empty
    3  empty  empty
    """
    collumn_names=range(1,ncols+1)
    mydata=pd.DataFrame(columns=collumn_names)
    mydata=mydata.reindex(range(1,nrows+1))
    mydata[mydata.isnull()]=value
    return mydata
# populate_dataframe()
##########################################################################################
# REMOVE ZERO VARIANCE COLLUMNS
##########################################################################################
def remove_zero_variance_collumns(mydata):
    """
    Remove columns with zero variance from the DataFrame.

    Parameters
    ----------
    mydata : pandas.DataFrame
        The input DataFrame from which columns with zero variance will be removed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns that have zero variance removed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'A': [1, 1, 1],
    >>>     'B': [1, 2, 3],
    >>>     'C': [4, 4, 4]
    >>> })
    >>> remove_zero_variance_columns(df)
       B
    0  1
    1  2
    2  3

    >>> df2 = pd.DataFrame({
    >>>     'X': [5, 5, 5],
    >>>     'Y': [10, 20, 30]
    >>> })
    >>> remove_zero_variance_columns(df2)
       Y
    0  10
    1  20
    2  30
    """
    mydata=mydata.drop(mydata.std()[mydata.std()==0].index.values,axis=1)
    return mydata
# remove_zero_variance_collumns(pd.DataFrame({'col1':[1,1,1,1,1,1,1,1],'col2':[1,1,1,1,1,1,1,2]}))
##########################################################################################
# REMOVE NA COLLUMNS
##########################################################################################
def remove_na_collumns(mydata):
    """
    Remove columns that contain only NA (missing) values from the DataFrame.

    Parameters
    ----------
    mydata : pandas.DataFrame
        The input DataFrame from which columns with only NA values will be removed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns containing only NA values removed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'A': [None, None, None],
    >>>     'B': [1, 2, 3],
    >>>     'C': [None, None, None]
    >>> })
    >>> remove_na_columns(df)
       B
    0  1
    1  2
    2  3

    >>> df2 = pd.DataFrame({
    >>>     'X': [None, None, None],
    >>>     'Y': [None, None, None]
    >>> })
    >>> remove_na_columns(df2)
    Empty DataFrame
    Columns: []
    Index: [0, 1, 2]
    """
    mydata=mydata.dropna(axis=1,how='all')
    return mydata
# remove_na_collumns(pd.DataFrame({'col1':[1,1,1,1],'col2':[np.NaN,np.NaN,np.NaN,np.NaN]}))
##########################################################################################
# GENERATE MISSING DATA
##########################################################################################
def generate_missing(vector,c=1):
    """
    Generate missing (NaN) values in a given vector by randomly replacing values with NaN.

    Parameters
    ----------
    vector : numpy.ndarray
        The input vector (1D array) in which missing values will be introduced.
    c : int, optional, default: 1
        The number of missing values (NaNs) to introduce into the vector. 
        The default value is 1.

    Returns
    -------
    numpy.ndarray
        The input vector with randomly generated missing (NaN) values.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.array([1, 2, 3, 4, 5])
    >>> generate_missing(vector, c=2)
    array([ 1.,  2., nan,  4., nan])

    >>> vector2 = np.array([10, 20, 30, 40])
    >>> generate_missing(vector2, c=3)
    array([nan, nan, 30., nan])

    Notes
    -----
    - This function modifies the input vector in place by replacing `c` randomly selected values with NaN.
    - If `c` is greater than the size of the vector, it will replace as many values as possible (up to the vector's size).
    """
    vector.ravel()[np.random.choice(vector.size,c,replace=True)]=np.nan
    return vector
# generate_missing(np.random.randn(10))
##########################################################################################
# GENERATE MISSING DATAFRAME
##########################################################################################
def generate_missing_df(mydata,p1=.1,p2=.9):
    """
    Generate missing (NaN) values in a DataFrame by randomly masking elements.

    Parameters
    ----------
    mydata : pandas.DataFrame
        The input DataFrame in which missing values will be introduced.
    p1 : float, optional, default: 0.1
        The probability of a value being set to NaN (True in mask). Must be between 0 and 1.
    p2 : float, optional, default: 0.9
        The probability of a value being kept (False in mask). Must be between 0 and 1.
        p1 + p2 must equal 1.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with randomly generated missing (NaN) values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> generate_missing_df(df, p1=0.2, p2=0.8)
       0    1    2
    0  NaN  2.0  3.0
    1  4.0  5.0  6.0
    2  NaN  8.0  9.0

    Notes
    -----
    - This function generates missing values (NaN) in the DataFrame `mydata` by creating a boolean mask with 
      probabilities `p1` for NaN values and `p2` for kept values.
    - The mask is applied to the DataFrame using `.mask()` method, which sets the selected elements to NaN.
    - The sum of `p1` and `p2` must equal 1, as they represent complementary probabilities.
    """
    mask=np.random.choice([True,False],size=mydata.shape,p=[p1,p2])
    return mydata.mask(mask)
# generate_missing_df(generate_normal())
##########################################################################################
# GENERATE NORMAL
##########################################################################################
def generate_normal(ncols=5,nrows=5,mean=0,sd=1):
    """
    Populate a DataFrame with random normal values.

    Parameters
    ----------
    ncols : int
        The number of columns in the DataFrame.
    nrows : int
        The number of rows in the DataFrame.
    mean : float, optional, default: 0
        The mean of the normal distribution used to generate the values.
    sd : float, optional, default: 1
        The standard deviation of the normal distribution used to generate the values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of size `nrows` x `ncols` with random normal values.

    Examples
    --------
    >>> populate_dataframe_with_normal_values(3, 5, mean=0, sd=1)
            1         2         3
    1   -1.234   0.876   -0.345
    2    0.762  -1.456    2.341
    3    0.456   1.234   -0.567
    4   -0.123   0.345    1.456
    5    1.234  -0.789   -1.234

    Notes
    -----
    This function generates a DataFrame with `ncols` columns and `nrows` rows. Each column is populated 
    with random values sampled from a normal distribution with a specified `mean` and `sd`. The function 
    uses `numpy.random.normal` to generate the random values for each column.
    """
    mydata=populate_dataframe(ncols,nrows)
    for x in range(ncols+1):
        mydata[x]=np.random.normal(mean,sd,size=nrows)
    return(mydata)
# generate_normal()
##########################################################################################
# GENERATE UNIFORM
##########################################################################################
def generate_uniform(ncols=5,nrows=5,mini=0,maxi=1,decimals=2):
    """
    Populate a DataFrame with random uniform values.

    Parameters
    ----------
    ncols : int, optional, default: 5
        The number of columns in the DataFrame.
    nrows : int, optional, default: 5
        The number of rows in the DataFrame.
    mini : float, optional, default: 0
        The minimum value of the uniform distribution.
    maxi : float, optional, default: 1
        The maximum value of the uniform distribution.
    decimals : int, optional, default: 2
        The number of decimal places to round the values to.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of size `nrows` x `ncols` with random uniform values.

    Examples
    --------
    >>> generate_uniform(3, 5, mini=0, maxi=10)
            1      2      3
    1   3.14   7.22   2.56
    2   8.12   1.45   9.34
    3   0.67   6.43   5.12
    4   4.90   9.21   7.31
    5   2.34   0.88   6.17

    Notes
    -----
    This function generates a DataFrame with `ncols` columns and `nrows` rows, where each column is populated 
    with random values sampled from a uniform distribution between `mini` and `maxi`. The function uses 
    `numpy.random.uniform` to generate the random values for each column. After generation, the values are 
    rounded to the specified number of `decimals` places.

    The function first calls `populate_dataframe` to create an empty DataFrame with the specified shape, and 
    then it fills each column with uniform random values.
    """
    mydata=populate_dataframe(ncols,nrows)
    for x in range(ncols+1):
        mydata[x]=np.random.uniform(mini,maxi,size=nrows)
    mydata=mydata.round(decimals=decimals)
    return(mydata)
# generate_uniform()
##########################################################################################
# GENERATE FACTOR EXACT
##########################################################################################
def generate_factor_exact(name=[rnd.choice(st.ascii_uppercase) for _ in range(2)],length=10):
    """
    Generate a factor-like vector with repeated values.

    Parameters
    ----------
    name : list of str, optional, default=['A', 'B']
        A list of characters (strings) to be repeated in the vector. The default is a random selection of two uppercase ASCII letters.
    length : int, optional, default=10
        The length of the resulting vector. The length is divided by the number of elements in `name` to determine how many times to repeat each element.

    Returns
    -------
    numpy.ndarray
        A vector with repeated elements from `name`, with a total length of `length`.

    Examples
    --------
    >>> generate_factor_exact(length=6)
    array(['A', 'B', 'A', 'B', 'A', 'B'], dtype='<U1')

    >>> generate_factor_exact(name=['X', 'Y', 'Z'], length=9)
    array(['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'], dtype='<U1')
    """
    vector=np.repeat(name,repeats=length/len(name))
    return vector
# generate_factor_exact()
##########################################################################################
# GENERATE FACTOR RANDOMIZED
##########################################################################################
def generate_factor_randomized(name=[rnd.choice(st.ascii_uppercase) for _ in range(2)],length=10):
    """
    Generate a factor-like vector with randomized values.

    Parameters
    ----------
    name : list of str, optional, default=['A', 'B']
        A list of characters (strings) from which values are randomly selected to populate the vector.
        The default is a list of two randomly chosen uppercase ASCII letters.
    length : int, optional, default=10
        The length of the resulting vector, which determines how many random values will be generated from `name`.

    Returns
    -------
    numpy.ndarray
        A vector of length `length`, with randomly chosen values from `name`.

    Examples
    --------
    >>> generate_factor_randomized(length=6)
    array(['B', 'A', 'A', 'B', 'A', 'B'], dtype='<U1')

    >>> generate_factor_randomized(name=['X', 'Y', 'Z'], length=9)
    array(['Z', 'X', 'Y', 'Z', 'Z', 'Y', 'X', 'Y', 'X'], dtype='<U1')
    """
    vector=np.random.choice(name,size=length,replace=True)
    return vector
# generate_factor_randomized()
##########################################################################################
# RANDOM STRING
##########################################################################################
def random_string(name=st.ascii_uppercase,vector_length=10,string_length=10):
    """
    Generate a list of random strings.

    Parameters
    ----------
    name : str, optional, default=string.ascii_uppercase
        A string containing characters to be randomly selected to form the random strings.
        The default is the uppercase English alphabet (A-Z).
    vector_length : int, optional, default=10
        The number of random strings to generate.
    string_length : int, optional, default=10
        The length of each random string.

    Returns
    -------
    list of str
        A list containing `vector_length` random strings, each of length `string_length`.

    Examples
    --------
    >>> random_string(vector_length=5, string_length=4)
    ['NBXO', 'SMQT', 'ZJPB', 'DKLG', 'FWEJ']

    >>> random_string(name='abc123', vector_length=3, string_length=6)
    ['a1b1c1', 'b1a1c1', '1a1b1c']
    """
    vector=[''.join(rnd.choice(name) for _ in range(string_length)) for _ in range(vector_length)]
    return vector
# random_string()
##########################################################################################
# GENERATE MULTIPLE RESPONCE VECTOR
##########################################################################################
def generate_multiple_responce_vector(responces=range(4),responded=range(4),vector_length=10):
    """
    Generate a list of multiple response vectors with varying response counts.

    Parameters
    ----------
    responces : iterable, optional, default=range(4)
        A collection of possible responses to select from. Default is the range [0, 1, 2, 3].
    responded : iterable, optional, default=range(4)
        The possible number of responses to be included in each vector. Default is the range [0, 1, 2, 3].
    vector_length : int, optional, default=10
        The number of response vectors to generate.

    Returns
    -------
    list of str
        A list containing `vector_length` strings, where each string represents a response vector.
        The string will contain a set of selected responses, separated by spaces.

    Examples
    --------
    >>> generate_multiple_responce_vector(responces=[1,2,3], responded=[1,2], vector_length=3)
    ['3', '2 1', '1 3']

    >>> generate_multiple_responce_vector(responces=range(5), responded=[2, 3], vector_length=2)
    ['1 4', '2 0']
    """
    # responces=''.join(map(str,responces))
    # responded=''.join(map(str,responded))
    vector=[' '.join(str(np.random.choice(responces,size=np.random.choice(responded),replace=False))) for _ in range(vector_length)]
    # vector=[int(x) for x in vector]
    return vector
# generate_multiple_responce_vector()
##########################################################################################
# GENERATE CORRELATION MATRIX
##########################################################################################
def generate_correlation_matrix(correlation_martix,nrows=1000):
    """
    Generate a DataFrame with correlated data based on a given correlation matrix.

    Parameters
    ----------
    correlation_matrix : 2D array-like
        The correlation matrix that defines the correlations between the variables.
        This should be a square matrix with shape (n, n), where n is the number of variables.
        The diagonal elements should be 1, representing perfect correlation with themselves, and 
        the off-diagonal elements represent pairwise correlations between the variables.
    nrows : int, optional, default=1000
        The number of rows (samples) to generate. Each row represents a sample of correlated variables.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with `nrows` samples of data drawn from a multivariate normal distribution, 
        based on the specified correlation matrix. The number of columns in the DataFrame matches 
        the number of variables (i.e., the size of the correlation matrix).

    Examples
    --------
    >>> correlation_matrix = np.array([[1, 0.8], [0.8, 1]])
    >>> df = generate_correlation_matrix(correlation_matrix, nrows=100)
    >>> df.head()
        0       1
    0  0.48  1.15
    1  0.89  1.39
    2 -0.12  0.63
    3  0.03  0.68
    4  0.79  1.13
    """
    mydata=pd.DataFrame(np.random.multivariate_normal(mean=np.repeat(0,len(correlation_martix)),cov=correlation_martix,size=nrows))
    return mydata
# generate_correlation_matrix(generate_normal().corr()).corr()
##########################################################################################
# SIMULATE CORRELATION FROM SAMPLE
##########################################################################################
def simulate_correlation_from_sample(cordata,nrows=1000):
    """
    Simulate a dataset with the same mean and covariance structure as a given sample data.

    Parameters
    ----------
    cordata : pandas.DataFrame
        A DataFrame containing the sample data from which the mean and covariance matrix will be calculated.
        The dataset should have multiple variables (columns) for which the correlation structure needs to be simulated.
    nrows : int, optional, default=1000
        The number of rows (samples) to generate. Each row will represent a sample of the data with the same mean
        and covariance structure as the provided sample data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with `nrows` simulated samples, drawn from a multivariate normal distribution. The mean and 
        covariance structure of the simulated data will match the sample data (`cordata`).

    Examples
    --------
    >>> import pandas as pd
    >>> cordata = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    >>> simulated_data = simulate_correlation_from_sample(cordata, nrows=100)
    >>> simulated_data.head()
             0        1
    0  2.078467  2.653842
    1  2.567963  2.832753
    2  3.324701  2.099676
    3  1.929354  2.096974
    4  3.051563  1.982198
    """
    mydata=pd.DataFrame(np.random.multivariate_normal(mean=np.array(cordata.mean()),cov=cordata.cov(),size=nrows))
    return mydata
# simulate_correlation_from_sample(generate_correlation_matrix(generate_normal().corr()).corr())
##########################################################################################
# DISPLAY LOWER DIAGONAL
##########################################################################################
def matrix_triangle(m,triangle,off_diagonal=np.nan,value=np.nan):
    """
    Modify a matrix to retain only the lower or upper triangle, filling the diagonal and off-diagonal elements.

    Parameters
    ----------
    m : numpy.ndarray
        A square matrix (2D NumPy array) to be modified. The shape of the matrix must be `(n, n)`.
    triangle : {'lower', 'upper'}
        Specifies which triangle of the matrix to keep. If 'lower', the lower triangular part is kept,
        and the upper part is zeroed. If 'upper', the upper triangular part is kept, and the lower part is zeroed.
    off_diagonal : scalar, optional, default=np.nan
        The value to fill for the off-diagonal elements of the matrix.
    value : scalar, optional, default=np.nan
        The value to fill for the diagonal elements of the matrix.

    Returns
    -------
    numpy.ndarray
        A modified matrix with the specified triangular portion retained, and diagonal and off-diagonal elements filled.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> matrix_triangle(m, 'lower', off_diagonal=0, value=5)
    array([[5., 0., 0.],
           [4., 5., 0.],
           [7., 8., 5.]])
    
    >>> matrix_triangle(m, 'upper', off_diagonal=0, value=5)
    array([[5., 2., 3.],
           [0., 5., 6.],
           [0., 0., 5.]])
    """
    if triangle=="lower":
        m=np.tril(m)
        np.fill_diagonal(m,value)
    if triangle=="upper":
        m=np.triu(m)
        np.fill_diagonal(m,value)
    np.fill_diagonal(m,off_diagonal)
    return m
# m=generate_correlation_matrix(generate_normal().corr()).corr()
# print(matrix_triangle(pd.DataFrame.as_matrix(m),triangle="lower"))
# print(matrix_triangle(pd.DataFrame.as_matrix(m),triangle="upper"))
##########################################################################################
# DISPLAY UPPER DIAGONAL AND LOWER DIAGONAL
##########################################################################################
def display_upper_lower_diagonal(m_upper,m_lower,value=np.nan):
    """
    Combine the upper and lower triangular parts of two matrices, filling the diagonal elements with a specified value.

    Parameters
    ----------
    m_upper : numpy.ndarray
        The upper triangular matrix (2D NumPy array). The shape of the matrix must be `(n, n)`.
    m_lower : numpy.ndarray
        The lower triangular matrix (2D NumPy array). The shape of the matrix must be `(n, n)`.
    value : scalar, optional, default=np.nan
        The value to fill for the diagonal elements of the combined matrix.

    Returns
    -------
    numpy.ndarray
        A new matrix obtained by combining the upper and lower triangular parts, with diagonal elements filled with the specified value.

    Examples
    --------
    >>> import numpy as np
    >>> m_upper = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    >>> m_lower = np.array([[1, 0, 0], [4, 2, 0], [7, 8, 3]])
    >>> display_upper_lower_diagonal(m_upper, m_lower, value=0)
    array([[0., 2., 3.],
           [4., 0., 5.],
           [7., 8., 0.]])
    """
    lower=matrix_triangle(m=m_lower,value=value)
    upper=matrix_triangle(m=m_upper,value=value)
    m=upper+lower
    return m
# cordata1=generate_normal(ncols=3).corr().as_matrix(columns=None)
# cordata2=generate_normal(ncols=3).corr().as_matrix(columns=None)
##########################################################################################
# LIST TO NUMBER STRING
##########################################################################################
def list_to_number_string(value):
    """
    Convert a list or tuple to a string of numbers, or return the input value if it is not a list or tuple.

    Parameters
    ----------
    value : list or tuple
        The input value, which can be a list or tuple of numbers.

    Returns
    -------
    str or same type as input
        If the input is a list or tuple, returns a string representation of the numbers in the list or tuple, 
        with square brackets removed. If the input is not a list or tuple, returns the original value.

    Examples
    --------
    >>> list_to_number_string([1, 2, 3])
    '1, 2, 3'

    >>> list_to_number_string((4, 5, 6))
    '4, 5, 6'

    >>> list_to_number_string(42)
    42
    """
    if isinstance(value, (list, tuple)):
        return str(value)[1:-1]
    else:
        return value

