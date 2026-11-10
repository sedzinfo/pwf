# -*- coding: utf-8 -*-
"""
Python port of R rwf::GLM_IRT_T.R: Thurstonian IRT (forced-choice and
ranking) utilities: comparison-matrix construction, rank-to-binary
recoding, the Thurstonian item characteristic curve, ML/MAP ability
scoring, and the MAP scorer of lavaan-fitted Thurstonian models.

extract_tirt_params and check_heywood work on a fitted model of pwf's
native lavaan port (_ordinal.py: thresholds, polychoric correlations,
ULSMV / WLSMV estimation with labels and linear constraints), which
reproduces thurstonianIRT::fit_TIRT_lavaan's lavaan fit
(tests/test_glm_irt_t.py); fit_tirt fits such a model from its lavaan
syntax. No R is needed.

Renamed parameters: any R parameter named `lambda` is `lambda_` here,
since `lambda` is a reserved word in Python.

R's two commented-out functions in this source file (get_mplus_thu_3t,
tirt_diagnose) are dead code in the R original itself and are not ported.

Preserved R quirks, by design:
  - generate_unique_comparisons_index(1) returns an empty (0, 2) result —
    a single item has no comparisons to make. Every function built on top
    of it (generate_comparisons_matrix, rank_to_binary, ...) inherits this
    edge case unchanged.
  - compute_icc_thurstonian(..., plot=False) returns {"icc": ..., "plot":
    False} — i.e. the "plot" key holds the literal boolean False (not
    None) when no plot was requested, exactly mirroring R's `plot` local
    variable being returned unmodified when the `if(plot)` branch never
    executes.
  - compute_ability's ability_ml/ability_map can be arrays of length > 1
    if the likelihood/posterior has tied maxima across theta grid points
    (matching R's which(x==max(x)), which returns every tied index, not
    just the first).
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from plotnine import ggplot, aes, geom_point, geom_line, theme_bw, labs

try:
    from ._ordinal import OrdinalSEM
except ImportError:
    from _ordinal import OrdinalSEM
##########################################################################################
# GENERATE INDEX FOR ITEM COMPARISONS
##########################################################################################
def compute_dummy_comparisons(items):
    """
    Number of unique pairwise comparisons among `items` items.

    Parameters:
    items (int): Number of items.

    Returns:
    float: items * (items - 1) / 2.

    Examples:
    >>> compute_dummy_comparisons(3)
    """
    return items * (items - 1) / 2


def generate_unique_comparisons_index(items):
    """
    All (i1, i2) index pairs with i1 < i2, 1-based, for `items` items.

    Parameters:
    items (int): Number of items.

    Returns:
    numpy.ndarray: Shape (n_pairs, 2), 1-based indices. Empty (shape
    (0, 2)) when items <= 1.

    Examples:
    >>> generate_unique_comparisons_index(3)
    """
    pairs = [(i1, i2) for i1 in range(1, items + 1) for i2 in range(1, items + 1) if i1 < i2]
    return np.array(pairs, dtype=int).reshape(-1, 2)
##########################################################################################
# INCREASE INDEX
##########################################################################################
def increase_index(blocks, items):
    """
    Build a (blocks, items) matrix of consecutive 1-based indices, one
    contiguous run of `items` values per row.

    Parameters:
    blocks (int): Number of rows/blocks.
    items (int): Number of consecutive indices per block.

    Returns:
    numpy.ndarray: Shape (blocks, items).

    Examples:
    >>> increase_index(3, 3)
    """
    rows = [np.arange(1 + b * items, 1 + b * items + items) for b in range(blocks)]
    return np.array(rows, dtype=int)
##########################################################################################
# GENERATE COMPARISONS MATRIX
##########################################################################################
def generate_comparisons_matrix(items):
    """
    Contrast matrix for all pairwise comparisons among `items` items: one
    row per comparison, +1 in the "winner" column, -1 in the "loser"
    column, 0 elsewhere.

    Parameters:
    items (int): Number of items.

    Returns:
    numpy.ndarray: Shape (n_pairs, items).

    Examples:
    >>> generate_comparisons_matrix(3)
    """
    comparisons = generate_unique_comparisons_index(items)
    result = np.zeros((comparisons.shape[0], items))
    for row, (i1, i2) in enumerate(comparisons):
        result[row, i1 - 1] = 1
        result[row, i2 - 1] = -1
    return result
##########################################################################################
# GENERATE MATRIX A
##########################################################################################
def generate_matrix_A(blocks=3, items=3):
    """
    Block-diagonal comparison (design) matrix for a Thurstonian model
    with `blocks` ranking blocks of `items` items each.

    Parameters:
    blocks (int, optional): Number of blocks. Defaults to 3.
    items (int, optional): Number of items per block. Defaults to 3.

    Returns:
    numpy.ndarray: Shape (n_pairs_per_block * blocks, items * blocks).

    Examples:
    >>> generate_matrix_A(blocks=3, items=3)
    """
    comparison = generate_comparisons_matrix(items)
    row_length = comparison.shape[0]
    alpha = np.zeros((row_length * blocks, items * blocks))
    col_index = increase_index(blocks=blocks, items=items)
    row_index = increase_index(blocks=blocks, items=row_length)
    for i in range(blocks):
        alpha[np.ix_(row_index[i] - 1, col_index[i] - 1)] = comparison
    return alpha
##########################################################################################
# GENERATE MATRIX LAMBDA HAT
##########################################################################################
def generate_matrix_lambda_hat(blocks=3, items=3):
    """
    Stack of `blocks` copies of the single-block comparison matrix
    (block-diagonal-without-the-diagonal-part; i.e. the same
    comparison contrast repeated once per block, row-stacked rather
    than block-diagonal — see generate_matrix_A for the block-diagonal
    version).

    Parameters:
    blocks (int, optional): Number of blocks. Defaults to 3.
    items (int, optional): Number of items per block. Defaults to 3.

    Returns:
    numpy.ndarray: Shape (n_pairs_per_block * blocks, items).

    Examples:
    >>> generate_matrix_lambda_hat(blocks=3, items=4)
    """
    comparison = generate_comparisons_matrix(items)
    return np.vstack([comparison] * blocks)
##########################################################################################
# RANK BLOCK TO BINARY
##########################################################################################
def rank_to_binary(mydata, items=None, reverse=True):
    """
    Convert a single ranking block (one row per respondent, one column
    per item, values are e.g. ranks or ratings) into Thurstonian binary
    pairwise-comparison outcomes.

    Parameters:
    mydata (pandas.DataFrame or array-like): One column per item.
    items (int, optional): Number of items (columns) in the block. If
        None (default), inferred from mydata's column count.
    reverse (bool, optional): If True (default), assumes higher values
        rank first (1 = "column i1 beat column i2"). If False, the
        binary outcomes are flipped.

    Returns:
    pandas.DataFrame: One column per comparison, named "i<i1><i2>".

    Examples:
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(12345)
    >>> mydata = pd.DataFrame({"i1": np.round(np.random.normal(2, 1, 10), 2),
    ...                        "i2": np.round(np.random.normal(2, 1, 10), 2),
    ...                        "i3": np.round(np.random.normal(2, 1, 10), 2)})
    >>> rank_to_binary(mydata, items=3)
    >>> rank_to_binary(mydata, items=3, reverse=False)
    """
    mydata = pd.DataFrame(mydata).reset_index(drop=True)
    if items is None:
        items = mydata.shape[1]
    index = generate_unique_comparisons_index(items)
    binary = {}
    for i1, i2 in index:
        name = f"i{i1}{i2}"
        binary[name] = (mydata.iloc[:, i1 - 1].to_numpy() > mydata.iloc[:, i2 - 1].to_numpy()).astype(int)
    result = pd.DataFrame(binary)
    if not reverse:
        result = 1 - result
    return result
##########################################################################################
# RANK DATAFRAME TO BINARY
##########################################################################################
def rank_df_to_binary(mydata, items, reverse=True):
    """
    Convert a full data frame of `blocks` ranking blocks (each with
    `items` columns) into Thurstonian binary pairwise-comparison
    outcomes for every block, column-concatenated.

    Parameters:
    mydata (pandas.DataFrame): blocks * items columns total.
    items (int): Number of items per block.
    reverse (bool, optional): Forwarded to rank_to_binary. Defaults to True.

    Returns:
    pandas.DataFrame: One column per comparison across all blocks.

    Examples:
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(12345)
    >>> mydata = pd.DataFrame({f"i{i+1}": np.random.normal(2, .5, 10) for i in range(6)})
    >>> rank_df_to_binary(mydata[["i1", "i2", "i3", "i4"]], 4)
    >>> rank_df_to_binary(mydata, 3)
    """
    mydata = pd.DataFrame(mydata).reset_index(drop=True)
    blocks = mydata.shape[1] // items
    index = increase_index(blocks=blocks, items=items)
    parts = [rank_to_binary(mydata.iloc[:, index[i] - 1], items, reverse) for i in range(blocks)]
    return pd.concat(parts, axis=1)
##########################################################################################
# TRIPLET PAIRS
##########################################################################################
def name_triplet_pairs(n, prefix="i", sep="", strict=True):
    """
    Pair labels ("i1i2", "i1i3", "i2i3", ...) from items grouped into
    consecutive triplets.

    Parameters:
    n (int or sequence of int): Total item count, or a vector of item
        indices (e.g. range(4, 19)).
    prefix (str, optional): Prefix before each item index. Defaults to "i".
    sep (str, optional): Separator between the two item labels in a pair.
        Defaults to "".
    strict (bool, optional): If True (default), raise when the item
        count isn't a multiple of 3. If False, silently drop leftover items.

    Returns:
    list of str: Length 3 * (number of complete triplets).

    Examples:
    >>> name_triplet_pairs(15)
    >>> name_triplet_pairs(6, prefix="i", sep="_")
    >>> name_triplet_pairs(range(4, 10))
    >>> name_triplet_pairs(10, strict=False)
    """
    if np.isscalar(n):
        if n % 3 != 0:
            if strict:
                raise ValueError("n must be a multiple of 3")
            n = n - (n % 3)
        items = list(range(1, n + 1))
    else:
        items = list(n)
        if len(items) % 3 != 0:
            if strict:
                raise ValueError("length(items) must be a multiple of 3")
            items = items[:len(items) - (len(items) % 3)]

    triplets = [items[i:i + 3] for i in range(0, len(items), 3)]
    out = []
    for g in triplets:
        if len(g) < 3:
            continue
        for x, y in combinations(g, 2):
            out.append(f"{prefix}{x}{sep}{prefix}{y}")
    return out
##########################################################################################
# RANK BINARY TO TRIPLETS
##########################################################################################
def rank3_to_triplets(mydata):
    """
    Decode 3 binary Thurstonian pairwise comparisons (i1 vs i2, i1 vs
    i3, i2 vs i3) back into per-item rank positions (1=lowest, 3=highest).

    Parameters:
    mydata (array-like): 3 columns of binary comparisons, in the order
        produced by rank_to_binary for a 3-item block.

    Returns:
    pandas.DataFrame: Columns item1, item2, item3 — each item's rank
    (1-3) per respondent.

    Examples:
    >>> import pandas as pd, numpy as np
    >>> np.random.seed(12345)
    >>> mydata = pd.DataFrame({f"i{i+1}": np.random.normal(2, .5, 10) for i in range(6)})
    >>> result = rank_to_binary(mydata.iloc[:, :3])
    >>> rank3_to_triplets(result)
    """
    mydata = np.asarray(mydata)
    c1, c2, c3 = mydata[:, 0], mydata[:, 1], mydata[:, 2]
    item1 = np.select([(c1 == 1) & (c2 == 1), (c1 == 1) & (c2 == 0),
                        (c1 == 0) & (c2 == 1), (c1 == 0) & (c2 == 0)], [3, 2, 2, 1])
    item2 = np.select([(c1 == 0) & (c3 == 1), (c1 == 0) & (c3 == 0),
                        (c1 == 1) & (c3 == 1), (c1 == 1) & (c3 == 0)], [3, 2, 2, 1])
    item3 = np.select([(c2 == 0) & (c3 == 0), (c2 == 1) & (c3 == 0),
                        (c2 == 0) & (c3 == 1), (c2 == 1) & (c3 == 1)], [3, 2, 2, 1])
    return pd.DataFrame({"item1": item1, "item2": item2, "item3": item3})
##########################################################################################
# RESPONSE DIMENSION
##########################################################################################
def response_dimension(response, dimensions, items):
    """
    From successive non-overlapping blocks of `dimensions` consecutive
    elements of `response`, pick out the elements at the (1-based)
    within-block positions given by `items`.

    Parameters:
    response (array-like): Values to select from, e.g. range(1, 19).
    dimensions (int): Block size.
    items (sequence of int): 1-based positions within each block to keep.

    Returns:
    numpy.ndarray

    Examples:
    >>> response_dimension(range(1, 19), 3, [1, 2])
    >>> response_dimension(range(1, 19), 3, [1, 3])
    >>> response_dimension(range(1, 19), 3, [2, 3])
    """
    response = np.asarray(list(response))
    items0 = [it - 1 for it in items]
    n_iters = len(response) // dimensions
    out = []
    for i in range(n_iters):
        block = response[i * dimensions: i * dimensions + dimensions]
        out.extend(block[items0])
    return np.array(out)
##########################################################################################
# INDEX FROM LAVAAN TO THURSTONIAN
##########################################################################################
def cfa_icc_index(nitems, nfactors=3):
    """
    Index to reorder items from a lavaan-style block layout (all items
    of factor 1, then all of factor 2, ...) into Thurstonian block order
    (factor 1/2/3 item, factor 1/2/3 item, ...).

    Parameters:
    nitems (int): Total number of items.
    nfactors (int, optional): Number of factors/traits. Defaults to 3.

    Returns:
    dict: {"index_vector": 1D numpy.ndarray, "index_matrix": 2D
    numpy.ndarray of shape (nitems/nfactors, nfactors)}.

    Examples:
    >>> cfa_icc_index(nitems=18, nfactors=3)
    """
    block = nitems // nfactors
    index_matrix = np.column_stack([np.arange(1 + f * block, 1 + f * block + block) for f in range(nfactors)])
    index_vector = index_matrix.flatten(order="C")
    return {"index_vector": index_vector, "index_matrix": index_matrix}
##########################################################################################
# ITEM CHARACTERISTIC CURVE FOR CFA INPUT
##########################################################################################
def icc_cfa(eta, gamma, lambda_, psi):
    """
    Thurstonian/CFA-parameterized item characteristic curve:
    P = Phi((-gamma + lambda_ * eta) / sqrt(psi)).

    Parameters:
    eta (float or array-like): Latent trait value(s).
    gamma (float): Threshold parameter.
    lambda_ (float): Loading parameter.
    psi (float): Residual (error) variance.

    Returns:
    float or numpy.ndarray

    Examples:
    >>> import numpy as np
    >>> icc_cfa(np.arange(-6, 6.1, .1), 1, 1, 1)
    """
    eta = np.asarray(eta, dtype=float)
    return norm.cdf((-gamma + lambda_ * eta) / np.sqrt(psi))
##########################################################################################
# COMPUTE ICC THURSTONIAN
##########################################################################################
def compute_icc_thurstonian(eta, gamma, lambda_, psi, plot=False):
    """
    Item characteristic curves for a set of binary Thurstonian-coded
    items along a single dimension.

    Parameters:
    eta (array-like): Latent trait grid.
    gamma (array-like): Threshold parameters, one per item.
    lambda_ (array-like): Loading parameters, one per item.
    psi (array-like): Residual variances, one per item.
    plot (bool, optional): If True, also build the ICC plot via
        plot_icc_thurstonian. Defaults to False.

    Returns:
    dict: {"icc": pandas.DataFrame (columns item1..itemN, eta), "plot":
    plotnine.ggplot if plot=True, else the literal value False — see
    module docstring}.

    Examples:
    >>> gamma = [0.556,-1.253,-1.729,0.618,0.937,0.295,-0.672,-1.127]
    >>> psi = [2.172,1.883,2.055,1.869,2.231,2.100,1.762,1.803]
    >>> lambda_ = [1.082,1.082,-1.297,-1.297,0.802,0.802,1.083,1.083]
    >>> import numpy as np
    >>> compute_icc_thurstonian(eta=np.arange(-6, 6.01, .01), gamma=gamma, lambda_=lambda_, psi=psi, plot=False)
    """
    eta = np.asarray(eta, dtype=float)
    items = {f"item{i + 1}": icc_cfa(eta, gamma[i], lambda_[i], psi[i]) for i in range(len(lambda_))}
    icc = pd.DataFrame(items)
    icc["eta"] = eta
    plot_obj = plot_icc_thurstonian(icc, title="Item Characteristic Curve") if plot else False
    return {"icc": icc, "plot": plot_obj}
##########################################################################################
# PLOT ICC THURSTONIAN
##########################################################################################
def plot_icc_thurstonian(mydata, title="Item Characteristic Curve"):
    """
    Plot item characteristic curves produced by compute_icc_thurstonian.

    Parameters:
    mydata (pandas.DataFrame): Must contain an "eta" column plus one or
        more value columns to plot against it.
    title (str, optional): Plot title. Defaults to "Item Characteristic Curve".

    Returns:
    plotnine.ggplot

    Examples:
    >>> result = compute_icc_thurstonian(eta=range(-6, 7), gamma=[.5], lambda_=[1], psi=[1])
    >>> plot_icc_thurstonian(result["icc"])
    """
    melted = mydata.melt(id_vars="eta", var_name="variable", value_name="value")
    return (ggplot(melted, aes(x="eta", y="value", group="variable", color="variable"))
            + geom_point(alpha=.1)
            + geom_line()
            + theme_bw()
            + labs(y=r"$P(\eta)$", x=r"$\eta$", title=title))
##########################################################################################
# COMPUTE MAP
##########################################################################################
def compute_map(eta, mean=0, sd=1):
    """
    Normalized prior density (sums to 1) over a theta grid — the
    "MAP prior weight" used by compute_ability.

    Parameters:
    eta (array-like): Theta grid.
    mean (float, optional): Prior mean. Defaults to 0.
    sd (float, optional): Prior SD. Defaults to 1.

    Returns:
    numpy.ndarray

    Examples:
    >>> import numpy as np
    >>> compute_map(eta=np.arange(-6, 6.1, .1), mean=0, sd=1)
    """
    eta = np.asarray(eta, dtype=float)
    prior_density = norm.pdf(eta, loc=mean, scale=sd)
    return prior_density / np.sum(prior_density)
##########################################################################################
# COMPUTE ABILITY
##########################################################################################
def compute_ability(response, eta, gamma, lambda_, psi, plot=False, map=None):
    """
    ML and MAP ability estimates for one respondent's binary Thurstonian
    item responses, via brute-force grid search over `eta`.

    Parameters:
    response (array-like): Binary responses, one per item.
    eta (array-like): Theta grid to search over.
    gamma, lambda_, psi (array-like): Item parameters (see icc_cfa).
    plot (bool, optional): If True, print/return an annotated ICC plot.
        Defaults to False.
    map (array-like, optional): Prior weights over `eta`. If None
        (default), computed via compute_map(eta, mean=0, sd=1).

    Returns:
    dict: {"product": likelihood at each theta, "icc": DataFrame with
    product/map columns prepended, "ability_ml": theta(s) maximizing the
    likelihood, "ability_map": theta(s) maximizing likelihood*prior}.

    Examples:
    >>> gamma = [0.556,-1.253,-1.729,0.618,0.937,0.295,-0.672,-1.127]
    >>> psi = [2.172,1.883,2.055,1.869,2.231,2.100,1.762,1.803]
    >>> lambda_ = [1.082,1.082,-1.297,-1.297,0.802,0.802,1.083,1.083]
    >>> import numpy as np
    >>> eta = np.arange(-6, 6.1, .1)
    >>> map_prior = compute_map(eta=eta, mean=0, sd=1)
    >>> compute_ability([0,0,0,0,0,0,0,0], eta, gamma, lambda_, psi, map=map_prior, plot=False)
    """
    if map is None:
        map = compute_map(eta=eta, mean=0, sd=1)
    response = np.asarray(response, dtype=float)
    result = compute_icc_thurstonian(eta, gamma, lambda_, psi, plot=False)
    icc = result["icc"]

    product = np.ones(len(icc))
    for i in range(len(response)):
        col = icc.iloc[:, i].to_numpy()
        product = product * (col ** response[i]) * ((1 - col) ** (1 - response[i]))
    product_map = product * np.asarray(map)

    eta_vals = icc["eta"].to_numpy()
    ability_ml = eta_vals[product == product.max()]
    ability_map = eta_vals[product_map == product_map.max()]

    plot_display = np.array(product, dtype=float)
    plot_display_map = np.array(product_map, dtype=float)
    plot_obj = False
    if plot:
        while plot_display.max() < .1:
            plot_display = plot_display * 10
        while plot_display_map.max() < .1:
            plot_display_map = plot_display_map * 10
        icc_df = icc.copy()
        icc_df.insert(0, "map", plot_display_map)
        icc_df.insert(0, "product", plot_display)
        title = (f"Item Characteristic Curve ML: {np.round(ability_ml, 2)} "
                 f"MAP: {np.round(ability_map, 2)} \nResponse: {response.sum()} "
                 f"Response Length: {len(response)}")
        plot_obj = plot_icc_thurstonian(icc_df, title=title)
    else:
        icc_df = icc.copy()
        icc_df.insert(0, "map", product_map)
        icc_df.insert(0, "product", product)

    return {"product": product, "icc": icc_df, "ability_ml": ability_ml, "ability_map": ability_map}
##########################################################################################
# COMPUTE SCORES
##########################################################################################
def compute_scores(mydata, **kwargs):
    """
    compute_ability's MAP estimate for every row (respondent) in `mydata`.

    Parameters:
    mydata (pandas.DataFrame): One row per respondent, one column per
        binary item response.
    **kwargs: Forwarded to compute_ability (eta, gamma, lambda_, psi,
        map, plot).

    Returns:
    numpy.ndarray: One MAP ability estimate per row.

    Examples:
    >>> gamma = [0.556,-1.253,-1.729,0.618,0.937,0.295,-0.672,-1.127]
    >>> psi = [2.172,1.883,2.055,1.869,2.231,2.100,1.762,1.803]
    >>> lambda_ = [1.082,1.082,-1.297,-1.297,0.802,0.802,1.083,1.083]
    >>> import numpy as np, pandas as pd
    >>> eta = np.arange(-6, 6.1, .1)
    >>> map_prior = compute_map(eta=eta, mean=0, sd=1)
    >>> response_df = pd.DataFrame([[0]*8, [1]*8, [1,0]*4, [0,1]*4])
    >>> compute_scores(response_df, eta=eta, gamma=gamma, lambda_=lambda_, psi=psi, map=map_prior, plot=False)
    """
    mydata = pd.DataFrame(mydata)
    ability = []
    for i in range(len(mydata)):
        response = mydata.iloc[i].to_numpy(dtype=float)
        result = compute_ability(response, **kwargs)
        ability.extend(np.atleast_1d(result["ability_map"]).tolist())
    return np.array(ability)
##########################################################################################
# SOLVE
##########################################################################################
def compute_solve(a, b=None):
    """
    Solve A X = B via Gauss-Jordan elimination with partial pivoting, or
    invert A when B is omitted.

    Parameters:
    a (array-like): Square matrix A.
    b (array-like, optional): Vector or matrix B. If None (default), the
        identity matrix is used (so the result is the inverse of A).

    Returns:
    numpy.ndarray: X solving A X = B (1D if b was 1D; a plain matrix
    inverse if b was omitted).

    Examples:
    >>> import numpy as np
    >>> A = np.array([[2., 1.], [1., 3.]])
    >>> b = np.array([1., 2.])
    >>> x = compute_solve(A, b)
    >>> A @ x  # should equal b
    >>> B = np.column_stack([[1., 2.], [0., 1.]])
    >>> X = compute_solve(A, B)
    >>> A @ X  # should equal B
    >>> A_inv = compute_solve(A)
    >>> A @ A_inv  # should be the identity
    """
    a = np.array(a, dtype=float)
    n = a.shape[0]
    if a.shape[1] != n:
        raise ValueError("'a' must be square")

    b_was_vector = b is not None and np.ndim(b) == 1
    if b is None:
        b = np.eye(n)
    else:
        b = np.array(b, dtype=float)
        if b.ndim == 1:
            b = b.reshape(-1, 1)
    if b.shape[0] != n:
        raise ValueError("'a' and 'b' have incompatible dimensions")

    m = np.hstack([a, b])
    for i in range(n):
        pivot = i + int(np.argmax(np.abs(m[i:n, i])))
        if m[pivot, i] == 0:
            raise ValueError("matrix is singular")
        if pivot != i:
            m[[i, pivot], :] = m[[pivot, i], :]
        m[i, :] = m[i, :] / m[i, i]
        for j in range(n):
            if j != i:
                m[j, :] = m[j, :] - m[j, i] * m[i, :]

    x = m[:, n:]
    return x.ravel() if b_was_vector else x
##########################################################################################
# SCORE RESPONSE PATTERN
##########################################################################################
def score_tirt_pattern(pattern, lambda_, theta_diag, tau, Psi, nu=None, init=None, control=None):
    """
    MAP (empirical Bayes modal) latent trait estimate for one Thurstonian
    IRT response pattern, given lavaan-fitted measurement parameters.

    Note: this is an independent, hand-rolled BFGS-based MAP scorer, not
    a port of thurstonianIRT's own predict() method — R's own docstring
    for this function invites exactly this comparison ("Check same
    scores from thurstonianIRT package"), and running both confirms they
    are meaningfully different (different estimators, e.g. differing in
    how the prior/latent covariance is used), not that one is a bug.
    This Python port was verified to match R's score_tirt_pattern (the
    function being ported) to ~5-6 decimal places, not thurstonianIRT::predict().

    Parameters:
    pattern (array-like): Observed responses (0/1), NaN allowed for
        missing (ignored in the likelihood). Must be aligned to the row
        order of `lambda_`.
    lambda_ (array-like): Loading matrix (indicators x traits).
    theta_diag (array-like): Residual variances, aligned to lambda_ rows.
    tau (array-like): Thresholds, aligned to lambda_ rows.
    Psi (array-like): Latent trait covariance matrix.
    nu (array-like, optional): Indicator intercepts. If None (default),
        zeros are used.
    init (array-like, optional): Optimizer starting values. If None
        (default), zeros.
    control (dict, optional): Extra scipy.optimize.minimize options,
        merged over defaults {"gtol": 1e-10, "maxiter": 500}.

    Returns:
    numpy.ndarray: MAP trait estimates, length = number of traits.

    Examples:
    >>> import numpy as np
    >>> lambda_ = np.array([[1.,0.],[0.,1.],[1.,-1.]])
    >>> theta_diag = np.array([1.,1.,1.])
    >>> tau = np.array([0.,0.,0.])
    >>> Psi = np.eye(2)
    >>> score_tirt_pattern([1,0,1], lambda_, theta_diag, tau, Psi)
    """
    lambda_ = np.array(lambda_, dtype=float)
    n_traits = lambda_.shape[1]
    if nu is None:
        nu = np.zeros(lambda_.shape[0])
    pattern = np.asarray(pattern, dtype=float)
    theta_diag = np.asarray(theta_diag, dtype=float)
    tau = np.asarray(tau, dtype=float)
    nu = np.asarray(nu, dtype=float)

    obs = ~np.isnan(pattern)
    L = lambda_[obs, :]
    y = pattern[obs]
    s = np.sqrt(theta_diag[obs])
    th = tau[obs] - nu[obs]
    iPsi = compute_solve(Psi)

    def nll(eta):
        z = (L @ eta - th) / s
        p = np.clip(norm.cdf(z), 1e-15, 1 - 1e-15)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
        lp = -0.5 * np.sum(eta * (iPsi @ eta))
        return -(ll + lp)

    def gnll(eta):
        z = (L @ eta - th) / s
        p = np.clip(norm.cdf(z), 1e-15, 1 - 1e-15)
        phi = norm.pdf(z)
        w = (y - p) * phi / (p * (1 - p)) / s
        return -(L.T @ w - iPsi @ eta)

    if init is None:
        init = np.zeros(n_traits)
    options = {"gtol": 1e-10, "maxiter": 500}
    if control:
        options.update(control)

    res = minimize(nll, init, jac=gnll, method="BFGS", options=options)
    return res.x
##########################################################################################
# SCORE MULTIPLE RESPONSE PATTERNS
##########################################################################################
def score_tirt(patterns, lambda_, theta_diag, tau, Psi, nu=None):
    """
    score_tirt_pattern applied to every row of `patterns`, with columns
    aligned to lambda_'s row order by name when both are available.

    Parameters:
    patterns (pandas.DataFrame or array-like): One row per respondent.
    lambda_ (pandas.DataFrame or array-like): Loading matrix.
    theta_diag, tau, Psi, nu: See score_tirt_pattern.

    Returns:
    numpy.ndarray: Shape (n_respondents, n_traits).

    Examples:
    >>> import numpy as np, pandas as pd
    >>> lambda_ = pd.DataFrame([[1.,0.],[0.,1.],[1.,-1.]], index=["a","b","c"])
    >>> patterns = pd.DataFrame({"a":[1,0],"b":[0,1],"c":[1,0]})
    >>> score_tirt(patterns, lambda_, theta_diag=[1.,1.,1.], tau=[0.,0.,0.], Psi=np.eye(2))
    """
    lambda_names = list(lambda_.index) if isinstance(lambda_, pd.DataFrame) else None
    pattern_cols = list(patterns.columns) if isinstance(patterns, pd.DataFrame) else None
    lambda_arr = lambda_.to_numpy() if isinstance(lambda_, pd.DataFrame) else np.asarray(lambda_, dtype=float)
    trait_names = list(lambda_.columns) if isinstance(lambda_, pd.DataFrame) else None

    if lambda_names is None or pattern_cols is None:
        import warnings
        warnings.warn("lambda rows or pattern columns are unnamed; assuming positional alignment.")
        patterns_arr = np.asarray(patterns, dtype=float)
    else:
        missing = [name for name in lambda_names if name not in pattern_cols]
        if missing:
            raise ValueError(f"Pair names in lambda not found in patterns:\n  {', '.join(missing)}")
        patterns_arr = patterns[lambda_names].to_numpy(dtype=float)

    scores = np.array([score_tirt_pattern(row, lambda_arr, theta_diag, tau, Psi, nu) for row in patterns_arr])
    if trait_names is not None:
        return pd.DataFrame(scores, columns=trait_names)
    return scores
##########################################################################################
# FIT A THURSTONIAN IRT MODEL (native lavaan WLSMV/ULSMV, see _ordinal.py)
##########################################################################################
def fit_tirt(model, data, estimator="ULSMV"):
    """
    Fit a Thurstonian IRT model given in lavaan syntax to binary pairwise
    comparisons, as thurstonianIRT::fit_TIRT_lavaan does with lavaan:
    lavaan::lavaan(model, data, ordered = all, auto.fix.first = FALSE,
    auto.th = TRUE, parameterization = "theta", estimator = estimator).
    Python-only helper (rwf takes the fit from thurstonianIRT); the
    estimation is pwf's native port of lavaan's WLSMV/ULSMV.

    Parameters:
    model (str): lavaan syntax, e.g. from thurstonianIRT's make_lavaan_code.
    data (pandas.DataFrame): The binary comparison columns (e.g. i1i2, i1i3, ...).
    estimator (str, optional): "ULSMV" (thurstonianIRT's default) or "WLSMV".

    Returns:
    OrdinalSEM: The fitted model (parameter_estimates(), fit_measures(), inspect()).

    Examples:
    >>> fit = fit_tirt(open("tirt_lavaan_model.txt").read(), triplets)
    >>> extract_tirt_params(fit)
    """
    return OrdinalSEM(model, data, estimator=estimator, parameterization="theta", auto=False)


def _fit_of(fit_object):
    """The model fit of a thurstonianIRT-like object (fit_object$fit in rwf) or the fit itself."""
    if isinstance(fit_object, dict) and "fit" in fit_object:
        return fit_object["fit"]
    return getattr(fit_object, "fit", fit_object)
##########################################################################################
# EXTRACT TIRT PARAMETERS
##########################################################################################
def extract_tirt_params(fit_lavaan_obj):
    """
    Measurement parameters of a fitted Thurstonian IRT model, as
    rwf::extract_tirt_params: from lavInspect(fit, "est"), the loadings,
    the residual variances, the thresholds and the intercepts of every
    indicator (aligned to the loadings' rows) and the trait covariances.

    Parameters:
    fit_lavaan_obj: A model fitted with fit_tirt (pwf's native lavaan), or
        an object holding it as .fit / ["fit"] (rwf's fit_lavaan_obj$fit).

    Returns:
    dict: lambda (DataFrame indicators x traits), theta_diag, tau and nu
    (Series by indicator) and Psi (DataFrame traits x traits).

    Examples:
    >>> pars = extract_tirt_params(fit_tirt(model, triplets))
    """
    est = _fit_of(fit_lavaan_obj).inspect("est")
    lambda_ = est["lambda"]
    ind = list(lambda_.index)
    theta_diag = pd.Series(np.diag(est["theta"].to_numpy()), index=est["theta"].index).reindex(ind)
    tau = est["tau"]["threshold"]
    tau = pd.Series(tau.to_numpy(), index=[name.split("|")[0] for name in tau.index]).reindex(ind)
    nu = est["nu"]["intercept"].reindex(ind)
    return {"lambda": lambda_, "theta_diag": theta_diag, "tau": tau, "nu": nu, "Psi": est["psi"]}
##########################################################################################
# CHECK HEYWOOD CASES AND MODEL ISSUES
##########################################################################################
def check_heywood(fit_model, verbose=True):
    """
    Screen a fitted model for Heywood cases and related estimation
    problems, as rwf::check_heywood: negative variances (~~), negative
    scaling factors (~*~), standardized loadings or correlations beyond
    +-1, standard errors above 10, and non-convergence.

    The standardized values (est.std) are the std.all column of
    parameter_estimates(); their pvalue column is the unstandardized test
    (lavaan's standardizedSolution tests the standardized values).

    Parameters:
    fit_model: A model fitted with pwf's native lavaan (fit_tirt, _ordinal.OrdinalSEM).
    verbose (bool, optional): Print each problem found and a summary. Defaults to True.

    Returns:
    dict: has_issues (bool), issues (dict of DataFrames or a message) and converged (bool).

    Examples:
    >>> check_heywood(fit_tirt(model, triplets), verbose=True)
    """
    fit = _fit_of(fit_model)
    params = fit.parameter_estimates(standardized=True)
    std = params.rename(columns={"std.all": "est.std"})
    issues = {}

    def report(name, rows, title, columns):
        if len(rows):
            issues[name] = rows
            if verbose:
                print(f"\n=== {title} ===")
                print(rows[columns])

    report("negative_variances", params[(params["op"] == "~~") & (params["lhs"] == params["rhs"]) &
                                        (params["est"] < 0)],
           "NEGATIVE VARIANCES (~~)", ["lhs", "op", "rhs", "est", "se", "pvalue"])
    report("negative_residuals", params[(params["op"] == "~*~") & (params["est"] < 0)],
           "NEGATIVE RESIDUAL VARIANCES (~*~)", ["lhs", "op", "rhs", "est", "se", "pvalue"])
    report("extreme_loadings", std[(std["op"] == "=~") & (std["est.std"].abs() > 1)],
           "STANDARDIZED LOADINGS > 1", ["lhs", "op", "rhs", "est.std", "pvalue"])
    report("extreme_correlations", std[(std["op"] == "~~") & (std["lhs"] != std["rhs"]) & (std["est.std"].abs() > 1)],
           "CORRELATIONS OUTSIDE [-1,1]", ["lhs", "op", "rhs", "est.std", "pvalue"])
    report("extreme_se", params[params["se"].notna() & (params["se"] > 10)],
           "EXTREME STANDARD ERRORS (> 10)", ["lhs", "op", "rhs", "est", "se", "pvalue"])
    converged = bool(fit.converged)
    if not converged:
        issues["convergence"] = "Model did not converge"
        if verbose:
            print("\n=== CONVERGENCE ISSUE ===\nModel did not converge properly!")
    has_issues = bool(issues)
    if verbose:
        print("\n=== SUMMARY ===")
        if has_issues:
            print("Issues found:")
            labels = {"negative_variances": "negative variance(s) (~~)",
                      "negative_residuals": "negative residual variance(s) (~*~)",
                      "extreme_loadings": "extreme standardized loading(s)",
                      "extreme_correlations": "extreme correlation(s)", "extreme_se": "extreme standard error(s)"}
            for key, label in labels.items():
                if key in issues:
                    print(f" -{len(issues[key])} {label}")
            if "convergence" in issues:
                print(" -Convergence issue")
        else:
            print("No Heywood cases or major issues detected!")
    return {"has_issues": has_issues, "issues": issues, "converged": converged}
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 80, "\ngenerate_matrix_A / generate_matrix_lambda_hat\n", "=" * 80, sep="")
    print(generate_matrix_A(blocks=3, items=3))
    print(generate_matrix_lambda_hat(blocks=3, items=4))

    print("\n" + "=" * 80, "\ncompute_dummy_comparisons\n", "=" * 80, sep="")
    for n in range(1, 7):
        print(n, "->", compute_dummy_comparisons(n))

    print("\n" + "=" * 80, "\ngenerate_unique_comparisons_index / generate_comparisons_matrix\n", "=" * 80, sep="")
    for n in range(1, 5):
        print(f"items={n}:\n", generate_unique_comparisons_index(n))
    print(generate_comparisons_matrix(3))

    print("\n" + "=" * 80, "\nincrease_index\n", "=" * 80, sep="")
    print(increase_index(3, 3))

    print("\n" + "=" * 80, "\nrank_to_binary / rank_df_to_binary\n", "=" * 80, sep="")
    np.random.seed(12345)
    mydata6 = pd.DataFrame({f"i{i + 1}": np.random.normal(2, .5, 10) for i in range(6)})
    print(rank_to_binary(mydata6[["i1", "i2", "i3"]], items=3))
    print(rank_to_binary(mydata6[["i1", "i2", "i3"]], items=3, reverse=False))
    print(rank_df_to_binary(mydata6, 3))

    print("\n" + "=" * 80, "\nname_triplet_pairs\n", "=" * 80, sep="")
    print(name_triplet_pairs(15))
    print(name_triplet_pairs(6, prefix="i", sep="_"))
    print(name_triplet_pairs(range(4, 10)))
    print(name_triplet_pairs(10, strict=False))

    print("\n" + "=" * 80, "\nrank3_to_triplets\n", "=" * 80, sep="")
    result3 = rank_to_binary(mydata6[["i1", "i2", "i3"]], items=3)
    print(rank3_to_triplets(result3))

    print("\n" + "=" * 80, "\nresponse_dimension / cfa_icc_index\n", "=" * 80, sep="")
    print(response_dimension(range(1, 19), 3, [1, 2]))
    print(response_dimension(range(1, 19), 3, [1, 3]))
    print(response_dimension(range(1, 19), 3, [2, 3]))
    print(cfa_icc_index(nitems=18, nfactors=3))

    print("\n" + "=" * 80, "\nicc_cfa / compute_map / compute_icc_thurstonian / compute_ability\n", "=" * 80, sep="")
    gamma = [0.556, -1.253, -1.729, 0.618, 0.937, 0.295, -0.672, -1.127, -0.446, 0.632, 1.147, 0.498]
    psi = [2.172, 1.883, 2.055, 1.869, 2.231, 2.100, 1.762, 1.803, 1.565, 1.892, 1.794, 1.686]
    lambda_ = [1.082, 1.082, -1.297, -1.297, 0.802, 0.802, 1.083, 1.083]
    gamma_12 = [gamma[i - 1] for i in response_dimension(range(1, 13), 3, [1, 2])]
    psi_12 = [psi[i - 1] for i in response_dimension(range(1, 13), 3, [1, 2])]
    eta = np.arange(-6, 6.01, .1)
    result = compute_icc_thurstonian(eta=eta, gamma=gamma_12, lambda_=lambda_, psi=psi_12, plot=True)
    result["plot"].save("plot_icc_thurstonian.png", verbose=False)
    print("saved plot_icc_thurstonian.png")

    map_prior = compute_map(eta=eta, mean=0, sd=1)
    for label, response in [("all-0", [0] * 8), ("all-1", [1] * 8), ("alt-10", [1, 0] * 4), ("alt-01", [0, 1] * 4)]:
        r = compute_ability(response, eta, gamma_12, lambda_, psi_12, map=map_prior, plot=False)
        print(f"{label}: ML={r['ability_ml']}, MAP={r['ability_map']}")

    print("\n" + "=" * 80, "\ncompute_scores\n", "=" * 80, sep="")
    response_df = pd.DataFrame([[0] * 8, [1] * 8, [1, 0] * 4, [0, 1] * 4])
    print(compute_scores(response_df, eta=eta, gamma=gamma_12, lambda_=lambda_, psi=psi_12, map=map_prior, plot=False))

    print("\n" + "=" * 80, "\ncompute_solve\n", "=" * 80, sep="")
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([1., 2.])
    x = compute_solve(A, b)
    print("x =", x, "| A@x =", A @ x, "(should equal b)")
    B = np.column_stack([[1., 2.], [0., 1.]])
    X = compute_solve(A, B)
    print("X =\n", X, "\nA@X =\n", A @ X, "(should equal B)")
    A_inv = compute_solve(A)
    print("A_inv =\n", A_inv, "\nA@A_inv =\n", A @ A_inv, "(should be identity)")

    print("\n" + "=" * 80, "\nscore_tirt_pattern / score_tirt (synthetic, no R needed)\n", "=" * 80, sep="")
    lambda_synth = np.array([[1., 0.], [0., 1.], [1., -1.]])
    theta_diag_synth = np.array([1., 1., 1.])
    tau_synth = np.array([0., 0., 0.])
    Psi_synth = np.eye(2)
    print(score_tirt_pattern([1, 0, 1], lambda_synth, theta_diag_synth, tau_synth, Psi_synth))
    patterns_synth = pd.DataFrame({"a": [1, 0], "b": [0, 1], "c": [1, 0]})
    lambda_df_synth = pd.DataFrame(lambda_synth, index=["a", "b", "c"], columns=["t1", "t2"])
    print(score_tirt(patterns_synth, lambda_df_synth, theta_diag_synth, tau_synth, Psi_synth))

    print("\n" + "=" * 80, "\nfit_tirt / extract_tirt_params / check_heywood\n", "=" * 80, sep="")
    import os
    folder = "data" if os.path.exists("data/tirt_triplets.csv") else "../data"
    triplets = pd.read_csv(f"{folder}/tirt_triplets.csv")
    with open(f"{folder}/tirt_lavaan_model.txt", encoding="utf-8") as handle:
        tirt_model = handle.read()
    fit = fit_tirt(tirt_model, triplets)
    pars = extract_tirt_params(fit)
    print("lambda:\n", pars["lambda"].head())
    print("Psi:\n", pars["Psi"])
    print("score_tirt head:\n", score_tirt(triplets, pars["lambda"], pars["theta_diag"], pars["tau"], pars["Psi"]).head())
    check_heywood(fit, verbose=True)
