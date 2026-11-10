# -*- coding: utf-8 -*-
"""
Text helpers (port of rwf ML_NLP.R): cleaning text, removing stopwords,
part-of-speech tagging, word-overlap similarity, the Tversky index, and
word and character statistics with a spelling-error count.

Differences from rwf:
  - tag_pos uses NLTK's averaged perceptron tagger instead of Apache
    OpenNLP's maxent tagger. Both use the Penn Treebank tag set, but the
    two models can tag some words differently. NLTK's data is downloaded
    on first use (averaged_perceptron_tagger_eng, punkt_tab).
  - stat_word_char checks spelling with spylls, a Python implementation of
    hunspell, against the same en_US hunspell dictionary rwf uses (through
    the R spelling package). The dictionary is not bundled: it is looked up
    in R's hunspell package, the usual hunspell folders, or a folder given
    with the dictionary argument or the PWF_HUNSPELL_DIR variable.
"""
##########################################################################################
# LOAD SYSTEM
##########################################################################################
import functools
import gc
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
##########################################################################################
# STOPWORDS
##########################################################################################
# stopwords::stopwords("english") (the Snowball list), in R's order, which matters for removal
STOPWORDS_ENGLISH = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought", "i'm", "you're", "he's",
    "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
    "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's", "what's",
    "here's", "there's", "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "will",
]
# hunspell::en_stats: statistics words the R hunspell package adds to the dictionary by default
# (hunspell(..., ignore = en_stats)), so spelling::spell_check_text accepts them too
EN_STATS = frozenset([
    'Accessor', 'accessor', 'accessors', 'ACF', 'acyclic', 'adaptively', 'Affymetrix',
    'agglomerative', 'AIC', 'Akaike', "Akaike's", 'alikes', 'allelic', 'ANOVA', 'API', 'APIs',
    'approximative', 'ARIMA', 'ARMA', 'arXiv', 'ascii', 'AUC', 'autocorrelation',
    'autocorrelations', 'autocovariance', 'autocovariances', 'autoregression', 'autoregressive',
    'autosomal', 'backend', 'backfitting', 'backquote', 'backquoted', 'backreference', 'backticks',
    'barplot', 'barplots', 'basename', 'bayesian', 'BFGS', 'BibTeX', 'bilinear', 'binomially',
    'Bioconductor', 'biplot', 'biplots', 'bitwise', 'Bivariate', 'bivariate', 'biweight', 'BLAS',
    'bool', 'boolean', 'boxplot', 'boxplots', 'breakpoint', 'capita', 'cardinality', 'cartesian',
    'CCA', 'CDF', 'CDFs', 'centroid', 'centroids', 'changepoint', 'checkbox', 'Cholesky', 'CIs',
    'CJK', 'cloneable', 'clusterings', 'CMYK', 'codepage', 'codings', 'coercible', 'collinearity',
    'colnames', 'columnwise', 'componentwise', 'config', 'confounders', 'cophenetic', 'correlation',
    'covariables', 'covariance', 'covariances', 'Covariate', 'covariate', 'Covariates',
    'covariates', 'Cp', 'CPUs', 'CRAN', 'criterium', 'CRLF', 'CRS', 'CSS', 'CSV', 'cumulated',
    'customizable', 'cutpoint', 'cutpoints', 'datafile', 'Dataframe', 'dataframe', 'dataframes',
    'datapoints', 'Dataset', 'dataset', 'Datasets', 'datasets', 'datetime', 'DBI', 'DCF',
    'decompositions', 'dendrogram', 'dendrograms', 'deparse', 'deparsed', 'deparses', 'deparsing',
    'detrended', 'devel', 'deviances', 'DIC', 'dichotomized', 'differenced', 'differencing',
    'differentiable', 'differentially', 'dimensionality', 'dimnames', 'dir', 'discretization',
    'discretize', 'discretized', 'DLL', 'DLLs', 'DOI', 'durations', 'EDF', 'eigenvector',
    'eigenvectors', 'EISPACK', 'elementwise', 'encodings', 'endian', 'EPS', 'equispaced', 'ESRI',
    'ESS', 'Euclidian', 'eval', 'exceedance', 'executables', 'expectation', 'exponentiated', 'FFT',
    'Filename', 'filename', 'filenames', 'filepath', 'filepaths', 'FIPS', 'forecasted', 'Fortran',
    'FreeBSD', 'frequentist', 'Frobenius', 'func', 'functionalities', 'functionals', 'GAM', 'GARCH',
    'gaussian', 'Gaussians', 'GCV', 'genomic', 'genotype', 'genotyped', 'genotypes', 'genotypic',
    'genotyping', 'Ghostscript', 'gif', 'GIS', 'GLM', 'GLMs', 'glyph', 'glyphs', 'Google', 'GPD',
    'GPL', 'grayscale', 'grob', 'grobs', 'GUIs', 'GWAS', 'haplotype', 'haplotypes', 'heatmap',
    'heatmaps', 'Helvetica', 'heteroscedastic', 'heteroscedasticity', 'hoc', 'homogenous',
    'homoscedastic', 'homozygous', 'HPD', 'HSV', 'html', 'HWE', 'hypergeometric', 'hyperparameter',
    'hyperparameters', 'identifiability', 'IEEE', 'iff', 'iid', "individual's", 'informations',
    'init', 'inline', 'internet', 'interpretable', 'interquartile', 'invariance', 'invertibility',
    'IQR', 'IRT', 'iteratively', 'IWLS', 'Jacobian', 'JASA', 'jitter', 'jittered', 'jittering',
    'JPEG', 'JSON', 'Kolmogorov', 'kriging', 'Kruskal', 'Kurtosis', 'kurtosis', 'LAPACK', 'LaTeX',
    'latin', 'LDA', 'LGPL', 'LHS', 'lhs', 'linearization', 'LINPACK', "LINPACK's", 'loadings',
    'localhost', 'locator', 'logicals', 'logit', 'logits', 'loglikelihood', 'loglinear',
    'lognormal', 'lookup', 'lowess', 'LRT', 'macOS', 'Mahalanobis', 'manhattan', 'MANOVA', 'MATLAB',
    'Matlab', 'matrixes', 'maxima', 'MCD', 'MCMC', 'MDI', 'medoids', 'metacharacters', 'metadata',
    'microarray', 'microarrays', 'microsatellite', 'minima', 'minimax', 'misclassification',
    'missingness', 'missings', 'MLE', 'MLEs', 'monospaced', 'monotonicity', 'MPI', 'MSE', 'multi',
    'multibyte', 'multicore', 'multiline', 'multilocus', 'multimodal', 'Multinomial', 'multinomial',
    'multinomially', 'namespace', 'namespaces', 'NaN', 'NAs', 'NCBI', 'nonlinearity', 'nonnegative',
    'Nonparametric', 'nonparametric', 'normals', 'numerics', 'OECD', 'ok', 'OLS', 'Omegahat',
    "one's", 'online', 'optimality', 'optimizers', 'orthogonalized', 'orthonormal', 'outlier',
    'outliers', 'overdispersed', 'overdispersion', 'overfitting', 'overlayed', 'overplotting',
    'pairlist', 'pairlists', 'parallelize', 'parallelized', 'parameterisation', 'parameterization',
    'parameterizations', 'parameterize', 'parameterized', 'parametrization', 'parametrizations',
    'parametrized', 'pathname', "patient's", 'PCA', 'PCR', 'PCRE', 'PDF', 'pdf', 'periodogram',
    'phenotypes', 'phenotypic', 'Phylogenetic', 'phylogenetic', 'piecewise', 'PLS', 'plugin', 'PNG',
    'pointsize', 'pointwise', 'polytomous', 'POSIX', 'posteriori', 'PostScript', 'precalculated',
    'precomputed', 'prepend', 'prepended', 'preprocessed', 'preprocessing', 'prespecified',
    'priori', 'probit', 'programmatically', 'pseudomedian', 'QQ', 'QTL', 'Quantile', 'quantile',
    'Quantiles', 'quantiles', 'quartile', 'quartiles', 'randomizations', 'rasters', 'rbind',
    'reals', 'recode', 'recoded', 'recoding', 'regressor', 'regressors', 'REML', 'reparameterized',
    'repo', 'reproducibility', 'resample', 'resampled', 'resamples', 'Resampling', 'resampling',
    'rescale', 'rescaled', 'rescales', 'rescaling', 'resize', 'resized', 'reweighted', 'RGB', 'rhs',
    'RMSE', 'RNG', 'ROC', 'rownames', 'RSS', 'runtime', 'SANN', 'SAS', 'scalings', 'Scatterplot',
    'scatterplot', 'scatterplots', 'screeplot', 'SEM', 'semidefinite', 'semiparametric',
    'shapefile', 'sib', 'sigmoid', 'significand', 'skewness', 'Smirnov', 'smoothers', 'SNP', 'SNPs',
    'Solaris', 'Spearman', "Spearman's", 'spinogram', 'spinograms', 'Splus', 'SPSS', 'SQL', 'sqrt',
    'src', 'startup', 'stationarity', 'stdout', 'stepsize', 'stepwise', 'stochastically',
    'studentized', 'subclasses', 'subdirectories', 'subdirectory', 'subexpression',
    'subexpressions', 'subgraph', 'subintervals', 'sublist', 'sublists', 'submatrices', 'submatrix',
    'submodel', 'submodels', 'subpopulations', 'subsample', 'subsampled', 'subsamples',
    'subsampling', 'subsequence', 'subseries', 'subsetted', 'Subsetting', 'subsetting', 'substring',
    'substrings', 'subtree', 'subtrees', 'superclass', 'superclasses', 'superset', 'supremum',
    'SVD', 'SVG', 'SVM', 'Sweave', 'taxa', 'taxon', 'Tcl', 'TCP', "TeX's", 'thresholded',
    'thresholding', 'tickmarks', 'timepoint', 'timepoints', 'timeseries', 'timestamp', 'timestep',
    'Tk', 'TODO', 'tooltip', 'tradeoff', 'Tukey', "Tukey's", 'tuple', 'txt', 'unadjusted', 'unary',
    'undirected', 'unevaluated', 'unexported', 'unimodal', 'uniquenesses', 'univariate', 'unix',
    'unlisting', 'unnormalized', 'unpenalized', 'unscaled', 'unstandardized', 'untransformed',
    'URI', 'url', 'URW', 'USD', 'username', 'UTC', 'UTF', 'utils', 'UTM', "variable's", 'variance',
    'variates', 'variogram', 'vectorised', 'vectorized', 'vertices', 'viewport', "viewport's",
    'viewports', 'viridis', 'voxel', 'warmup', 'webpage', 'Weibull', 'WGS', 'whitespace',
    'Wilcoxon', 'wildcard', 'wildcards', 'wireframe', 'Wishart', 'wrt', "x's", 'xml', 'YAML',
])
##########################################################################################
# CLEAR TEXT
##########################################################################################
_DIGITS = re.compile(r"[0-9]+")
_PUNCTUATION = re.compile(r"(?:[^\w\s]|_)+")  # POSIX [[:punct:]]: graphic, not alphanumeric
_SPACES = re.compile(r"\s+")


def _as_list(text):
    return [text] if isinstance(text, str) else list(text)


def clear_text(text):
    """
    Lower-case text with digits and punctuation replaced by spaces and runs
    of white space collapsed (port of rwf::clear_text).

    Parameters:
    text (str or list of str): Text.

    Returns:
    list of str: One cleaned string per input string.

    Examples:
    >>> clear_text(["word_one word_two", "It uses a dictionary of over 200 Latin words."])
    ['word one word two', 'it uses a dictionary of over latin words']
    """
    result = []
    for value in _as_list(text):
        value = _DIGITS.sub(" ", value)
        value = _PUNCTUATION.sub(" ", value)
        value = _SPACES.sub(" ", value)
        result.append(value.strip().lower())
    return result
##########################################################################################
# REMOVE STOPWORDS
##########################################################################################
def clear_stopwords(text, stopwords=None):
    """
    Remove stopwords and one-letter words, then clean the text with
    clear_text (port of rwf::clear_stopwords). Matching is case sensitive,
    as in rwf, so capitalised stopwords stay.

    Parameters:
    text (str or list of str): Text.
    stopwords (list of str, optional): Words to remove. Defaults to the
        English Snowball list, as stopwords::stopwords("english").

    Returns:
    list of str: One cleaned string per input string.

    Examples:
    >>> clear_stopwords(["All the Lorem Ipsum generators on the Internet tend to repeat"])
    ['all lorem ipsum generators internet tend repeat']
    """
    stopwords = STOPWORDS_ENGLISH if stopwords is None else list(stopwords)
    pattern = re.compile("|".join(rf"\b{re.escape(word)}\b" for word in stopwords), flags=re.ASCII)
    single_letter = re.compile(r" *\b[^\W\d_]\b *")
    removed = [single_letter.sub(" ", pattern.sub("", value)) for value in _as_list(text)]
    return clear_text(removed)
##########################################################################################
# PART OF SPEECH TAGGING
##########################################################################################
def _nltk():
    import nltk
    for resource, path in [("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                           ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    return nltk


def tag_pos(text):
    """
    Part-of-speech tags (Penn Treebank) of every word of the text, taken as
    one passage (port of rwf::tag_pos, which uses Apache OpenNLP; this uses
    NLTK's tagger, so some tags can differ).

    Parameters:
    text (str or list of str): Text; several strings are joined with new lines.

    Returns:
    dict: POStagged (one string of "word/TAG" pairs) and POStags (list of tags).

    Examples:
    >>> tag_pos("The generated Lorem Ipsum is therefore always free from repetition.")
    """
    nltk = _nltk()
    tagged = nltk.pos_tag(nltk.word_tokenize("\n".join(_as_list(text))))
    return {"POStagged": " ".join(f"{word}/{tag}" for word, tag in tagged), "POStags": [tag for _, tag in tagged]}
##########################################################################################
# TEXT SIMILARITY
##########################################################################################
def _unique(values):
    """R's unique(): first occurrences, in order."""
    return list(dict.fromkeys(values))


def compute_tversky_index(x, y, alpha=0.5, beta=0.5):
    """
    Tversky index of two sets, |x and y| / (|x and y| + alpha |x - y| + beta |y - x|)
    (port of rwf::compute_tversky_index). alpha = beta = 0.5 gives the
    Sorensen-Dice coefficient, alpha = beta = 1 the Jaccard index. Values
    are compared as text and duplicates are ignored.

    Parameters:
    x, y (list): The two sets.
    alpha (float, optional): Weight of the elements only in x. Defaults to 0.5.
    beta (float, optional): Weight of the elements only in y. Defaults to 0.5.

    Returns:
    float: Between 0 (no overlap) and 1 (the same sets).

    Examples:
    >>> compute_tversky_index(["a", "b", "c", "d"], ["b", "c", "d", "e"])
    0.75
    """
    x, y = [str(v) for v in _as_list(x)], [str(v) for v in _as_list(y)]
    x_set, y_set = set(x), set(y)
    intersection = len([v for v in _unique(x) if v in y_set])
    x_only = len([v for v in _unique(x) if v not in y_set])
    y_only = len([v for v in _unique(y) if v not in x_set])
    return intersection / (intersection + alpha * x_only + beta * y_only)


def compute_text_similarity(text1, text2):
    """
    Word-overlap measures of two texts given as lists of words (port of
    rwf::compute_text_similarity).

    Parameters:
    text1, text2 (list of str): Words of each text.

    Returns:
    pandas.DataFrame: One row with tversky (Sorensen-Dice of the word sets),
    intersect (number of distinct shared words), intersect_weight (sum over
    shared words of count in text1 x count in text2), setdiff1 and setdiff2
    (distinct words only in text1 / text2), lengtht1 and lengtht2 (numbers
    of words).

    Examples:
    >>> compute_text_similarity("word_one word_two word_three".split(), "word_three word_four word_six".split())
    """
    text1, text2 = _as_list(text1), _as_list(text2)
    set1, set2 = set(text1), set(text2)
    shared = [w for w in _unique(text1) if w in set2]
    counts1, counts2 = pd.Series(text1).value_counts(), pd.Series(text2).value_counts()
    return pd.DataFrame([{
        "tversky": compute_tversky_index(text1, text2),
        "intersect": len(shared),
        "intersect_weight": int(sum(counts1[w] * counts2[w] for w in shared)),
        "setdiff1": len([w for w in _unique(text1) if w not in set2]),
        "setdiff2": len([w for w in _unique(text2) if w not in set1]),
        "lengtht1": len(text1),
        "lengtht2": len(text2),
    }])
##########################################################################################
# STATISTICS FOR CHARACTERS AND WORDS
##########################################################################################
def _dictionary_folders():
    folders = [os.environ.get("PWF_HUNSPELL_DIR", "")]
    for library in [os.environ.get("R_LIBS_USER", ""), os.path.expanduser("~/AppData/Local/R/win-library/*"),
                    os.path.expanduser("~/R/*/*"), "C:/Program Files/R/*/library", "/usr/lib/R/site-library",
                    "/usr/local/lib/R/site-library", "/Library/Frameworks/R.framework/Resources/library"]:
        folders += glob.glob(os.path.join(library, "hunspell", "dict"))
    folders += ["/usr/share/hunspell", "/usr/share/myspell", "/usr/share/myspell/dicts", "/Library/Spelling"]
    return [f for f in folders if f]


@functools.lru_cache(maxsize=4)
def _hunspell_dictionary(folder=None):
    from spylls.hunspell import Dictionary
    candidates = [folder] if folder else _dictionary_folders()
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "en_US.dic")) and os.path.exists(os.path.join(candidate, "en_US.aff")):
            # spylls leaves the .aff file open; collect it here, where its ResourceWarning is silenced
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                dictionary = Dictionary.from_files(os.path.join(candidate, "en_US"))
                gc.collect()
            return dictionary
    raise FileNotFoundError("no hunspell en_US dictionary (en_US.dic and en_US.aff) found; pass its folder as "
                            "dictionary= or set PWF_HUNSPELL_DIR (R users have it in the hunspell package)")


def stat_word_char(text, dictionary=None):
    """
    Number of words, mean, sd, largest and smallest word length, and number
    of distinct misspelled words of each text, after clear_text (port of
    rwf::stat_word_char). Spelling is checked against the hunspell en_US
    dictionary, as in rwf.

    Parameters:
    text (str or list of str): Texts.
    dictionary (str, optional): Folder with en_US.dic and en_US.aff. Python
        only; by default R's hunspell package or the usual hunspell folders
        are searched.

    Returns:
    pandas.DataFrame: One row per text with words, mean_char, sd_char,
    max_char, min_char, spell_error. As in rwf, the statistics words of R's
    hunspell::en_stats (EN_STATS) count as spelled correctly.

    Examples:
    >>> stat_word_char("There are many variations of passages of Lorem Ipsum available")
    """
    checker = _hunspell_dictionary(dictionary)
    rows = []
    for value in clear_text(text):
        words = value.split(" ") if value else []
        lengths = np.array([len(w) for w in words if len(w) > 0], dtype=float)
        misspelled = {w for w in words if w and w not in EN_STATS and not checker.lookup(w)}
        rows.append({
            "words": len(words),
            "mean_char": float(lengths.mean()) if len(lengths) else np.nan,
            "sd_char": float(lengths.std(ddof=1)) if len(lengths) > 1 else np.nan,
            "max_char": float(lengths.max()) if len(lengths) else -np.inf,
            "min_char": float(lengths.min()) if len(lengths) else np.inf,
            "spell_error": len(misspelled),
        })
    return pd.DataFrame(rows)
##########################################################################################
# EXAMPLES
##########################################################################################
if __name__ == "__main__":
    texts = [
        "word_one word_two word_three",
        "word_three word_four word_six",
        "All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, "
        "making this the first true generator on the Internet.",
        "It uses a dictionary of over 200 Latin words, combined with a handful of model sentence "
        "structures, to generate Lorem Ipsum which looks reasonable.",
        "The generated Lorem Ipsum is therefore always free from repetition, injected humour, or "
        "non-characteristic words etc.",
    ]
    print("=" * 80, "\nclear_text\n", "=" * 80, sep="")
    print(clear_text(texts))
    print("\n" + "=" * 80, "\nclear_stopwords\n", "=" * 80, sep="")
    print(clear_stopwords(texts))
    print("\n" + "=" * 80, "\ntag_pos\n", "=" * 80, sep="")
    print(tag_pos(texts)["POStagged"])
    print("\n" + "=" * 80, "\ncompute_text_similarity\n", "=" * 80, sep="")
    words = [t.split(" ") for t in texts]
    print(compute_text_similarity(words[0], words[1]))
    print("\n" + "=" * 80, "\ncompute_tversky_index\n", "=" * 80, sep="")
    print(compute_tversky_index(["a", "b", "c", "d"], ["b", "c", "d", "e"], alpha=1, beta=1))
    print("\n" + "=" * 80, "\nstat_word_char\n", "=" * 80, sep="")
    print(stat_word_char(texts))
