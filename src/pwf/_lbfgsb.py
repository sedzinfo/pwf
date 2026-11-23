# -*- coding: utf-8 -*-
"""
R's optim(par, fn, gr, method = "L-BFGS-B"), ported line by line so that an
estimate stops exactly where R's does, also when R's answer is not the
optimum (psych's fa() optimizes some objectives with another objective's
gradient, and then the result is wherever R's line searches leave it).

Sources (R 4.6.2):
- src/appl/lbfgsb.c: L-BFGS-B version 2.3 (Zhu, Byrd, Lu-Chen and Nocedal),
  as translated by f2c and edited by R Core (setulb, mainlb, active, bmv,
  cauchy, cmprlb, errclb, formk, formt, freev, hpsolb, lnsrlb, matupd, projgr,
  subsm, dcsrch, dcstep). scipy ships version 3.0, whose subspace step and
  line search differ, so scipy's L-BFGS-B leaves R's path near the bounds.
- src/appl/optim.c lbfgsb(): the driver (maxit counted at each new iterate,
  the convergence codes 0, 1, 51 and 52, the message, the counts).
- src/appl/dpofa.f (R's version, with its 1e-14 relative tolerance) and
  dtrsl.f: the LINPACK routines lbfgsb.c calls.
- src/library/stats/src/optim.c: fminfn/fmingr, parscale and fnscale, and the
  numerical gradient (ndeps, shortened at the bounds) used when gr is None.
- optim() in R: defaults, recycling of lower/upper.

The BLAS calls of lbfgsb.c (ddot, daxpy, dscal, dcopy) are written as the
reference BLAS does them: ddot is a sequential sum from 0 (the reference
code's unrolled loop adds in that order). R on Windows uses the reference
BLAS; an R linked to OpenBLAS (Linux) sums some dot products in another
order, so its results can differ from this port in the last bits.

The arrays keep the C code's 1-based, column-major indexing: vectors are
lists with an unused element 0, matrices are flat lists indexed as
a[i + j * dim - (1 + dim)], exactly as the f2c code after its pointer
adjustments.
"""
import math

import numpy as np

_DBL_EPSILON = float(np.finfo(float).eps)


def _div(a, b):
    """a / b as C computes it: a zero divisor gives inf or NaN instead of Python's ZeroDivisionError."""
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0.0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _sqrt(a):
    """C's sqrt(): NaN for a negative argument instead of Python's ValueError."""
    return math.sqrt(a) if not a < 0.0 else math.nan


# ---------------------------------------------------------------- reference BLAS and R's LINPACK

def _ddot(n, x, ix, y, iy):
    """ddot(n, x[ix..], 1, y[iy..], 1) of the reference BLAS: sequential, from 0."""
    total = 0.0
    for k in range(n):
        total += x[ix + k] * y[iy + k]
    return total


def _daxpy(n, da, x, ix, y, iy):
    """y[iy..] += da * x[ix..] (reference daxpy, which returns at once when da is 0)."""
    if n <= 0 or da == 0.0:
        return
    for k in range(n):
        y[iy + k] = y[iy + k] + da * x[ix + k]


def _dscal(n, da, x, ix):
    for k in range(n):
        x[ix + k] = da * x[ix + k]


def _dcopy(n, x, ix, y, iy):
    for k in range(n):
        y[iy + k] = x[ix + k]


def _dpofa(a, ia, lda, n):
    """
    R's LINPACK dpofa: Cholesky factor of the upper triangle of a(1:n, 1:n), whose a(1, 1) is a[ia]
    (column-major, leading dimension lda). Returns info (0, or the failing column).
    """
    for j in range(1, n + 1):
        s = 0.0
        cj = ia + (j - 1) * lda
        for k in range(1, j):
            ck = ia + (k - 1) * lda
            t = a[cj + k - 1] - _ddot(k - 1, a, ck, a, cj)
            t = t / a[ck + k - 1]
            a[cj + k - 1] = t
            s = s + t * t
        s = a[cj + j - 1] - s
        if s <= 1e-14 * abs(a[cj + j - 1]):
            return j
        a[cj + j - 1] = math.sqrt(s)
    return 0


def _dtrsl(t, it, ldt, n, b, ib, job):
    """LINPACK dtrsl: solves a triangular system in place; t(1, 1) is t[it], b(1) is b[ib]."""
    for info in range(1, n + 1):
        if t[it + (info - 1) + (info - 1) * ldt] == 0.0:
            return info

    def tt(i, j):
        return it + (i - 1) + (j - 1) * ldt

    def bb(i):
        return ib + i - 1

    kase = 1
    if job % 10 != 0:
        kase = 2
    if (job % 100) // 10 != 0:
        kase += 2
    if kase == 1:  # T * x = b, T lower triangular
        b[bb(1)] = b[bb(1)] / t[tt(1, 1)]
        for j in range(2, n + 1):
            temp = -b[bb(j - 1)]
            _daxpy(n - j + 1, temp, t, tt(j, j - 1), b, bb(j))
            b[bb(j)] = b[bb(j)] / t[tt(j, j)]
    elif kase == 2:  # T * x = b, T upper triangular
        b[bb(n)] = b[bb(n)] / t[tt(n, n)]
        for jj in range(2, n + 1):
            j = n - jj + 1
            temp = -b[bb(j + 1)]
            _daxpy(j, temp, t, tt(1, j + 1), b, bb(1))
            b[bb(j)] = b[bb(j)] / t[tt(j, j)]
    elif kase == 3:  # trans(T) * x = b, T lower triangular
        b[bb(n)] = b[bb(n)] / t[tt(n, n)]
        for jj in range(2, n + 1):
            j = n - jj + 1
            b[bb(j)] = b[bb(j)] - _ddot(jj - 1, t, tt(j + 1, j), b, bb(j + 1))
            b[bb(j)] = b[bb(j)] / t[tt(j, j)]
    else:  # trans(T) * x = b, T upper triangular
        b[bb(1)] = b[bb(1)] / t[tt(1, 1)]
        for j in range(2, n + 1):
            b[bb(j)] = b[bb(j)] - _ddot(j - 1, t, tt(1, j), b, bb(1))
            b[bb(j)] = b[bb(j)] / t[tt(j, j)]
    return 0


# ---------------------------------------------------------------- L-BFGS-B subroutines

def _active(n, l, u, nbd, x, iwhere):
    """Projects x onto the box and initializes iwhere; returns (prjctd, cnstnd, boxed)."""
    prjctd, cnstnd, boxed = False, False, True
    for i in range(1, n + 1):
        if nbd[i] > 0:
            if nbd[i] <= 2 and x[i] <= l[i]:
                if x[i] < l[i]:
                    prjctd = True
                    x[i] = l[i]
            elif nbd[i] >= 2 and x[i] >= u[i]:
                if x[i] > u[i]:
                    prjctd = True
                    x[i] = u[i]
    for i in range(1, n + 1):
        if nbd[i] != 2:
            boxed = False
        if nbd[i] == 0:
            iwhere[i] = -1
        else:
            cnstnd = True
            if nbd[i] == 2 and u[i] - l[i] <= 0.0:
                iwhere[i] = 3
            else:
                iwhere[i] = 0
    return prjctd, cnstnd, boxed


def _bmv(m, sy, wt, col, v, iv, p, ip):
    """Product of the 2m x 2m middle matrix with v; v(i) is v[iv + i], p(i) is p[ip + i]. Returns info."""
    if col == 0:
        return 0
    om = 1 + m
    p[ip + col + 1] = v[iv + col + 1]
    for i in range(2, col + 1):
        i2 = col + i
        total = 0.0
        for k in range(1, i):
            total += sy[i + k * m - om] * v[iv + k] / sy[k + k * m - om]
        p[ip + i2] = v[iv + i2] + total
    info = _dtrsl(wt, 0, m, col, p, ip + col + 1, 11)
    if info != 0:
        return info
    for i in range(1, col + 1):
        p[ip + i] = v[iv + i] / math.sqrt(sy[i + i * m - om])
    info = _dtrsl(wt, 0, m, col, p, ip + col + 1, 1)
    if info != 0:
        return info
    for i in range(1, col + 1):
        p[ip + i] = -p[ip + i] / math.sqrt(sy[i + i * m - om])
    for i in range(1, col + 1):
        total = 0.0
        for k in range(i + 1, col + 1):
            total += sy[k + i * m - om] * p[ip + col + k] / sy[i + i * m - om]
        p[ip + i] += total
    return 0


def _cauchy(n, x, l, u, nbd, g, iorder, iwhere, t, d, xcp, m, wy, ws, sy, wt, theta, col, head,
            wa, ip, ic, iwbp, iv, nint, sbgnrm, epsmch):
    """The generalized Cauchy point. p, c, wbp and v live in wa at the bases ip, ic, iwbp, iv. Returns (nint, info)."""
    on = 1 + n
    if sbgnrm <= 0.0:
        _dcopy(n, x, 1, xcp, 1)
        return nint, 0
    bnded = True
    nfree = n + 1
    nbreak = 0
    ibkmin = 0
    bkmin = 0.0
    col2 = col << 1
    f1 = 0.0
    tl = tu = 0.0
    for i in range(1, col2 + 1):
        wa[ip + i] = 0.0
    for i in range(1, n + 1):
        neggi = -g[i]
        if iwhere[i] != 3 and iwhere[i] != -1:
            if nbd[i] <= 2:
                tl = x[i] - l[i]
            if nbd[i] >= 2:
                tu = u[i] - x[i]
            xlower = nbd[i] <= 2 and tl <= 0.0
            xupper = nbd[i] >= 2 and tu <= 0.0
            iwhere[i] = 0
            if xlower:
                if neggi <= 0.0:
                    iwhere[i] = 1
            elif xupper:
                if neggi >= 0.0:
                    iwhere[i] = 2
            else:
                if abs(neggi) <= 0.0:
                    iwhere[i] = -3
        pointr = head
        if iwhere[i] != 0 and iwhere[i] != -1:
            d[i] = 0.0
        else:
            d[i] = neggi
            f1 -= neggi * neggi
            for j in range(1, col + 1):
                wa[ip + j] += wy[i + pointr * n - on] * neggi
                wa[ip + col + j] += ws[i + pointr * n - on] * neggi
                pointr = pointr % m + 1
            if nbd[i] <= 2 and nbd[i] != 0 and neggi < 0.0:
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tl / (-neggi)
                if nbreak == 1 or t[nbreak] < bkmin:
                    bkmin = t[nbreak]
                    ibkmin = nbreak
            elif nbd[i] >= 2 and neggi > 0.0:
                nbreak += 1
                iorder[nbreak] = i
                t[nbreak] = tu / neggi
                if nbreak == 1 or t[nbreak] < bkmin:
                    bkmin = t[nbreak]
                    ibkmin = nbreak
            else:
                nfree -= 1
                iorder[nfree] = i
                if abs(neggi) > 0.0:
                    bnded = False
    if theta != 1.0:
        _dscal(col, theta, wa, ip + col + 1)
    _dcopy(n, x, 1, xcp, 1)
    if nbreak == 0 and nfree == n + 1:
        return nint, 0
    for j in range(1, col2 + 1):
        wa[ic + j] = 0.0
    f2 = -theta * f1
    f2_org = f2
    if col > 0:
        info = _bmv(m, sy, wt, col, wa, ip, wa, iv)
        if info != 0:
            return nint, info
        f2 -= _ddot(col2, wa, iv + 1, wa, ip + 1)
    dtm = -f1 / f2
    tsum = 0.0
    nint = 1
    skip_to_end = False  # L999: the last breakpoint was the n-th; xcp already holds the bounds
    if nbreak > 0:
        nleft = nbreak
        iteration = 1
        tj = 0.0
        while True:  # L777
            tj0 = tj
            if iteration == 1:
                tj = bkmin
                ibp = iorder[ibkmin]
            else:
                if iteration == 2:
                    if ibkmin != nbreak:
                        t[ibkmin] = t[nbreak]
                        iorder[ibkmin] = iorder[nbreak]
                _hpsolb(nleft, t, iorder, iteration - 2)
                tj = t[nleft]
                ibp = iorder[nleft]
            dt = tj - tj0
            if dtm < dt:
                break  # L888
            tsum += dt
            nleft -= 1
            iteration += 1
            dibp = d[ibp]
            d[ibp] = 0.0
            if dibp > 0.0:
                zibp = u[ibp] - x[ibp]
                xcp[ibp] = u[ibp]
                iwhere[ibp] = 2
            else:
                zibp = l[ibp] - x[ibp]
                xcp[ibp] = l[ibp]
                iwhere[ibp] = 1
            if nleft == 0 and nbreak == n:
                dtm = dt
                skip_to_end = True
                break
            nint += 1
            dibp2 = dibp * dibp
            f1 += dt * f2 + dibp2 - theta * dibp * zibp
            f2 -= theta * dibp2
            if col > 0:
                _daxpy(col2, dt, wa, ip + 1, wa, ic + 1)
                pointr = head
                for j in range(1, col + 1):
                    wa[iwbp + j] = wy[ibp + pointr * n - on]
                    wa[iwbp + col + j] = theta * ws[ibp + pointr * n - on]
                    pointr = pointr % m + 1
                info = _bmv(m, sy, wt, col, wa, iwbp, wa, iv)
                if info != 0:
                    return nint, info
                wmc = _ddot(col2, wa, ic + 1, wa, iv + 1)
                wmp = _ddot(col2, wa, ip + 1, wa, iv + 1)
                wmw = _ddot(col2, wa, iwbp + 1, wa, iv + 1)
                _daxpy(col2, -dibp, wa, iwbp + 1, wa, ip + 1)
                f1 += dibp * wmc
                f2 += (2.0 * dibp * wmp - dibp2 * wmw)
            bound = epsmch * f2_org
            if f2 < bound:
                f2 = bound
            if nleft > 0:
                dtm = -f1 / f2
                continue
            elif bnded:
                f1 = 0.0
                f2 = 0.0
                dtm = 0.0
            else:
                dtm = -f1 / f2
            break
    if not skip_to_end:  # L888
        if dtm <= 0.0:
            dtm = 0.0
        tsum += dtm
        _daxpy(n, tsum, d, 1, xcp, 1)
    if col > 0:  # L999
        _daxpy(col2, dtm, wa, ip + 1, wa, ic + 1)
    return nint, 0


def _cmprlb(n, m, x, g, ws, wy, sy, wt, z, r, wa, indx, theta, col, head, nfree, cnstnd):
    """r = -Z'B(xcp - xk) - Z'g, using W'(xcp - x) from cauchy (in wa after 2m). Returns info."""
    on = 1 + n
    if not cnstnd and col > 0:
        for i in range(1, n + 1):
            r[i] = -g[i]
    else:
        for i in range(1, nfree + 1):
            k = indx[i]
            r[i] = -theta * (z[k] - x[k]) - g[k]
        info = _bmv(m, sy, wt, col, wa, 2 * m, wa, 0)
        if info != 0:
            return -8
        pointr = head
        for j in range(1, col + 1):
            a1 = wa[j]
            a2 = theta * wa[col + j]
            for i in range(1, nfree + 1):
                k = indx[i]
                r[i] += wy[k + pointr * n - on] * a1 + ws[k + pointr * n - on] * a2
            pointr = pointr % m + 1
    return 0


def _errclb(n, m, factr, l, u, nbd, task):
    """Checks the input; returns (task, info, k)."""
    info = k = 0
    if n <= 0:
        task = "ERROR: N .LE. 0"
    if m <= 0:
        task = "ERROR: M .LE. 0"
    if factr < 0.0:
        task = "ERROR: FACTR .LT. 0"
    for i in range(1, n + 1):
        if nbd[i] < 0 or nbd[i] > 3:
            task = "ERROR: INVALID NBD"
            info = -6
            k = i
        if nbd[i] == 2:
            if l[i] > u[i]:
                task = "ERROR: NO FEASIBLE SOLUTION"
                info = -7
                k = i
    return task, info, k


def _formk(n, nsub, ind, nenter, ileave, indx2, iupdat, updatd, wn, wn1, m, ws, wy, sy, theta, col, head):
    """The LEL' factorization of the indefinite matrix K of the subspace step. Returns info."""
    m2 = 2 * m
    o2, on, om = 1 + m2, 1 + n, 1 + m
    if updatd:
        if iupdat > m:  # shift the old part of N
            for jy in range(1, m):
                js = m + jy
                _dcopy(m - jy, wn1, (jy + 1) + (jy + 1) * m2 - o2, wn1, jy + jy * m2 - o2)
                _dcopy(m - jy, wn1, (js + 1) + (js + 1) * m2 - o2, wn1, js + js * m2 - o2)
                _dcopy(m - 1, wn1, (m + 2) + (jy + 1) * m2 - o2, wn1, (m + 1) + jy * m2 - o2)
        pbegin, pend = 1, nsub
        dbegin, dend = nsub + 1, n
        iy = col
        is_ = m + col
        ipntr = head + col - 1
        if ipntr > m:
            ipntr -= m
        jpntr = head
        for jy in range(1, col + 1):
            js = m + jy
            temp1 = temp2 = temp3 = 0.0
            for k in range(pbegin, pend + 1):
                k1 = ind[k]
                temp1 += wy[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
            for k in range(dbegin, dend + 1):
                k1 = ind[k]
                temp2 += ws[k1 + ipntr * n - on] * ws[k1 + jpntr * n - on]
                temp3 += ws[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
            wn1[iy + jy * m2 - o2] = temp1
            wn1[is_ + js * m2 - o2] = temp2
            wn1[is_ + jy * m2 - o2] = temp3
            jpntr = jpntr % m + 1
        jy = col
        jpntr = head + col - 1
        if jpntr > m:
            jpntr -= m
        ipntr = head
        for i in range(1, col + 1):
            is_ = m + i
            temp3 = 0.0
            for k in range(pbegin, pend + 1):
                k1 = ind[k]
                temp3 += ws[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
            ipntr = ipntr % m + 1
            wn1[is_ + jy * m2 - o2] = temp3
        upcl = col - 1
    else:
        upcl = col
    ipntr = head
    for iy in range(1, upcl + 1):
        is_ = m + iy
        jpntr = head
        for jy in range(1, iy + 1):
            js = m + jy
            temp1 = temp2 = temp3 = temp4 = 0.0
            for k in range(1, nenter + 1):
                k1 = indx2[k]
                temp1 += wy[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
                temp2 += ws[k1 + ipntr * n - on] * ws[k1 + jpntr * n - on]
            for k in range(ileave, n + 1):
                k1 = indx2[k]
                temp3 += wy[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
                temp4 += ws[k1 + ipntr * n - on] * ws[k1 + jpntr * n - on]
            wn1[iy + jy * m2 - o2] = wn1[iy + jy * m2 - o2] + temp1 - temp3
            wn1[is_ + js * m2 - o2] = wn1[is_ + js * m2 - o2] - temp2 + temp4
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1
    ipntr = head
    for is_ in range(m + 1, m + upcl + 1):
        jpntr = head
        for jy in range(1, upcl + 1):
            temp1 = temp3 = 0.0
            for k in range(1, nenter + 1):
                k1 = indx2[k]
                temp1 += ws[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
            for k in range(ileave, n + 1):
                k1 = indx2[k]
                temp3 += ws[k1 + ipntr * n - on] * wy[k1 + jpntr * n - on]
            if is_ <= jy + m:
                wn1[is_ + jy * m2 - o2] += temp1 - temp3
            else:
                wn1[is_ + jy * m2 - o2] += -temp1 + temp3
            jpntr = jpntr % m + 1
        ipntr = ipntr % m + 1
    for iy in range(1, col + 1):
        is_ = col + iy
        is1 = m + iy
        for jy in range(1, iy + 1):
            js = col + jy
            js1 = m + jy
            wn[jy + iy * m2 - o2] = wn1[iy + jy * m2 - o2] / theta
            wn[js + is_ * m2 - o2] = wn1[is1 + js1 * m2 - o2] * theta
        for jy in range(1, iy):
            wn[jy + is_ * m2 - o2] = -wn1[is1 + jy * m2 - o2]
        for jy in range(iy, col + 1):
            wn[jy + is_ * m2 - o2] = wn1[is1 + jy * m2 - o2]
        wn[iy + iy * m2 - o2] += sy[iy + iy * m - om]
    if _dpofa(wn, 0, m2, col) != 0:
        return -1
    col2 = col << 1
    for js in range(col + 1, col2 + 1):
        _dtrsl(wn, 0, m2, col, wn, js * m2 + 1 - o2, 11)  # its info is overwritten below, as in R
    for is_ in range(col + 1, col2 + 1):
        for js in range(is_, col2 + 1):
            wn[is_ + js * m2 - o2] += _ddot(col, wn, is_ * m2 + 1 - o2, wn, js * m2 + 1 - o2)
    if _dpofa(wn, col + 1 + (col + 1) * m2 - o2, m2, col) != 0:
        return -2
    return 0


def _formt(m, wt, sy, ss, col, theta):
    """T = theta*SS + L*D^(-1)*L' in the upper triangle of wt, Cholesky factorized. Returns info."""
    om = 1 + m
    for j in range(1, col + 1):
        wt[1 + j * m - om] = theta * ss[1 + j * m - om]
    for i in range(2, col + 1):
        for j in range(i, col + 1):
            k1 = j if i > j else i - 1  # lbfgsb.c's min(i,j) - 1 macro; j >= i here
            ddum = 0.0
            for k in range(1, k1 + 1):
                ddum += sy[i + k * m - om] * sy[j + k * m - om] / sy[k + k * m - om]
            wt[i + j * m - om] = ddum + theta * ss[i + j * m - om]
    if _dpofa(wt, 0, m, col) != 0:
        return -3
    return 0


def _freev(n, nfree, indx, indx2, iwhere, updatd, cnstnd, iteration):
    """Free and active variables at the Cauchy point. Returns (nfree, nenter, ileave, wrk)."""
    nenter = 0
    ileave = n + 1
    if iteration > 0 and cnstnd:
        for i in range(1, nfree + 1):
            k = indx[i]
            if iwhere[k] > 0:
                ileave -= 1
                indx2[ileave] = k
        for i in range(nfree + 1, n + 1):
            k = indx[i]
            if iwhere[k] <= 0:
                nenter += 1
                indx2[nenter] = k
    wrk = ileave < n + 1 or nenter > 0 or updatd
    nfree = 0
    iact = n + 1
    for i in range(1, n + 1):
        if iwhere[i] <= 0:
            nfree += 1
            indx[nfree] = i
        else:
            iact -= 1
            indx[iact] = i
    return nfree, nenter, ileave, wrk


def _hpsolb(n, t, iorder, iheap):
    """Heap sort step of the breakpoints t(1..n) with their indices."""
    if iheap == 0:
        for k in range(2, n + 1):
            ddum = t[k]
            indxin = iorder[k]
            i = k
            while i > 1:
                j = i // 2
                if ddum < t[j]:
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                else:
                    break
            t[i] = ddum
            iorder[i] = indxin
    if n > 1:
        i = 1
        out = t[1]
        indxou = iorder[1]
        ddum = t[n]
        indxin = iorder[n]
        while True:
            j = i + i
            if j <= n - 1:
                if t[j + 1] < t[j]:
                    j += 1
                if t[j] < ddum:
                    t[i] = t[j]
                    iorder[i] = iorder[j]
                    i = j
                    continue
            break
        t[i] = ddum
        iorder[i] = indxin
        t[n] = out
        iorder[n] = indxou


def _matupd(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head, rr, dr, stp, dtd):
    """Updates S, Y, S'Y and S'S. Returns (itail, col, head, theta)."""
    on, om = 1 + n, 1 + m
    if iupdat <= m:
        col = iupdat
        itail = (head + iupdat - 2) % m + 1
    else:
        itail = itail % m + 1
        head = head % m + 1
    _dcopy(n, d, 1, ws, itail * n + 1 - on)
    _dcopy(n, r, 1, wy, itail * n + 1 - on)
    theta = rr / dr
    if iupdat > m:
        for j in range(1, col):
            _dcopy(j, ss, (j + 1) * m + 2 - om, ss, j * m + 1 - om)
            _dcopy(col - j, sy, j + 1 + (j + 1) * m - om, sy, j + j * m - om)
    pointr = head
    for j in range(1, col):
        sy[col + j * m - om] = _ddot(n, d, 1, wy, pointr * n + 1 - on)
        ss[j + col * m - om] = _ddot(n, ws, pointr * n + 1 - on, d, 1)
        pointr = pointr % m + 1
    if stp == 1.0:
        ss[col + col * m - om] = dtd
    else:
        ss[col + col * m - om] = stp * stp * dtd
    sy[col + col * m - om] = dr
    return itail, col, head, theta


def _projgr(n, l, u, nbd, x, g):
    """Infinity norm of the projected gradient."""
    sbgnrm = 0.0
    for i in range(1, n + 1):
        gi = g[i]
        if nbd[i] != 0:
            if gi < 0.0:
                if nbd[i] >= 2:
                    bound = x[i] - u[i]
                    if gi < bound:
                        gi = bound
            else:
                if nbd[i] <= 2:
                    bound = x[i] - l[i]
                    if gi > bound:
                        gi = bound
        size = abs(gi)
        if sbgnrm < size:
            sbgnrm = size
    return sbgnrm


def _subsm(n, m, nsub, ind, l, u, nbd, x, d, ws, wy, theta, col, head, wv, wn, iword):
    """Subspace minimization (version 2.3: one step to the first bound). Returns (info, iword)."""
    on, o2 = 1 + n, 1 + 2 * m
    ns = nsub
    if ns <= 0:
        return 0, iword
    pointr = head
    for i in range(1, col + 1):
        temp1 = temp2 = 0.0
        for j in range(1, ns + 1):
            k = ind[j]
            temp1 += wy[k + pointr * n - on] * d[j]
            temp2 += ws[k + pointr * n - on] * d[j]
        wv[i] = temp1
        wv[col + i] = theta * temp2
        pointr = pointr % m + 1
    m2 = 2 * m
    col2 = col << 1
    info = _dtrsl(wn, 0, m2, col2, wv, 1, 11)
    if info != 0:
        return info, iword
    for i in range(1, col + 1):
        wv[i] = -wv[i]
    info = _dtrsl(wn, 0, m2, col2, wv, 1, 1)
    if info != 0:
        return info, iword
    pointr = head
    for jy in range(1, col + 1):
        js = col + jy
        for i in range(1, ns + 1):
            k = ind[i]
            d[i] += wy[k + pointr * n - on] * wv[jy] / theta + ws[k + pointr * n - on] * wv[js]
        pointr = pointr % m + 1
    for i in range(1, ns + 1):
        d[i] /= theta
    alpha = 1.0
    temp1 = alpha  # not reset for each variable, as in version 2.3
    ibd = 0
    for i in range(1, ns + 1):
        k = ind[i]
        dk = d[i]
        if nbd[k] != 0:
            if dk < 0.0 and nbd[k] <= 2:
                temp2 = l[k] - x[k]
                if temp2 >= 0.0:
                    temp1 = 0.0
                elif dk * alpha < temp2:
                    temp1 = temp2 / dk
            elif dk > 0.0 and nbd[k] >= 2:
                temp2 = u[k] - x[k]
                if temp2 <= 0.0:
                    temp1 = 0.0
                elif dk * alpha > temp2:
                    temp1 = temp2 / dk
            if temp1 < alpha:
                alpha = temp1
                ibd = i
    if alpha < 1.0:
        dk = d[ibd]
        k = ind[ibd]
        if dk > 0.0:
            x[k] = u[k]
            d[ibd] = 0.0
        elif dk < 0.0:
            x[k] = l[k]
            d[ibd] = 0.0
    for i in range(1, ns + 1):
        x[ind[i]] += alpha * d[i]
    iword = 1 if alpha < 1.0 else 0
    return 0, iword


def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """MINPACK-2 dcstep as in lbfgsb.c. Returns (stx, fx, dx, sty, fy, dy, stp, brackt)."""
    sgnd = dp * _div(dx, abs(dx))
    if fp > fx:
        theta = _div((fx - fp) * 3.0, stp - stx) + dx + dp
        d1, d2 = abs(theta), abs(dx)
        d1 = d2 if d1 < d2 else d1
        d2 = abs(dp)
        s = d2 if d1 < d2 else d1
        d1 = _div(theta, s)
        gamm = s * _sqrt(d1 * d1 - _div(dx, s) * _div(dp, s))
        if stp < stx:
            gamm = -gamm
        p = gamm - dx + theta
        q = gamm - dx + gamm + dp
        r = _div(p, q)
        stpc = stx + r * (stp - stx)
        stpq = stx + _div(dx, _div(fx - fp, stp - stx) + dx) / 2.0 * (stp - stx)
        if abs(stpc - stx) < abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = _div((fx - fp) * 3.0, stp - stx) + dx + dp
        d1, d2 = abs(theta), abs(dx)
        d1 = d2 if d1 < d2 else d1
        d2 = abs(dp)
        s = d2 if d1 < d2 else d1
        d1 = _div(theta, s)
        gamm = s * _sqrt(d1 * d1 - _div(dx, s) * _div(dp, s))
        if stp > stx:
            gamm = -gamm
        p = gamm - dp + theta
        q = gamm - dp + gamm + dx
        r = _div(p, q)
        stpc = stp + r * (stx - stp)
        stpq = stp + _div(dp, dp - dx) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = _div((fx - fp) * 3.0, stp - stx) + dx + dp
        d1, d2 = abs(theta), abs(dx)
        d1 = d2 if d1 < d2 else d1
        d2 = abs(dp)
        s = d2 if d1 < d2 else d1
        d1 = _div(theta, s)
        d1 = d1 * d1 - _div(dx, s) * _div(dp, s)
        gamm = 0.0 if d1 < 0 else s * _sqrt(d1)
        if stp > stx:
            gamm = -gamm
        p = gamm - dp + theta
        q = gamm + (dx - dp) + gamm
        r = _div(p, q)
        if r < 0.0 and gamm != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + _div(dp, dp - dx) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            d1 = stp + (sty - stp) * 0.66
            if stp > stx:
                stpf = stpf if d1 > stpf else d1
            else:
                stpf = stpf if d1 < stpf else d1
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = stpf if stpmax > stpf else stpmax
            stpf = stpf if stpmin < stpf else stpmin
    else:
        if brackt:
            theta = _div((fp - fy) * 3.0, sty - stp) + dy + dp
            d1, d2 = abs(theta), abs(dy)
            d1 = d2 if d1 < d2 else d1
            d2 = abs(dp)
            s = d2 if d1 < d2 else d1
            d1 = _div(theta, s)
            gamm = s * _sqrt(d1 * d1 - _div(dy, s) * _div(dp, s))
            if stp > sty:
                gamm = -gamm
            p = gamm - dp + theta
            q = gamm - dp + gamm + dy
            r = _div(p, q)
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin
    if fp > fx:
        sty, fy, dy = stp, fp, dp
    else:
        if sgnd < 0.0:
            sty, fy, dy = stx, fx, dx
        stx, fx, dx = stp, fp, dp
    return stx, fx, dx, sty, fy, dy, stpf, brackt


class _Dcsrch:
    """MINPACK-2 dcsrch as in lbfgsb.c; its C static variables are attributes."""

    def __init__(self):
        self.stage = 0
        self.brackt = False
        self.ginit = self.gtest = self.gx = self.gy = self.finit = self.fx = self.fy = 0.0
        self.stx = self.sty = self.stmin = self.stmax = self.width = self.width1 = 0.0

    def __call__(self, f, g, stp, ftol, gtol, xtol, stpmin, stpmax, task):
        """Returns (stp, task)."""
        if task.startswith("START"):
            if stp < stpmin:
                task = "ERROR: STP .LT. STPMIN"
            if stp > stpmax:
                task = "ERROR: STP .GT. STPMAX"
            if g >= 0.0:
                task = "ERROR: INITIAL G .GE. ZERO"
            if ftol < 0.0:
                task = "ERROR: FTOL .LT. ZERO"
            if gtol < 0.0:
                task = "ERROR: GTOL .LT. ZERO"
            if xtol < 0.0:
                task = "ERROR: XTOL .LT. ZERO"
            if stpmin < 0.0:
                task = "ERROR: STPMIN .LT. ZERO"
            if stpmax < stpmin:
                task = "ERROR: STPMAX .LT. STPMIN"
            if task.startswith("ERROR"):
                return stp, task
            self.brackt = False
            self.stage = 1
            self.finit = f
            self.ginit = g
            self.gtest = ftol * self.ginit
            self.width = stpmax - stpmin
            self.width1 = self.width / 0.5
            self.stx, self.fx, self.gx = 0.0, self.finit, self.ginit
            self.sty, self.fy, self.gy = 0.0, self.finit, self.ginit
            self.stmin = 0.0
            self.stmax = stp + stp * 4.0
            return stp, "FG"
        ftest = self.finit + stp * self.gtest
        if self.stage == 1 and f <= ftest and g >= 0.0:
            self.stage = 2
        if self.brackt and (stp <= self.stmin or stp >= self.stmax):
            task = "WARNING: ROUNDING ERRORS PREVENT PROGRESS"
        if self.brackt and self.stmax - self.stmin <= xtol * self.stmax:
            task = "WARNING: XTOL TEST SATISFIED"
        if stp == stpmax and f <= ftest and g <= self.gtest:
            task = "WARNING: STP = STPMAX"
        if stp == stpmin and (f > ftest or g >= self.gtest):
            task = "WARNING: STP = STPMIN"
        if f <= ftest and abs(g) <= gtol * (-self.ginit):
            task = "CONVERGENCE"
        if task.startswith("WARN") or task.startswith("CONV"):
            return stp, task
        if self.stage == 1 and f <= self.fx and f > ftest:
            fm = f - stp * self.gtest
            fxm = self.fx - self.stx * self.gtest
            fym = self.fy - self.sty * self.gtest
            gm = g - self.gtest
            gxm = self.gx - self.gtest
            gym = self.gy - self.gtest
            self.stx, fxm, gxm, self.sty, fym, gym, stp, self.brackt = _dcstep(
                self.stx, fxm, gxm, self.sty, fym, gym, stp, fm, gm, self.brackt, self.stmin, self.stmax)
            self.fx = fxm + self.stx * self.gtest
            self.fy = fym + self.sty * self.gtest
            self.gx = gxm + self.gtest
            self.gy = gym + self.gtest
        else:
            self.stx, self.fx, self.gx, self.sty, self.fy, self.gy, stp, self.brackt = _dcstep(
                self.stx, self.fx, self.gx, self.sty, self.fy, self.gy, stp, f, g, self.brackt,
                self.stmin, self.stmax)
        if self.brackt:
            if abs(self.sty - self.stx) >= self.width1 * 0.66:
                stp = self.stx + (self.sty - self.stx) * 0.5
            self.width1 = self.width
            self.width = abs(self.sty - self.stx)
        if self.brackt:
            self.stmin = self.sty if self.stx > self.sty else self.stx
            self.stmax = self.sty if self.stx < self.sty else self.stx
        else:
            self.stmin = stp + (stp - self.stx) * 1.1
            self.stmax = stp + (stp - self.stx) * 4.0
        if stp < stpmin:
            stp = stpmin
        if stp > stpmax:
            stp = stpmax
        if (self.brackt and (stp <= self.stmin or stp >= self.stmax)) or \
                (self.brackt and (self.stmax - self.stmin <= xtol * self.stmax)):
            stp = self.stx
        return stp, "FG"


# ---------------------------------------------------------------- mainlb with R's driver folded in

def _lbfgsb(n, m, x, l, u, nbd, evaluate, factr, pgtol, maxit):
    """
    mainlb() run under R's lbfgsb() driver. x, l, u, nbd are 1-based lists (x is changed in place);
    evaluate(x) returns (f, g as a 1-based list). Returns (f, fail, nfgv, task).
    """
    # wa of setulb, zeroed as R's S_alloc does (snd must start at zero)
    ws = [0.0] * (m * n)
    wy = [0.0] * (m * n)
    sy = [0.0] * (m * m)
    ss = [0.0] * (m * m)
    wt = [0.0] * (m * m)
    wn = [0.0] * (4 * m * m)
    snd = [0.0] * (4 * m * m)
    z = [0.0] * (n + 1)
    r = [0.0] * (n + 1)
    d = [0.0] * (n + 1)
    t = [0.0] * (n + 1)
    wa = [0.0] * (8 * m + 1)
    indx = [0] * (n + 1)
    iwhere = [0] * (n + 1)
    indx2 = [0] * (n + 1)
    g = [0.0] * (n + 1)
    f = float("nan")  # the driver's f is uninitialized until the first evaluation
    dcsrch = _Dcsrch()

    epsmch = _DBL_EPSILON
    fold = dnorm = gd = sbgnrm = stp = stpmx = gdold = dtd = 0.0
    col, head, theta, iupdat, updatd = 0, 1, 1.0, 0, False
    iback = itail = ifun = iword = nact = ileave = nenter = 0
    iteration = nfgv = nint = nintol = nskip = 0
    nfree = n
    tol = factr * epsmch
    info = 0
    driver_iter = 0
    fail = 0
    csave = ""

    task, info, _ = _errclb(n, m, factr, l, u, nbd, "START")
    if task.startswith("ERROR"):
        return f, 52, nfgv, task
    prjctd, cnstnd, boxed = _active(n, l, u, nbd, x, iwhere)

    def restart():
        nonlocal info, col, head, theta, iupdat, updatd
        info, col, head, theta, iupdat, updatd = 0, 0, 1, 1.0, 0, False

    task = "FG_START"
    f, g = evaluate(x)
    nfgv = 1
    sbgnrm = _projgr(n, l, u, nbd, x, g)
    if sbgnrm <= pgtol:
        return f, 0, nfgv, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"

    while True:  # L222
        iword = -1
        if not cnstnd and col > 0:
            _dcopy(n, x, 1, z, 1)
            wrk = updatd
            nint = 0
        else:
            nint, info = _cauchy(n, x, l, u, nbd, g, indx2, iwhere, t, d, z, m, wy, ws, sy, wt, theta, col, head,
                                 wa, 0, 2 * m, 4 * m, 6 * m, nint, sbgnrm, epsmch)
            if info != 0:  # singular triangular system: refresh the memory
                restart()
                continue
            nintol += nint
            nfree, nenter, ileave, wrk = _freev(n, nfree, indx, indx2, iwhere, updatd, cnstnd, iteration)
            nact = n - nfree
        # L333
        if not (nfree == 0 or col == 0):
            if wrk:
                info = _formk(n, nfree, indx, nenter, ileave, indx2, iupdat, updatd, wn, snd, m, ws, wy, sy,
                              theta, col, head)
            if info != 0:
                restart()
                continue
            info = _cmprlb(n, m, x, g, ws, wy, sy, wt, z, r, wa, indx, theta, col, head, nfree, cnstnd)
            if info == 0:
                info, iword = _subsm(n, m, nfree, indx, l, u, nbd, z, r, ws, wy, theta, col, head, wa, wn, iword)
            if info != 0:  # L444
                restart()
                continue
        # L555: the search direction, then the line search (lnsrlb)
        for i in range(1, n + 1):
            d[i] = z[i] - x[i]
        dtd = _ddot(n, d, 1, d, 1)
        dnorm = math.sqrt(dtd)
        stpmx = 1e10
        if cnstnd:
            if iteration == 0:
                stpmx = 1.0
            else:
                for i in range(1, n + 1):
                    a1 = d[i]
                    if nbd[i] != 0:
                        if a1 < 0.0 and nbd[i] <= 2:
                            a2 = l[i] - x[i]
                            if a2 >= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx < a2:
                                stpmx = a2 / a1
                        elif a1 > 0.0 and nbd[i] >= 2:
                            a2 = u[i] - x[i]
                            if a2 <= 0.0:
                                stpmx = 0.0
                            elif a1 * stpmx > a2:
                                stpmx = a2 / a1
        if iteration == 0 and not boxed:
            d1 = _div(1.0, dnorm)
            stp = stpmx if d1 > stpmx else d1
        else:
            stp = 1.0
        _dcopy(n, x, 1, t, 1)
        _dcopy(n, g, 1, r, 1)
        fold = f
        ifun = 0
        iback = 0
        csave = "START"
        restarted = False
        while True:  # L666 / L556
            gd = _ddot(n, g, 1, d, 1)
            if ifun == 0:
                gdold = gd
                if gd >= 0.0:  # the direction is not a descent direction
                    info = -4
            if info == 0:
                stp, csave = dcsrch(f, gd, stp, 0.001, 0.9, 0.1, 0.0, stpmx, csave)
                if not csave.startswith("CONV") and not csave.startswith("WARN"):
                    task = "FG_LNSRCH"
                    ifun += 1
                    nfgv += 1
                    iback = ifun - 1
                    if stp == 1.0:
                        _dcopy(n, z, 1, x, 1)
                    else:
                        for i in range(1, n + 1):
                            x[i] = stp * d[i] + t[i]
                else:
                    task = "NEW_X"
            if info != 0 or iback >= 20:  # restore the previous iterate
                _dcopy(n, t, 1, x, 1)
                _dcopy(n, r, 1, g, 1)
                f = fold
                if col == 0:
                    if info == 0:
                        info = -9
                        nfgv -= 1
                        ifun -= 1
                        iback -= 1
                    iteration += 1
                    return f, 52, nfgv, "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH"
                if info == 0:
                    nfgv -= 1
                restart()
                task = "RESTART_FROM_LNSRCH"
                restarted = True
                break
            if task.startswith("FG_LN"):
                f, g = evaluate(x)
                continue
            break
        if restarted:
            continue
        # NEW_X
        iteration += 1
        sbgnrm = _projgr(n, l, u, nbd, x, g)
        driver_iter += 1
        if driver_iter > maxit:
            return f, 1, nfgv, "NEW_X"
        # L777: termination tests
        if sbgnrm <= pgtol:
            return f, 0, nfgv, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
        d1, d2 = abs(fold), abs(f)
        d1 = d2 if d1 < d2 else d1
        ddum = 1.0 if d1 < 1.0 else d1
        if fold - f <= tol * ddum:
            return f, 0, nfgv, "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH"
        for i in range(1, n + 1):
            r[i] = g[i] - r[i]
        rr = _ddot(n, r, 1, r, 1)
        if stp == 1.0:
            dr = gd - gdold
            ddum = -gdold
        else:
            dr = (gd - gdold) * stp
            _dscal(n, stp, d, 1)
            ddum = -gdold * stp
        if dr <= epsmch * ddum:  # skip the L-BFGS update
            nskip += 1
            updatd = False
            continue
        updatd = True
        iupdat += 1
        itail, col, head, theta = _matupd(n, m, ws, wy, sy, ss, d, r, itail, iupdat, col, head, rr, dr, stp, dtd)
        info = _formt(m, wt, sy, ss, col, theta)
        if info != 0:
            restart()
            continue


# ---------------------------------------------------------------- optim(method = "L-BFGS-B")

def _r_rep_len(value, n):
    values = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if n == 0:
        return np.zeros(0)
    if values.size == 0:
        return np.full(n, np.nan)
    return np.resize(values, n)


def r_optim_lbfgsb(fn, par, gr=None, lower=-np.inf, upper=np.inf, fnscale=1.0, parscale=None, ndeps=1e-3,
                   maxit=100, factr=1e7, pgtol=0.0, lmm=5):
    """
    R's optim(par, fn, gr, method = "L-BFGS-B", lower, upper, control = list(fnscale, parscale, ndeps,
    maxit, factr, pgtol, lmm)), with optim's defaults.

    Parameters:
    fn (callable): Objective; takes the parameter vector (numpy array, unscaled as in R) and returns a number.
    par (array-like): Starting values.
    gr (callable, optional): Gradient of fn. Without it, R's central differences with steps ndeps on the
        parscale'd parameters (shortened at the bounds) are used.
    lower, upper (float or array-like): Bounds, recycled to the length of par as rep_len().
    fnscale, parscale, ndeps, maxit, factr, pgtol, lmm: optim's control values. parscale and ndeps may be
        numbers (recycled) or vectors of the length of par.

    Returns:
    dict: par, value, counts (function, gradient), convergence (0; 1 when maxit was reached; 51 warning;
    52 error) and message (R's text, for example "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH", or
    "NEW_X" when maxit was reached), as optim's.

    Examples:
    >>> r_optim_lbfgsb(lambda p: (p[0] - 1) ** 2 + (p[1] + 2) ** 2, [0, 0], lower=[-1, -1])["par"]
    """
    par = np.atleast_1d(np.asarray(par, dtype=float)).ravel().copy()
    npar = par.size
    if parscale is None:
        parscale = np.ones(npar)
    parscale = np.atleast_1d(np.asarray(parscale, dtype=float)).ravel()
    if parscale.size == 1 and npar != 1:
        parscale = np.full(npar, parscale[0])
    if parscale.size != npar:
        raise ValueError("'parscale' is of the wrong length")
    ndeps = np.atleast_1d(np.asarray(ndeps, dtype=float)).ravel()
    if ndeps.size == 1 and npar != 1:
        ndeps = np.full(npar, ndeps[0])
    if gr is None and ndeps.size != npar:
        raise ValueError("'ndeps' is of the wrong length")
    fnscale = float(fnscale)
    lower = _r_rep_len(lower, npar)
    upper = _r_rep_len(upper, npar)
    ps = [float(v) for v in parscale]

    dpar = [0.0] + [_div(float(par[i]), ps[i]) for i in range(npar)]
    low = [0.0] + [_div(float(lower[i]), ps[i]) for i in range(npar)]
    upp = [0.0] + [_div(float(upper[i]), ps[i]) for i in range(npar)]
    nbd = [0] * (npar + 1)
    for i in range(1, npar + 1):
        if not math.isfinite(low[i]):
            nbd[i] = 0 if not math.isfinite(upp[i]) else 3
        else:
            nbd[i] = 1 if not math.isfinite(upp[i]) else 2

    def call_fn(values):
        value = np.asarray(fn(np.array(values, dtype=float)), dtype=float).ravel()
        if value.size != 1:
            raise ValueError(f"objective function in optim evaluates to length {value.size} not 1")
        return float(value[0])

    def fminfn(p):
        for i in range(1, npar + 1):
            if not math.isfinite(p[i]):
                raise ValueError("non-finite value supplied by optim")
        return _div(call_fn([p[i] * ps[i - 1] for i in range(1, npar + 1)]), fnscale)

    def fmingr(p):
        df = [0.0] * (npar + 1)
        if gr is not None:
            for i in range(1, npar + 1):
                if not math.isfinite(p[i]):
                    raise ValueError("non-finite value supplied by optim")
            values = np.asarray(gr(np.array([p[i] * ps[i - 1] for i in range(1, npar + 1)])), dtype=float).ravel()
            if values.size != npar:
                raise ValueError(f"gradient in optim evaluated to length {values.size} not {npar}")
            for i in range(1, npar + 1):
                df[i] = _div(float(values[i - 1]) * ps[i - 1], fnscale)
            return df
        x = [p[i] * ps[i - 1] for i in range(1, npar + 1)]
        for i in range(1, npar + 1):  # usebounds: always so for L-BFGS-B
            epsused = eps = float(ndeps[i - 1])
            tmp = p[i] + eps
            if tmp > upp[i]:
                tmp = upp[i]
                epsused = tmp - p[i]
            x[i - 1] = tmp * ps[i - 1]
            val1 = _div(call_fn(x), fnscale)
            tmp = p[i] - eps
            if tmp < low[i]:
                tmp = low[i]
                eps = p[i] - tmp
            x[i - 1] = tmp * ps[i - 1]
            val2 = _div(call_fn(x), fnscale)
            df[i] = _div(val1 - val2, epsused + eps)
            if not math.isfinite(df[i]):
                raise ValueError(f"non-finite finite-difference value [{i}]")
            x[i - 1] = p[i] * ps[i - 1]
        return df

    def evaluate(p):
        value = fminfn(p)
        if not math.isfinite(value):
            raise ValueError("L-BFGS-B needs finite values of 'fn'")
        return value, fmingr(p)

    if npar == 0:  # not handled in setulb
        value = _div(call_fn([]), fnscale)
        return {"par": par, "value": value * fnscale, "counts": (1, 0), "convergence": 0,
                "message": "NOTHING TO DO"}
    f, fail, nfgv, message = _lbfgsb(npar, int(lmm), dpar, low, upp, nbd, evaluate, float(factr), float(pgtol),
                                     int(maxit))
    result = np.array([dpar[i] * ps[i - 1] for i in range(1, npar + 1)])
    return {"par": result, "value": f * fnscale, "counts": (nfgv, nfgv), "convergence": fail, "message": message}
