def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def is_left(p, q, r):
    p = p.coordinates
    q = q.coordinates
    r = r.coordinates
    return q[0]*r[1]+p[0]*q[1]+p[1]*r[0]-q[0]*p[1]-r[0]*q[1]-r[1]*p[0] >= -1e-10
