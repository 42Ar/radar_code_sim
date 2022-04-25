# -*- coding: utf-8 -*-


def check_code(c, k, h, Delta, N, M):
    L = len(c)
    cc = c*3
    r = sum((cc[L + j + Delta]*cc[L + j]*
             cc[L + j + Delta + h - k]*cc[L + j + h - k])
            for j in range(L))
    if r != 0:
        print("code error:", k, h, Delta, r/L)
        
        def check_code(k, h, Delta):
            L = len(c)
            cc = c*3
            r = sum((cc[L + j + Delta]*cc[L + j]*
                     cc[L + j + Delta + h - k]*cc[L + j + h - k])
                    for j in range(L))
            if r != 0:
                print("code error:", k, h, Delta, r/L)

def full_code_check(c, N, M):
    for k in range(N):
        for h in range(N):
            if k != h:
                for Delta in range(1, M+1):
                    check_code(c, k, h, Delta, N, M)
                    
def check_good_modulation_code(g):
    for s in range(1, len(g) - 1):
        k = sum(g[k]*g[k+s] for k in range(len(g) - s))
        if abs(k) != (len(g) - s)%2:
            print("modulation code error at shift {s}")