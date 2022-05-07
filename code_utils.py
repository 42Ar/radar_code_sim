# -*- coding: utf-8 -*-


def check_code(c, k, h, Delta, N, M):
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
                    
def check_good_modulation_code(g, max_acf):
    for s in range(1, len(g) - 1):
        k = sum(g[k]*g[k+s] for k in range(len(g) - s))
        if (len(g) - s)%2 == 0:
            if k != 0:
                print(f"modulation code error at shift {s}")
        elif abs(k) > max_acf:
            print(f"modulation code error at shift {s}")
