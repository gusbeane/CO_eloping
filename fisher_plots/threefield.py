import numpy as np

def gen_Vk(kmax, Vsurv):
    ans = kmax**3/(6.*np.pi**2)
    return Vsurv*ans

def diagonal_component(B1, B2, B3, N1, N2, N3, Vk):
    B1tot = np.sqrt(B1**2 + N1**2)
    B2tot = np.sqrt(B2**2 + N2**2)
    B3tot = np.sqrt(B3**2 + N3**2)

    ans = B2**2/( (B1*B2)**2 + B1tot**2 * B2tot**2 )
    ans += B3**2/( (B1*B3)**2 + B1tot**2 * B3tot**2 )
    ans += 2*B2*B3/( B1tot**2*B2*B3 + B1**2*B2*B3 )

    return Vk*ans

def offdiagonal_component(B1, B2, B3, N1, N2, N3, Vk):
    B1tot = np.sqrt(B1**2 + N1**2)
    B2tot = np.sqrt(B2**2 + N2**2)
    B3tot = np.sqrt(B3**2 + N3**2)

    ans = B1*B2/( (B1*B2)**2 + B1tot**2*B2tot**2 )
    ans += B1*B3/( B1tot**2*B2*B3 + B1*(B1*B2*B3) )
    ans += B2*B3/( B2tot**2*B3*B1 + B2*(B1*B2*B3) )
    ans += B3*B3/( B3tot**2*B1*B2 + B3*(B1*B2*B3) )

    return Vk*ans

def gen_fisher_matrix(B1, B2, B3, N1, N2, N3, kmax, Vsurv):
    Vk = gen_Vk(kmax, Vsurv)

    F11 = diagonal_component(B1, B2, B3, N1, N2, N3, Vk)
    F22 = diagonal_component(B2, B3, B1, N2, N3, N1, Vk)
    F33 = diagonal_component(B3, B1, B2, N3, N1, N2, Vk)

    F12 = offdiagonal_component(B1, B2, B3, N1, N2, N3, Vk)
    F23 = offdiagonal_component(B2, B3, B1, N2, N3, N1, Vk)
    F31 = offdiagonal_component(B3, B1, B2, N3, N1, N2, Vk)

    F21 = F12
    F32 = F23
    F13 = F31

    mat = Vk * np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

    return mat

