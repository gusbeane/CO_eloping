import numpy as np
from scipy.stats import chi2

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

class chisq(object):
    def __init__(self, B1, B2, B3, cov_mat):
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3

        self.cov_mat = cov_mat

        # self.sigma1 = np.sqrt(cov_mat[0][0])
        # self.sigma2 = np.sqrt(cov_mat[1][1])
        # self.sigma3 = np.sqrt(cov_mat[2][2])

        # self.sigma12 = cov_mat[0][1]
        # self.sigma23 = cov_mat[1][2]
        # self.sigma31 = cov_mat[2][0]

        # self.rho12 = self.sigma12/(self.sigma1*self.sigma2)
        # self.rho23 = self.sigma23/(self.sigma2*self.sigma3)
        # self.rho31 = self.sigma31/(self.sigma3*self.sigma1)

    def __call__(self, b1, b2, b3, chisq):
        d1 = b1 - self.B1
        d2 = b2 - self.B2
        d3 = b3 - self.B3
        d = np.array([d1, d2, d3])

        chisq = np.matmul(np.matmul(np.transpose(d), np.linalg.inv(self.cov_mat)), d)
        
        return chisq

def corner_plot(B1, B2, B3, N1, N2, N3, kmax, Vsurv):
    fisher_mat = gen_fisher(B1, B2, B3, N1, N2, N3, kmax, Vsurv)

    cov_mat = np.inverse(fisher_mat)

    b1list = np.linspace(0.7*B1, 1.3*B1, 100)
    b2list = np.linspace(0.7*B2, 1.3*B2, 100)
    b3list = np.linspace(0.7*B3, 1.3*B3, 100)

    b12_1, b12_2 = np.meshgrid(b1list, b2list)
    b23_2, b23_3 = np.meshgrid(b2list, b3list)
    b31_3, b31_1 = np.meshgrid(b3list, b1list)

    chisq_b12 = chisq()

    p1sigma = 0.68269   # probability inside 1 sigma
    p2sigma = 0.95449   # probability inside 2 sigma
    df = 3

    delta1chisq = chi2.isf(1.-p1sigma, df=df)
    delta2chisq = chi2.isf(1.-p2sigma, df=df)
    contour_list = [delta1chisq, delta2chisq]

    fig = plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 5)
    ax3 = plt.subplot(3, 3, 9)

    ax12 = plt.subplot(3, 3, 4, sharex=ax1)
    ax23 = plt.subplot(3, 3, 8, sharex=ax2)
    ax13 = plt.subplot(3, 3, 7, sharex=ax1, sharey=ax23)

    ax12.contour()
