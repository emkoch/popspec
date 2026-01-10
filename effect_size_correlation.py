import numpy as np
import scipy as sp
import scipy.special as spy
import scipy.integrate as spi
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib

### SFS FUNCTIONS ###
#####################

def sfs_ud_params_sigma(xx, theta, SS):
    """Calculate the intensity of the site frequency spectrum under underdominant selection.
    Args:
        xx (float): The frequency of the derived allele.
        theta (float): The population-scaled mutation rate.
        SS (float): The selection coefficient.
    """
    SS = np.abs(SS) + 1e-8
    return (theta*np.exp(-SS*xx*(1-xx))/(xx*(1-xx))*
        (1 + spy.erf(np.sqrt(SS)*(0.5-xx))/
         spy.erf(np.sqrt(SS)/2)))


def sfs_dir_params(xx, theta, SS):
    """Calculate the intensity of the site frequency spectrum under directional selection.
    Args:
        xx (float): The frequency of the derived allele.
        theta (float): The population-scaled mutation rate.
        SS (float): The scaled selection coefficient.
    """
    SS = SS + 1e-8
    return (2*theta*(np.exp(-SS)-np.exp(-SS*xx)) / (xx*(1-xx)*(np.exp(-SS)-1)))

def sfs_dir_low(xx, theta, SS):
    """
    Calculate the site frequency spectrum for a strong selection approximation.
    
    Parameters:
        xx (float): The frequency of the derived allele.
        theta (float): The scaled mutation rate.
        SS (float): The scaled selection coefficient.
    """
    return 2 * theta * np.exp(-SS*xx) / xx

def sfs_drift_barrier(xx, theta, SS):
    """
    Calculate the site frequency spectrum for a drift barrier approximation.
    
    Parameters:
        xx (float): The frequency of the derived allele.
        theta (float): The scaled mutation rate.
        SS (float): The scaled selection coefficient.
    """
    SS = np.where(SS < 1, 1, SS)
    barrier = 1/(SS)
    return np.where(xx<=barrier, 2*theta/xx, 0)


"""
The following functions calculate the expected burden for each model 
as well as the probability of a variant being in a given MAF range.
"""

def poly_prob_maf_range(y1, y2, SS):
    return spi.quad(sfs_ud_params_sigma, y1, y2, args=(1, SS))[0] + spi.quad(sfs_ud_params_sigma, 1-y2, 1-y1, args=(1, sigma))[0]

def poly_prob(yy, SS):
    return spi.quad(sfs_ud_params_sigma, yy, 1-yy, args=(1, SS))[0]

def poly_prob_dir(yy, S):
    return spi.quad(sfs_dir_params, yy, 1-yy, args=(1, S))[0]

def poly_prob_dir_low(yy, S):
    return spi.quad(sfs_dir_low, yy, 1-yy, args=(1, S))[0]

def poly_prob_barrier(yy, S):
    S = np.where((S)<1, 1/2, S)
    upper_lim = np.minimum(1-yy, 1/(S))
    result =  np.log(upper_lim) - np.log(yy)
    return 2*np.where(result<0, 0, result)

ud_burden = lambda xx, theta, SS: xx*sfs_ud_params_sigma(xx, theta, SS)
dir_burden = lambda xx, theta, S: xx*sfs_dir_params(xx, theta, S)
dir_low_burden = lambda xx, theta, S: xx*sfs_dir_low(xx, theta, S)
barrier_burden = lambda xx, theta, S: xx*sfs_drift_barrier(xx, theta, S)

def ex_burden(sigma):
    return spi.quad(ud_burden, 0, 1, args=(1, sigma))[0]

def ex_burden_dir(S):
    return spi.quad(dir_burden, 0, 1, args=(1, S))[0]

def ex_burden_dir_low(S):
    return (1-np.exp(-S))/S * 2

def ex_burden_barrier(S):
    S = S + 1e-8
    return 2*np.where((S)<1, 1, 1/(S))

### CORRELATION FUNCTIONS ###
#############################

def stab_corr(yy, SS):
    """Calculate the correlation between beta_i and beta_j for a given MAF and 
       selection coefficient under a single effect size model with symmetric mutation.
       
    Parameters:
        yy (float): The MAF threshold.
        SS (float): The scaledselection coefficient.
        """
    burd = ex_burden(SS)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)
    return burd * (double_prob - neut_prob) / (burd * (double_prob + neut_prob) + single_prob**2)

def stab_corr_posLD(yy, SS):
    """Calculate the correlation between beta_i and beta_j for a given MAF and
         selection coefficient under a single effect size model with symmetric 
         mutation and positive LD.
         
    Parameters:
        yy (float): The MAF threshold.
        SS (float): The scaled selection coefficient.
    """
    burd = ex_burden(SS)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    return burd * (double_prob - neut_prob) / (burd * (double_prob + neut_prob)) if burd * (double_prob + neut_prob) > 0 else 1

def stab_corr_posLD_poly(yy, SS):
    def ex_diff(xx):
        return xx * (poly_prob(yy, (2*np.sqrt(SS)-xx*np.sqrt(SS))**2 ) - poly_prob(yy, xx**2*SS)) * sfs_ud_params_sigma(xx, 1, SS)
    def ex_sum(xx):
        return xx * (poly_prob(yy, (2*np.sqrt(SS)-xx*np.sqrt(SS))**2 ) + poly_prob(yy, xx**2*SS)) * sfs_ud_params_sigma(xx, 1, SS)
    return spi.quad(ex_diff, 0, 1)[0] / spi.quad(ex_sum, 0, 1)[0]
    
def stab_corr_posLD_neut(yy, SS, zero_prob):
    """
    Calculate the correlation between beta_i and beta_j for a given MAF and
    selection coefficient under a single effect size model with symmetric
    mutation and positive LD, and a given probability of zero variants.
    
    Parameters:
        yy (float): The MAF threshold.
        SS (float): The scaled selection coefficient.
        zero_prob (float): The probability of zero variants.
    """
    burd = ex_burden(SS)
    neut_burd = ex_burden(1e-8)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)
    result = (1-zero_prob) * burd * (double_prob - neut_prob) 
    result /= zero_prob * (neut_burd + burd) * single_prob + (1-zero_prob) * burd * (double_prob + neut_prob)
    return result


def stab_corr_posLD_poly(yy, SS):
    def ex_diff(xx):
        return xx * (poly_prob(yy, (2*np.sqrt(SS)-xx*np.sqrt(SS))**2 ) - poly_prob(yy, xx**2*SS)) * sfs_ud_params_sigma(xx, 1, SS)
    def ex_sum(xx):
        return xx * (poly_prob(yy, (2*np.sqrt(SS)-xx*np.sqrt(SS))**2 ) + poly_prob(yy, xx**2*SS)) * sfs_ud_params_sigma(xx, 1, SS)
    return quad(ex_diff, 0, 1)[0] / quad(ex_sum, 0, 1)[0]
  

def stab_corr_neut(yy, SS, zero_prob):
    """
    Calculate the correlation between beta_i and beta_j for a given MAF and
    selection coefficient under a single effect size model with symmetric
    mutation and a given probability of zero variants.

    Parameters:
        yy (float): The MAF threshold.
        SS (float): The scaled selection coefficient.
        zero_prob (float): The probability of zero variants.
    """
    burd = ex_burden(SS)
    neut_burd = ex_burden(1e-8)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)
    result = (1-zero_prob) * burd * (double_prob - neut_prob) 
    result /= (zero_prob * (single_prob*neut_prob + (neut_burd + burd) * single_prob) +
                (1-zero_prob) * (single_prob**2 + burd * (double_prob + neut_prob)))
    return result

def stab_corr_num(yy, SS):
    """Just compute the numerator for the symmetric single effect size model"""
    burd = ex_burden(SS)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    return (double_prob - neut_prob) * burd

def asym_sel_corr(yy, SS, qq, kk=1):
    """
    Calculate the correlation between beta_i and beta_j for a given MAF and
    selection coefficient under a single effect size model with asymmetric
    selection. The model assumes that the selection coefficient for the smaller
    alleles is qq*SS and double mutants combine like kk*(SS_i+SS_j).

    Parameters:
        yy (float): The MAF threshold.
        SS (float): The scaled selection coefficient.
        qq (float): The multiplier for the smaller alleles.
        kk (float): The multiplier for the double mutants.
    """
    burd_plus = ex_burden_dir(SS)
    burd_minus = ex_burden_dir(qq*SS)
    double_plus = poly_prob_dir(yy, 2*kk*SS)
    double_minus = poly_prob_dir(yy, 2*kk*qq*SS)
    double_mix = poly_prob_dir(yy, kk*(SS+qq*SS))

    P_2 = 2 * burd_plus * double_plus
    P_1 = (burd_plus + burd_minus) * double_mix
    P_0 = 2 * burd_minus * double_minus
    try:
        result = (P_0*P_2 - P_1**2) / ((P_0 + P_1) * (P_1 + P_2))
    except ZeroDivisionError:
        result = np.nan
    # If the sum of the P are < 1e-100 then the result is likely to be numerically unstable
    if np.sum([P_0, P_1, P_2]) < 1e-100:
        result = np.nan
    return result

"""
Functions for a toy model of mutational bias using a balances between
a single non-zero effect size and a zero effect sizes.
"""

def bias_test(yy, SS, zero_prob, mult=2):
    """Compte the correlation between beta_i and beta_j for a given MAF and
       selection coefficient under a single effect size model with single positive effect size
       and a given probability of zero variants.
       
       Parameters:
           yy (float): The MAF threshold.
           SS (float): The scaled selection coefficient.
           zero_prob (float): The probability of zero variants.
           mult (float): The multiplier for the selection coefficient when considering double haplotypes.
    """
    P0 = poly_prob(yy, 1e-8)
    PS = poly_prob_dir(yy, SS)
    P2S = poly_prob_dir(yy, mult*SS)
    x0 = ex_burden_dir(1e-8)
    xS = ex_burden_dir(SS)
    p_lambda = zero_prob**2 * (P0**2 + 2*x0*P0) + 2*zero_prob*(1-zero_prob)*(P0*PS + (x0+xS)*PS) + (1-zero_prob)**2 * (PS**2 + 2*xS*P2S)
    p_0 = P0**2 + 2*x0*P0
    p_1 = P0*PS + (x0+xS)*PS
    p_2 = PS**2 + 2*xS*P2S
    num = p_lambda * p_2 * (1-zero_prob)**2 - (zero_prob*(1-zero_prob)*p_1 + (1-zero_prob)**2 * p_2)**2
    denom = p_lambda * (zero_prob*(1-zero_prob)*p_1 + (1-zero_prob)**2 * p_2) - (zero_prob*(1-zero_prob)*p_1 + (1-zero_prob)**2 * p_2)**2
    return num/denom

def bias_test_posLD(yy, SS, zero_prob, mult=2):
    """Same as bias_test but for the positive LD case"""
    P0 = poly_prob(yy, 1e-8)
    PS = poly_prob_dir(yy, SS)
    P2S = poly_prob_dir(yy, mult*SS)
    x0 = ex_burden_dir(1e-8)
    xS = ex_burden_dir(SS)
    p_0 = 2*x0*P0
    p_1 = (x0+xS)*PS
    p_2 = 2*xS*P2S
    return zero_prob*(1-zero_prob)*(p_0*p_2-p_1**2) / ((zero_prob*p_0 + (1-zero_prob)*p_1)*(zero_prob*p_1 + (1-zero_prob)*p_2))

"""
Functions for calculating correaltions using distributions of effect sizes
"""

def make_A_B_matrix(SS_set_new, burden_set, poly_probs_set, SS_set):
    """
    Calculates the A and B matrices based on the given input parameters.

    Parameters:
    SS_set_new (array-like): New set of SS values.
    burden_set (array-like): Set of burden values.
    poly_probs_set (array-like): Set of polymorphism probability values.
    SS_set (array-like): Set of SS values.

    Returns:
    tuple: A tuple containing the A and B matrices.
    """

    SS_grid_1, SS_grid_2 = np.meshgrid(SS_set_new, SS_set_new)
    burden_grid_1 = np.interp(SS_grid_1, SS_set, burden_set)
    burden_grid_2 = np.interp(SS_grid_2, SS_set, burden_set)
    poly_prob_grid_plus  = np.interp(SS_grid_1 + SS_grid_2 + 2*np.sqrt(SS_grid_1*SS_grid_2), SS_set, poly_probs_set)
    poly_prob_grid_minus = np.interp(SS_grid_1 + SS_grid_2 - 2*np.sqrt(SS_grid_1*SS_grid_2), SS_set, poly_probs_set)

    return (0.5 * (burden_grid_1 + burden_grid_2) * (poly_prob_grid_plus - poly_prob_grid_minus),
            0.5 * (burden_grid_1 + burden_grid_2) * (poly_prob_grid_plus + poly_prob_grid_minus))


def corr_DFE(SS_set_new, poly_probs_set, SS_set, AA, BB, dfe_f, dfe_F, posLD=False):
    """
    Calculate the correlation between two traits using the Distribution of Fitness Effects (DFE) model.

    Parameters:
    - SS_set_new (ndarray): Array of new SS values.
    - poly_probs_set (ndarray): Array of polygenic probabilities.
    - SS_set (ndarray): Array of SS values.
    - AA (ndarray): Array of AA values.
    - BB (ndarray): Array of BB values.
    - dfe_f (function): Function that calculates the DFE for a given SS value.
    - dfe_F (function): Function that calculates the cumulative DFE for a given SS value.

    Returns:
    - float: The correlation between the two traits.
    """
    
    lower_prob = dfe_F(SS_set_new[0])
    upper_prob = 1 - dfe_F(SS_set_new[-1])
    SS_grid_1, SS_grid_2 = np.meshgrid(SS_set_new, SS_set_new)

    # Calculate the numerator
    num_mat = np.sqrt(SS_grid_1 * SS_grid_2) * AA * dfe_f(SS_grid_1) * dfe_f(SS_grid_2)
    lower_vals = (np.sqrt(SS_grid_1[:,0] * SS_grid_2[:,0]) * AA[:,0])
    upper_vals = (np.sqrt(SS_grid_1[:,-1] * SS_grid_2[:,-1]) * AA[:,-1])
    # Integrate over SS_2
    num_mat_vec = np.trapz(y=num_mat, x=SS_set_new, axis=-1)
    num_mat_vec += lower_vals * lower_prob
    num_mat_vec += upper_vals * upper_prob
    # Integrate over SS_1
    num = np.trapz(y=num_mat_vec, x=SS_set_new)
    num += num_mat_vec[0] * lower_prob
    num += num_mat_vec[-1] * upper_prob

    # Calculate the denominator
    P_bar = np.trapz(y=poly_probs_set * dfe_f(SS_set), x=SS_set)
    P_bar += poly_probs_set[0] * lower_prob
    P_bar += poly_probs_set[-1] * upper_prob
    S_bar = np.trapz(y=poly_probs_set * SS_set * dfe_f(SS_set), x=SS_set)
    S_bar += poly_probs_set[0] * SS_set[0] * lower_prob
    S_bar += poly_probs_set[-1] * SS_set[-1] * upper_prob

    denom_mat = SS_grid_1 * BB * dfe_f(SS_grid_1) * dfe_f(SS_grid_2)
    lower_vals = SS_grid_1[:,0] * BB[:,0]
    upper_vals = SS_grid_1[:,-1] * BB[:,-1]
    # Integrate over SS_2
    denom_mat_vec = np.trapz(y=denom_mat, x=SS_set_new, axis=-1)
    denom_mat_vec += lower_vals * lower_prob
    denom_mat_vec += upper_vals * upper_prob
    # Integrate over SS_1
    denom = np.trapz(y=denom_mat_vec, x=SS_set_new)
    denom += denom_mat_vec[0] * lower_prob
    denom += denom_mat_vec[-1] * upper_prob

    if posLD:
        return num / (P_bar * S_bar + denom), num / denom
    return num / (P_bar * S_bar + denom)


def corr_DFE_MC_interp(dfe_samp, burden_set, poly_prob_set, SS_interp_set, n=1000000):
    """
    Calculate the correlation between effect sizes using Monte Carlo simulation.

    Parameters:
    - dfe_samp (function): A function that samples from the DFE.
    - burden_set (array-like): Array-like object containing the burden values.
    - poly_prob_set (array-like): Array-like object containing the polymorphism probability values.
    - SS_interp_set (array-like): Array-like object containing the SS interpolation values.
    - n (int): Number of samples to generate (default: 10000).

    Returns:
    - tuple: A tuple containing two correlation values:
        - The correlation between effect sizes (numerator / denominator).
        - The correlation between effect sizes (numerator / denominator) under D>0.
    """
    
    # sample from the dfe gamma distribution for the numerator
    SS_samp_i = dfe_samp(n)
    SS_samp_j = dfe_samp(n)
    ex_burdens_i = np.interp(SS_samp_i, SS_interp_set, burden_set)
    ex_burdens_j = np.interp(SS_samp_j, SS_interp_set, burden_set)
    poly_probs_concordant = np.interp(SS_samp_i + SS_samp_j + 2*np.sqrt(SS_samp_i*SS_samp_j), 
                                      SS_interp_set, poly_prob_set)
    poly_probs_discordant = np.interp(SS_samp_i + SS_samp_j - 2*np.sqrt(SS_samp_i*SS_samp_j),
                                      SS_interp_set, poly_prob_set)
    num = np.mean(np.sqrt(SS_samp_i * SS_samp_j) * (ex_burdens_i + ex_burdens_j) *
                    (poly_probs_concordant - poly_probs_discordant))
    # Do a second set of samples for the denominator so errors are uncorrelated
    SS_samp_i = dfe_samp(n)
    SS_samp_j = dfe_samp(n)
    ex_burdens_i = np.interp(SS_samp_i, SS_interp_set, burden_set)
    ex_burdens_j = np.interp(SS_samp_j, SS_interp_set, burden_set)
    poly_probs_concordant = np.interp(SS_samp_i + SS_samp_j + 2*np.sqrt(SS_samp_i*SS_samp_j), 
                                      SS_interp_set, poly_prob_set)
    poly_probs_discordant = np.interp(SS_samp_i + SS_samp_j - 2*np.sqrt(SS_samp_i*SS_samp_j),
                                      SS_interp_set, poly_prob_set)
    denom = np.mean(SS_samp_i * (ex_burdens_i + ex_burdens_j) *
                    (poly_probs_concordant + poly_probs_discordant))
    # Do the same for P_bar and S_bar
    SS_samp = dfe_samp(n)
    P_bar = np.mean(np.interp(SS_samp, SS_interp_set, poly_prob_set))
    S_bar = np.mean(SS_samp * np.interp(SS_samp, SS_interp_set, poly_prob_set))
    return num / (P_bar * S_bar + denom), num / denom

def corr_DFE_auto1_quad(yy, dfe_f):
    """Correlation under a DFE model with symmetric mutation and perfect autocorrelation
    
    Parameters:
    - yy (float): The MAF threshold.
    - dfe_f (function): Function that calculates the DFE for a given SS value.

    Returns:
    - float: The correlation between the two traits.
    - float: The correlation between the two traits under D>0
    """
    num_func = lambda SS: SS * ex_burden(SS) * (poly_prob(yy, 4*SS) - poly_prob(yy, 0)) * dfe_f(SS)
    denom_func = lambda SS: SS * ex_burden(SS) * (poly_prob(yy, 4*SS) + poly_prob(yy, 0)) * dfe_f(SS)
    num = spi.quad(num_func, 0, np.inf)[0]
    denom = spi.quad(denom_func, 0, np.inf)[0]
    P_bar = spi.quad(lambda x: poly_prob(yy, x) * dfe_f(x), 0, np.inf)[0]
    S_bar = spi.quad(lambda x: poly_prob(yy, x) * x * dfe_f(x), 0, np.inf)[0]

    return num / (P_bar * S_bar + denom), num / denom

def corr_DFE_auto1(yy, dfe_f, ppf_f):
    """Same as corr_DFE_auto1_quad but using trapezoid rule instead of quadrature"""
    # get upper and lower bounds based on cdf_f
    SS_lower = ppf_f(1e-4)
    SS_upper = ppf_f(1-1e-4)
    SS_set_tmp = np.logspace(np.log10(SS_lower), np.log10(SS_upper), 100)
    num_func = np.vectorize(lambda SS: SS * ex_burden(SS) * (poly_prob(yy, 4*SS) - poly_prob(yy, 0)) * dfe_f(SS))
    denom_func = np.vectorize(lambda SS: SS * ex_burden(SS) * (poly_prob(yy, 4*SS) + poly_prob(yy, 0)) * dfe_f(SS))
    num = np.trapz(num_func(SS_set_tmp), SS_set_tmp)
    denom = np.trapz(denom_func(SS_set_tmp), SS_set_tmp)

    P_func = np.vectorize(lambda SS: poly_prob(yy, SS) * dfe_f(SS))
    S_func = np.vectorize(lambda SS: SS * poly_prob(yy, SS) * dfe_f(SS))
    
    P_bar = np.trapz(P_func(SS_set_tmp), SS_set_tmp)
    S_bar = np.trapz(S_func(SS_set_tmp), SS_set_tmp)

    return num / (P_bar * S_bar + denom), num / denom

### Pleiotropy functions ###
#############################
from scipy.optimize import bisect
from scipy.integrate import quad

def F_Sp(SS, nn, S1):
    return 1 - np.exp( spy.loggamma(nn/2) - spy.loggamma((nn-1)/2) )  * 2* np.sqrt(S1/(np.pi*SS)) * spy.hyp2f1(0.5, (3-nn)/2, 1.5, S1/SS)

def F_inv_Sp(xx, nn, S1):
    # Solve for the Sp value that corresponds to the given quantile xx
    #return np.abs(fsolve(lambda SS: F_Sp(np.abs(SS), nn, S1) - xx, 1+1e-8)[0])    
    return np.abs(bisect(lambda SS: F_Sp(np.abs(SS), nn, S1) - xx, 1, 1e16))

# vectorize the function
F_inv_Sp_vec = np.vectorize(F_inv_Sp)

def sample_cos_alpha(nn, size):
    """
    Sample the cosine between the two random angles in n-dimensions

    nn:   number of dimensions
    size: size of the array to sample
    """
    if nn == 1:
        return np.random.choice([-1, 1], size=size)
    elif nn > 1:
        return 2 * np.random.beta((nn-1)/2, (nn-1)/2, size=size) - 1
    elif nn < 1:
        return np.zeros(size)
    else:
        return None

def Sp_sample(nn, S1, size=1000):
    """
    Sample from the Sp distribution using the CDF F_Sp and the inverse transform sampling method.
    """
    UU = np.random.uniform(size=size)
    return F_inv_Sp_vec(UU, nn, S1)

def Sp_sample_hap(nn, S1, n=1000):
    S1i = Sp_sample(nn, S1, n) - S1
    S1j = Sp_sample(nn, S1, n) - S1
    cos_alpha = sample_cos_alpha(nn-1)
    return S1i + S1j + 2 * np.sqrt(S1i + S1j) * cos_alpha

def Sp_sample_samps(nn, S1, burden_set, poly_prob_set, sigma_set, size=1000):

    SS_set = np.logspace(np.log10(S1), 8, 100000)
    F_Sp_set = F_Sp(SS_set, nn, S1)
    F_Sp_set[0] = 0
    
    Spi = np.interp(np.random.uniform(size=n), F_Sp_set, SS_set) - S1
    Spj = np.interp(np.random.uniform(size=n), F_Sp_set, SS_set) - S1

    # Calculate avg allele burden for sampled S values
    burden_i = np.interp(Spi + S1, sigma_set, burden_set)
    burden_j = np.interp(Spj + S1, sigma_set, burden_set)

    # Sample the cos angle between the two pleiotropic effects
    cos_alpha = sample_cos_alpha(nn-1, size)

    # Compute approximate selection coefficients for double mutant haplotypes
    double_concordant =  Spi + Spj + 2*np.sqrt(Spi*Spj)*cos_alpha + 4*S1
    double_discordant =  Spi + Spj + 2*np.sqrt(Spi*Spj)*cos_alpha

    # Compute polymorphism probabilities for the double mutants
    poly_probs_concordant = np.interp(double_concordant, sigma_set, poly_prob_set)
    poly_probs_discordant = np.interp(double_discordant, sigma_set, poly_prob_set)

    return (burden_i + burden_j) * (poly_probs_concordant - poly_probs_discordant), \
           (burden_i + burden_j) * (poly_probs_concordant + poly_probs_discordant)

def Sp_sample_xi(nn, S1, burden_set, poly_prob_set, sigma_set, size=1000):
    num, denom = Sp_sample_samps(nn, S1, burden_set, poly_prob_set, sigma_set, size)
    return np.nansum(num) / np.nansum(denom)

def dfe_integrate(num_set, denom_set, S1_set, f_S1):
    def numerator(S1):
        return np.interp(S1, S1_set, num_set) * f_S1(S1)
    def denominator(S1):
        return np.interp(S1, S1_set, denom_set) * f_S1(S1)
    # use quad to integrate
    num_int = quad(numerator, 0, np.infty, limit=1000)[0]
    denom_int = quad(denominator, 0, np.infty, limit=1000)[0]
    return num_int / denom_int

def vmf_sample(nn, kappa, SS, poly_prob_set, sigma_set, size):
    # Sample first vectors from uniform distribution on the sphere (size x nn)
    u1 = stats.uniform_direction(nn).rvs(size)
    # Sample the angle from the von Mises-Fisher distribution (size x nn)
    u2 = np.zeros((size, nn))
    for i in range(size):
        u2[i] = stats.vonmises_fisher(u1[i], kappa).rvs()
    # burden_S = ex_burden(SS)
    weight = np.abs(u1[:,0]) + np.abs(u2[:,0])
    S_concordant = SS * np.sum((u1 + u2)**2, axis=1)
    S_discordant = SS * np.sum((u1 - u2)**2, axis=1)
    poly_probs_concordant = np.interp(S_concordant, sigma_set, poly_prob_set)
    poly_probs_discordant = np.interp(S_discordant, sigma_set, poly_prob_set)

    return weight, poly_probs_concordant, poly_probs_discordant

### PLOTTING FUNCTIONS ###
##########################
  
def are_points_touching(coord1, radius1, coord2, radius2):
    """
    Determines whether two points are touching or overlapping based on their coordinates and radii.

    Parameters:
    coord1 (tuple): The coordinates of the first point in the form (x, y).
    radius1 (float): The radius of the first point.
    coord2 (tuple): The coordinates of the second point in the form (x, y).
    radius2 (float): The radius of the second point.

    Returns:
    bool: True if the points are touching or overlapping, False otherwise.
    """
    distance_squared = (coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2
    min_distance_squared = (radius1 + radius2)**2
    return distance_squared < min_distance_squared

def scale_sizes(sizes, scaling_factor):
    """
    Scales the sizes of a list of objects by a given scaling factor.
    
    Args:
        sizes (list): A list of sizes to be scaled.
        scaling_factor (float): The factor by which the sizes should be scaled.
    
    Returns:
        list: A new list containing the scaled sizes.
    """
    scaled_sizes = [size * scaling_factor for size in sizes]
    return scaled_sizes

def adjust_sizes(coordinates, sizes, final_scale=0.5):
    """
    Adjusts the sizes of points based on their coordinates to avoid overlapping.
    
    Args:
        coordinates (list): List of coordinate tuples (x, y) for each point.
        sizes (list): List of sizes for each point.
        final_scale (float, optional): Final scaling factor to apply to the sizes. Defaults to 0.5.
    
    Returns:
        list: List of adjusted sizes for each point.
    """

    # Initial guess for scaling factor
    scaling_factor = 1.0

    # Perform iterations to adjust sizes
    for _ in range(100):  # You can adjust the number of iterations
        # Scale sizes
        scaled_sizes = scale_sizes(sizes, scaling_factor)

        # Check if any points are touching
        touching = False
        for i, coord1 in enumerate(coordinates):
            for j, coord2 in enumerate(coordinates):
                if i != j and are_points_touching(coord1, scaled_sizes[i], coord2, scaled_sizes[j]):
                    touching = True
                    break

        # If points are not touching, break the loop
        if not touching:
            break

        # Adjust scaling factor
        scaling_factor *= 0.95

    return [size * final_scale for size in scaled_sizes]

def weighted_linear_regression(coordinates, probabilities):
    """
    Perform weighted linear regression on the given coordinates and probabilities.

    Args:
        coordinates (list): List of coordinate pairs (x, y).
        probabilities (list): List of corresponding probabilities.

    Returns:
        tuple: A tuple containing the slope and intercept of the regression line.
    """

    # Convert the coordinates and probabilities to numpy arrays for easy calculations
    x_values, y_values = np.array(coordinates).T
    probabilities = np.array(probabilities)

    # Calculate weighted means
    weighted_x_mean = np.sum(x_values * probabilities) / np.sum(probabilities)
    weighted_y_mean = np.sum(y_values * probabilities) / np.sum(probabilities)

    # Calculate weighted covariance and variance
    weighted_covariance = np.sum(probabilities * (x_values - weighted_x_mean) * (y_values - weighted_y_mean))
    weighted_variance = np.sum(probabilities * (x_values - weighted_x_mean)**2)

    # Calculate the slope and intercept
    slope = weighted_covariance / weighted_variance
    intercept = weighted_y_mean - slope * weighted_x_mean

    return slope, intercept

def cross_plot(coord_list, size_list, figsize=(10, 10), fig=None, ax=None, 
               annotate=False, annotate_fontsize=20, label_fontsize=27):
    """
    Function to plot a cross with the given coordinates and sizes.
    Draw circles at the given coordinates with the given sizes, scaled so that circles do not overlap.
    Draw a cross at (0, 0).
    Draw a regression line assuming the proportion of observations at each point in coord_list is given by size_list.

    Parameters
    ----------
    coord_list : list of tuples
        List of coordinates to plot.
    size_list : list of floats
        List of sizes to plot. These are the areas of the circles.
    figsize : tuple
        Size of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    slope, intercept = weighted_linear_regression(coord_list, np.array(size_list) / np.sum(size_list))

    # calculate radii from areas
    radii = np.sqrt(np.array(size_list) / np.pi)

    # Draw a cross at (0, 0)
    ax.axhline(0, color='black', linestyle='-', linewidth=3, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.5)

    scaled_radii = adjust_sizes(coord_list, radii)

    # Draw circles
    for i, coord in enumerate(coord_list):
        ax.add_patch(plt.Circle(coord, scaled_radii[i], color='b', 
                                edgecolor="black", alpha=0.5, linewidth=2.5))

    # get boundaries of the plot based on the radii
    x_max = max([np.abs(coord[0]) + radius for coord, radius in zip(coord_list, scaled_radii)])
    y_max = max([np.abs(coord[1]) + radius for coord, radius in zip(coord_list, scaled_radii)])
    overall_max = max(x_max, y_max)

    x = np.linspace(-overall_max*1.05, overall_max*1.05, 100)
    y = slope * x + intercept
    ax.plot(x, y, color='r', linestyle='--', linewidth=4, alpha=0.7)

    ax.set_xlim(-overall_max*1.05, overall_max*1.05)
    ax.set_ylim(-overall_max*1.05, overall_max*1.05)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axis('off')

    ax.text(overall_max*1.2, 0, r"$\beta_i$", fontsize=label_fontsize, ha='center', va='center')
    ax.text(0, overall_max*1.2, r"$\beta_j$", fontsize=label_fontsize, ha='center', va='center')

    if annotate:
        # Add the regression line equation in the top right corner
        ax.text(overall_max*1.2, overall_max*1.2, 
                r'$\beta_j = {slope:.2f}\beta_i + {intercept:.1f}$'.format(slope=slope, intercept=intercept),
                fontsize=annotate_fontsize, ha='right', va='top')

    return fig, ax, overall_max

def annotate_selection(ax, overall_max, SS=None, yy=None, p_neut=None, qq=None, annotate_fontsize=20, ll=0.03):
    ax.plot([1, 1], [0, ll], 'k-', linewidth=1)
    ax.plot([0, ll], [1, 1], 'k-', linewidth=1)
    ax.plot([-1, -1], [0, ll], 'k-', linewidth=1)
    ax.plot([0, ll], [-1, -1], 'k-', linewidth=1)
    # Add the selection coefficient and y^* to the plot in the upper left corner
    if p_neut is None and qq is None:
        ax.text(-overall_max*1.2, overall_max*1.2, 
                r'$S={SS}$'.format(SS=SS)+r' $y^*={yy}$'.format(yy=yy), 
                fontsize=annotate_fontsize, ha='left', va='top')
    elif p_neut is not None:
        ax.text(-overall_max*1.2, overall_max*1.2, 
                r'$S={SS}$'.format(SS=SS)+r' $y^*={yy}$'.format(yy=yy)+ "\n          " +r'$\pi={p_neut}$'.format(p_neut=p_neut), 
                fontsize=annotate_fontsize, ha='left', va='top')
    elif qq is not None:
        ax.text(-overall_max*1.2, overall_max*1.2, 
                r'$S={SS}$'.format(SS=SS)+r' $y^*={yy}$'.format(yy=yy)+ "\n          " +r'$q={qq}$'.format(qq=qq), 
                fontsize=annotate_fontsize, ha='left', va='top')


def cross_plot_toy_model_1(SS, yy, figsize=(10,10), fig=None, ax=None, scaling=100, pos_LD=True,
                           annotate=False, annotate_fontsize=20, label_fontsize=27):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    burd = ex_burden(SS)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)

    coord_list = [(1,1), (1,-1), (-1,-1), (-1,1)]
    if pos_LD:
        size_list = np.array([double_prob, neut_prob, double_prob, neut_prob]) * scaling
    else:
        concordant = burd * double_prob + single_prob**2
        discordant = burd * neut_prob + single_prob**2
        size_list = np.array([concordant, discordant, concordant, discordant]) * scaling

    fig, ax, overall_max = cross_plot(coord_list, size_list, fig=fig, ax=ax, 
                                      annotate=annotate, annotate_fontsize=annotate_fontsize,
                                      label_fontsize=label_fontsize)

    if annotate:
        annotate_selection(ax, overall_max, SS=SS, yy=yy, annotate_fontsize=annotate_fontsize)
        # Might be nice to add the beta labels
        # ax.text(-ll*2, 1, r'$+\beta$', ha='right', va='center', fontsize=18, color='black')
        # ax.text(1, -ll*2, r'$+\beta$', ha='center', va='top', fontsize=18, color='black')
        # ax.text(-ll*2, -1, r'$-\beta$', ha='right', va='center', fontsize=18, color='black')
        # ax.text(-1, -ll*2, r'$-\beta$', ha='center', va='top', fontsize=18, color='black')

    return fig, ax

def cross_plot_toy_model_2(SS, yy, p_neut, figsize=(10,10), fig=None, ax=None, scaling=100, 
                           pos_LD=True, annotate=False, annotate_fontsize=20):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    burd = ex_burden(SS)
    neut_burd = ex_burden(1e-8)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)

    coord_list = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,-1), (-1,1)]
    if pos_LD:
        zero = 2 * neut_burd * neut_prob * p_neut**2
        zero_nonzero = (neut_burd + burd) * single_prob * p_neut * (1-p_neut) / 2
        size_list = np.concatenate(([zero], [zero_nonzero] * 4,
            np.array([double_prob, neut_prob, 
                    double_prob, neut_prob]) * burd * (1-p_neut)**2 / 2)) * scaling
    else:
        zero = (2 * neut_burd * neut_prob + neut_prob**2) * p_neut**2
        zero_nonzero = ((neut_burd + burd) * single_prob + 
                        single_prob * neut_prob) * p_neut * (1-p_neut) / 2
        concordant = (2 * burd * double_prob + single_prob**2) * (1-p_neut)**2 / 4
        discordant = (2 * burd * neut_prob + single_prob**2)   * (1-p_neut)**2 / 4
        size_list = np.concatenate(([zero], [zero_nonzero] * 4,
            np.array([concordant, discordant, 
                        concordant, discordant]))) * scaling

    fig, ax, overall_max = cross_plot(coord_list, size_list, fig=fig, ax=ax, 
                                        annotate=annotate, annotate_fontsize=annotate_fontsize)

    if annotate:
        annotate_selection(ax, overall_max, SS=SS, yy=yy, p_neut=p_neut, annotate_fontsize=annotate_fontsize)

def cross_plot_toy_model_3(SS, yy, p_neut, figsize=(10,10), fig=None, ax=None, scaling=100, pos_LD=True,
                           annotate=False, annotate_fontsize=20):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    burd = ex_burden(SS)
    neut_burd = ex_burden(1e-8)
    neut_prob = poly_prob(yy, 0)
    double_prob = poly_prob(yy, 4*SS)
    single_prob = poly_prob(yy, SS)

    coord_list = [(0,0), (1,1), (1,-1), (-1,-1), (-1,1)]
    if pos_LD:
        size_list = np.concatenate(([2 * neut_burd * neut_prob * p_neut],
                                    np.array([double_prob, neut_prob, 
                                              double_prob, neut_prob]) * burd * (1-p_neut) / 2)) * scaling                                    
    else:
        zero = (2 * neut_burd * neut_prob + neut_prob**2) * p_neut
        concordant = (2 * burd * double_prob + single_prob**2) * (1-p_neut) / 4
        discordant = (2 * burd * neut_prob + single_prob**2) * (1-p_neut) / 4
        size_list = np.array([zero, concordant, discordant, concordant, discordant]) * scaling

    fig, ax, overall_max = cross_plot(coord_list, size_list, fig=fig, ax=ax, 
                                      annotate=annotate, annotate_fontsize=annotate_fontsize)

    if annotate:
        annotate_selection(ax, overall_max, SS=SS, yy=yy, p_neut=p_neut, annotate_fontsize=annotate_fontsize)

    return fig, ax

def cross_plot_toy_model_4(SS, yy, p_neut, figsize=(10,10), fig=None, ax=None, scaling=100, pos_LD=True,
                            annotate=False, annotate_fontsize=20):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    burd = ex_burden_dir(SS)
    neut_burd = ex_burden_dir(1e-8)
    neut_prob = poly_prob_dir(yy, 0)
    double_prob = poly_prob_dir(yy, 4*SS)
    single_prob = poly_prob_dir(yy, SS)

    coord_list = [(0,0), (0,1), (1,0), (1,1)]

    if pos_LD:
        zero = 2 * neut_burd * neut_prob * p_neut**2
        zero_nonzero = (neut_burd + burd) * single_prob * p_neut * (1-p_neut)
        nonzero_nonzero = 2 * burd * double_prob * (1-p_neut)**2
    else:
        zero = (2 * neut_burd * neut_prob + neut_prob**2) * p_neut**2
        zero_nonzero = ((neut_burd + burd) * single_prob + single_prob*neut_prob) * p_neut * (1-p_neut)
        nonzero_nonzero = (2 * burd * double_prob + single_prob**2) * (1-p_neut)**2
    size_list = np.concatenate(([zero], [zero_nonzero] * 2, [nonzero_nonzero])) * scaling

    fig, ax, overall_max = cross_plot(coord_list, size_list, fig=fig, ax=ax, 
                                      annotate=annotate, annotate_fontsize=annotate_fontsize)
    
    if annotate:
        annotate_selection(ax, overall_max, SS=SS, yy=yy, p_neut=p_neut, annotate_fontsize=annotate_fontsize)

    return fig, ax

def cross_plot_toy_model_5(SS, yy, qq, kk, figsize=(10,10), fig=None, ax=None, scaling=100,
                            annotate=False, annotate_fontsize=20, zero_weight=0.01):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    burd_plus = ex_burden_dir(SS)
    burd_minus = ex_burden_dir(qq*SS)
    double_plus = poly_prob_dir(yy, 2*kk*SS)
    double_minus = poly_prob_dir(yy, 2*kk*qq*SS)
    double_mix = poly_prob_dir(yy, kk*(SS+qq*SS))

    P_2 = 2 * burd_plus * double_plus
    P_1 = (burd_plus + burd_minus) * double_mix
    P_0 = 2 * burd_minus * double_minus

    coord_list = [(-1,-1), (-1,1), (1,-1), (1,1), (0,0)]

    size_list = np.concatenate(([P_0], [P_1] * 2, [P_2], [P_0*zero_weight])) * scaling

    fig, ax, overall_max = cross_plot(coord_list, size_list, fig=fig, ax=ax, 
                                      annotate=annotate, annotate_fontsize=annotate_fontsize)
    if annotate:
        annotate_selection(ax, overall_max, SS=SS, yy=yy, qq=qq, annotate_fontsize=annotate_fontsize)
    
    return fig, ax


### SCHEMATIC FIGURES ###
#########################

def draw_schematic_1(fig, ax, beta=0.5):
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['axes.labelsize'] = 30

    effect_sizes = [-beta, beta]
    probabilities = [0.5, 0.5]

    ax.stem(effect_sizes, probabilities, basefmt=' ', linefmt='k-', markerfmt='ko')

    ax.set_ylabel('Probability', fontsize=20)

    ax.set_xticks([-beta, beta])
    ax.set_xticklabels([r'-$\beta$', r'$+\beta$'], fontsize=20)

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=20)

    ax.axvline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.axhline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.grid(axis='y', linestyle='--', alpha=0.0) 

    ax.set_xlim(-1.5 * beta, 1.5 * beta)

    ax.set_ylim(top=1)

def draw_schematic_2(fig, ax, p_neut, beta=0.5):
    """
    Same as draw_schematic_1 but with a point mass at zero with probability p_neut
    """
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['axes.labelsize'] = 30

    effect_sizes = [-beta, 0, beta]
    probabilities = [(1-p_neut)/2, p_neut, (1-p_neut)/2]

    ax.stem(effect_sizes, probabilities, basefmt=' ', linefmt='k-', markerfmt='ko')
    ax.text(0.08, p_neut, r'$\pi$', fontsize=30, ha='center', va='bottom')

    ax.set_ylabel('Probability', fontsize=20)

    ax.set_xticks([-beta, 0, beta])
    ax.set_xticklabels([r'-$\beta$', 0, r'$+\beta$'], fontsize=20)

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=20)

    ax.axvline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.axhline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.grid(axis='y', linestyle='--', alpha=0.0) 

    ax.set_xlim(-1.5 * beta, 1.5 * beta)

    ax.set_ylim(top=1)


def draw_schematic_3(fig, ax, p_neut, beta=0.5):
    """
    Same as draw_schematic_2 but only a positive effect size
    """
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['axes.labelsize'] = 30

    effect_sizes = [0, beta]
    probabilities = [ p_neut, (1-p_neut)]

    ax.stem(effect_sizes, probabilities, basefmt=' ', linefmt='k-', markerfmt='ko')
    ax.text(0.08, p_neut, r'$\pi$', fontsize=30, ha='center', va='bottom')

    ax.set_ylabel('Probability', fontsize=20)

    ax.set_xticks([0, beta])
    ax.set_xticklabels([0, r'$+\beta$'], fontsize=20)

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=20)

    ax.axvline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.axhline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.grid(axis='y', linestyle='--', alpha=0.0) 

    ax.set_xlim(-0.2, 1.5 * beta)

    ax.set_ylim(top=1)

def draw_schematic_4(fig, ax, beta):
    """
    Same as draw_schematic_1 but give labels to the points at the top 
    of the stems. Label -\beta as qS and \beta as S.
    """
    matplotlib.rcParams.update({'font.size': 20})
    matplotlib.rcParams['axes.labelsize'] = 30

    effect_sizes = [-beta, beta]
    probabilities = [0.5, 0.5]

    ax.stem(effect_sizes, probabilities, basefmt=' ', linefmt='k-', markerfmt='ko')

    ax.set_ylabel('Probability', fontsize=20)

    ax.set_xticks([-beta, beta])
    ax.set_xticklabels([r'$-\beta$', r'$+\beta$'], fontsize=20)

    # Add labels to the points
    ax.text(-beta, 0.5, r'$qS$', fontsize=40, ha='center', va='bottom')
    ax.text(beta, 0.5, r'$S$', fontsize=40, ha='center', va='bottom')

    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=20)

    ax.axvline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.axhline(0, color='black', linewidth=1.3, linestyle='--') 

    ax.grid(axis='y', linestyle='--', alpha=0.0) 

    ax.set_xlim(-1.5 * beta, 1.5 * beta)

    ax.set_ylim(top=1)