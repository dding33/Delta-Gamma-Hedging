import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import norm
import pdb
import matplotlib.pyplot as plt 
import scipy.stats
from tqdm import tqdm

np.random.seed(123)
# creating black scholes price and greeks
class BS():
    
    def CallPrice(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        dm = (np.log(S/K) + (r-0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        
        return S*norm.cdf(dp) - K*np.exp(-r*T)*norm.cdf(dm)
    
    def PutPrice(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        dm = (np.log(S/K) + (r-0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        
        return K*np.exp(-r*T)*norm.cdf(-dm) - S*norm.cdf(-dp)
    
    def CallDelta(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/(np.sqrt(T)*sigma)
        return norm.cdf(dp)
    
    def PutDelta(S, T, K, sigma, r):
        
        return BS.CallDelta(S, T, K, sigma, r)-1
    
    def CallGamma(S, T, K, sigma, r):
        
        dp = (np.log(S/K) + (r+0.5*sigma**2)*T)/np.sqrt(T)/sigma
        
        return norm.pdf(dp)/(S*sigma*np.sqrt(T))
    
    def PutGamma(S, T, K, sigma, r):
        
        return BS.CallGamma(S, T, K, sigma, r)

# get s
def get_dS(dt, mu, sigma, S_last):
    std = np.sqrt(dt)
    z = np.random.normal(0, std,1)[0]
    return mu * S_last * dt + sigma * S_last * z

# get m: alpha denotes position held in the stock, beta is position in the call option
def get_M(M_last, r, dt, alpha, alpha_last, S, phi_equity, C, beta, beta_last, phi_option):
    return (M_last * np.exp(r * dt) - (alpha - alpha_last) * S - (beta - beta_last)* C \
        - phi_equity * np.abs(alpha - alpha_last) - phi_option * np.abs(beta - beta_last))

def BS_put(S, T, K, sigma, r):
    return BS.PutPrice(S, T, K, sigma, r), BS.PutDelta(S, T, K, sigma, r), BS.PutGamma(S, T, K, sigma, r)

def BS_call(S, T, K, sigma, r):
    return BS.CallPrice(S, T, K, sigma, r), BS.CallDelta(S, T, K, sigma, r), BS.CallGamma(S, T, K, sigma, r)
# Set up
S0 = 100
K = 100
K_call = 100
sigma = 0.2
mu = 0.1
r = 0.02
T = 0.25

phi_equity = 0.005
phi_option = 0.01

P0, delta_P, gamma_P = BS_put(S0, T, K, sigma, r)
C0, delta_C, gamma_C = BS_call(S0, T, K_call, sigma, r)

beta0 = gamma_P/gamma_C
alpha0 = delta_P - beta0*delta_C

M0 = P0 - alpha0 * S0 - phi_equity * np.abs(alpha0) \
    - beta0 * C0 - phi_option * np.abs(beta0)

dt = 1/365 # day is the smallest interval
Ndt = int(T*365)
t_list = np.linspace(0,T, int(round(Ndt,0))+1)
list_M_final = []
check_s = lambda s: s < K
for sim in tqdm(range(1000)):
    s_init = S0
    alpha_init = alpha0
    beta_init = beta0

    M_init = M0
    
    for day in t_list[:-1]: #Gamma can only be calculated till two days before expiry since it's the second order derivative

        stock_price = s_init + get_dS(dt, mu, sigma, s_init)
        call_price, delta_C, gamma_C = BS_call(stock_price, T-day, K_call, sigma, r)
        delta_P, gamma_P = BS_put(stock_price, T-day, K, sigma, r)[1:]
        beta = gamma_P/gamma_C
        alpha = delta_P - beta*delta_C

        money_account = get_M(M_init, r, dt, alpha, alpha_init, stock_price, phi_equity, call_price, beta, beta_init, phi_option)

        s_init = stock_price
        alpha_init = alpha
        beta_init = beta
        M_init = money_account

    stock_price_final = s_init + get_dS(dt, mu, sigma, s_init)
    call_price_final = BS.CallPrice(stock_price_final, T-t_list[-1], K, sigma, r)

    # financial settling
    money_account = M_init * np.exp(r*dt) \
            + alpha * stock_price_final - phi_equity * np.abs(alpha) \
                + beta * call_price_final - phi_option * np.abs(beta) \
                    + (stock_price_final < K) * (- K + stock_price_final)

    list_M_final.append(money_account)

plt.hist(list_M_final,50)
plt.xlabel('Profit or Loss at time T')
plt.ylabel('count')
plt.title('P/L Distribution, Time-based Delta Hedge')
plt.savefig('Profit Distribition, Time-based Delta Gamma Hedge.png')
plt.show()

# Calculate CVaR
list_M_final = np.array(list_M_final)
confidence = 10
mean_hedge = np.average(list_M_final)
std_hedge = np.std(list_M_final)
VaR = np.percentile(list_M_final, confidence)
CVaR = np.mean(list_M_final[list_M_final<=VaR])
m = - CVaR - 0.02
m_discounted = m * np.exp(-r * T)
price_CVaR_Adjusted = P0 + m_discounted
print(price_CVaR_Adjusted)
