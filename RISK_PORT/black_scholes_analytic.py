# Author: Jonathan Tekverk

# - Math utility functions for the analytic European option solution to Black-Scholes
# - All interest rates are compounded continuously
# - Time units are given in units of calendar years

import numpy as np
from scipy.special import erf

class EuropeanOptions:

    @staticmethod
    def norm_pdf(x):
        """
        - Normal probability density function (PDF)
        - Computes φ(x) = (1/√(2π)) * exp(-x² / 2) for scalar or numpy array input
        - Used in computing option Greeks within the Black-Scholes framework
        
        Inputs:
        - x : float or numpy array

        Outputs:
        - pdf : float or numpy array
        """        
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)

    @staticmethod
    def norm_cdf(x):
        """
        - Normal cumulative distribution function (CDF)
        - Computes N(x) = 0.5 * [1 + erf(x / √2)] for scalar or numpy array input
        - Used to calculate N(d1) and N(d2) in the Black-Scholes pricing formula
        
        Inputs:
        - x : float or numpy array

        Outputs:
        - cdf : float or numpy array
        """
        x = np.asarray(x, dtype=float)

        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def d1_d2(S, K, r, q, sigma, T):
        """
        - Computes d1 and d2 for the Black-Scholes model in forward-price form
        - Forward price: F = S * exp((r - q) * T)
        - d1 = [ln(F / K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
        - d2 = d1 - sigma * sqrt(T)
        
        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float
        
        Outputs:
        - (d1, d2)                       : tuple of floats or numpy arrays
        """
        if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
            return "Warning: Spot (S) and Strike (K) must be positive."
        if np.any(np.asarray(T) <= 0):
            return "Warning: Time to maturity (T) must be positive."
        if np.any(np.asarray(sigma) <= 0):
            return "Warning: Volatility (sigma) must be positive."

        F = S * np.exp((r - q) * T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - (sigma * np.sqrt(T) )

        return d1, d2

    @staticmethod
    def call_price(S, K, r, q, sigma, T):
        """
        - Black-Scholes European call option price (forward-price form)
        - Forward price: F = S * exp((r - q) * T)
        - Closed form: C = exp(-r*T) * [ F * N(d1) - K * N(d2) ]
        - Uses d1, d2 from the forward-price formulation

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Call option price              : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd
        F = S * np.exp((r - q) * T)

        return np.exp(-r * T) * (F * EuropeanOptions.norm_cdf(d1) - K * EuropeanOptions.norm_cdf(d2))

    @staticmethod
    def put_price(S, K, r, q, sigma, T):
        """
        - Black-Scholes European put option price (forward-price form)
        - Forward price: F = S * exp((r - q) * T)
        - Closed form: P = exp(-r*T) * [ K * N(-d2) - F * N(-d1) ]
        - Uses d1, d2 from the forward-price formulation

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Put option price               : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd
        F = S * np.exp((r - q) * T)

        return np.exp(-r * T) * (K * EuropeanOptions.norm_cdf(-d2) - F * EuropeanOptions.norm_cdf(-d1))

    @staticmethod
    def put_from_call(C, S, K, r, q, T):
        """
        - Computes the theoretical European put price from the call price using put-call parity
        - Relationship: P = C - S * exp(-q*T) + K * exp(-r*T)
        - Holds under no-arbitrage for European options on the same underlying, strike, and maturity

        Inputs:
        - Call option price C            : float or numpy array
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Time to maturity T             : float

        Outputs:
        - Put option price               : float or numpy array
        """
        if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
            return "Warning: Spot (S) and Strike (K) must be positive."
        if np.any(np.asarray(T) <= 0):
            return "Warning: Time to maturity (T) must be positive."
        
        return C - S * np.exp(-q * T) + K * np.exp(-r * T)

    @staticmethod
    def call_from_put(P, S, K, r, q, T):
        """
        - Computes the theoretical European call price from a known put price using put-call parity
        - Relationship: C = P + S * exp(-q*T) - K * exp(-r*T)
        - Holds under no-arbitrage for European options on the same underlying, strike, and maturity

        Inputs:
        - Put option price P             : float or numpy array
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Time to maturity T             : float

        Outputs:
        - Call option price              : float or numpy array
        """
        if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
            return "Warning: Spot (S) and Strike (K) must be positive."
        if np.any(np.asarray(T) <= 0):
            return "Warning: Time to maturity (T) must be positive."
        
        return P + S * np.exp(-q * T) - K * np.exp(-r * T)

    @staticmethod
    def check_parity(C, P, S, K, r, q, T):
        """
        - Verifies the put-call parity relationship for given prices
        - Theoretical equality: C - P = S * exp(-q*T) - K * exp(-r*T)
        - Returns the parity difference (should be near zero if consistent)

        Inputs:
        - Call option price C            : float or numpy array
        - Put option price P             : float or numpy array
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Time to maturity T             : float

        Outputs:
        - Parity difference              : float or numpy array
        """
        if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
            return "Warning: Spot (S) and Strike (K) must be positive."
        if np.any(np.asarray(T) <= 0):
            return "Warning: Time to maturity (T) must be positive."
        
        return (C - P) - (S * np.exp(-q * T) - K * np.exp(-r * T))

    @staticmethod
    def delta_call(S, K, r, q, sigma, T):
        """
        - Computes Delta (∂V/∂S) for a European call option under the Black-Scholes model
        - Measures the sensitivity of the call price to a change in the underlying asset price
        - In forward form: Delta (call) = exp(-q*T) * N(d1)
        
        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float
        
        Outputs:
        - Delta (call)                   : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd

        d1, _ = dd

        return np.exp(-q * T) * EuropeanOptions.norm_cdf(d1)

    @staticmethod
    def delta_put(S, K, r, q, sigma, T):
        """
        - Computes Delta (∂V/∂S) for a European put option under the Black-Scholes model
        - Measures the sensitivity of the put price to a change in the underlying asset price
        - In forward form: Delta = exp(-q*T) * (N(d1) - 1)
        
        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float
        
        Outputs:
        - Delta                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd

        d1, _ = dd

        return np.exp(-q * T) * (EuropeanOptions.norm_cdf(d1) - 1.0)

    @staticmethod
    def gamma(S, K, r, q, sigma, T):
        """
        - Computes Gamma (∂²V/∂S²) for a European option under the Black-Scholes model
        - Measures the rate of change of Delta with respect to the underlying asset price
        - Identical for calls and puts since payoffs are linear in asset price S
        - In forward form: Gamma = exp(-q*T) * φ(d1) / (S * sigma * sqrt(T))

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Gamma                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd

        d1, _ = dd
        
        return np.exp(-q * T) * EuropeanOptions.norm_pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, r, q, sigma, T):
        """
        - Computes Vega (∂V/∂sigma) for a European option under the Black-Scholes model
        - Sensitivity of the option price to volatility. Identical for calls and puts
        - Vega = S * exp(-q*T) * φ(d1) * sqrt(T) in both spot and forward form

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Vega                           : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, _ = dd

        return S * np.exp(-q * T) * EuropeanOptions.norm_pdf(d1) * np.sqrt(T)

    @staticmethod
    def theta_call(S, K, r, q, sigma, T):
        """
        - Computes Theta (∂V/∂T) for a European call option (per year) under Black-Scholes
        - Time decay of the call option price
        - Theta (call) = -(S * exp(-q*T) * φ(d1) * sigma) / (2*sqrt(T))
                         - r * K * exp(-r*T) * N(d2) + q * S * exp(-q*T) * N(d1)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Theta (call)                   : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd

        term1 = -(S * np.exp(-q * T) * EuropeanOptions.norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
        term2 = -r * K * np.exp(-r * T) * EuropeanOptions.norm_cdf(d2)
        term3 =  q * S * np.exp(-q * T) * EuropeanOptions.norm_cdf(d1)

        return term1 + term2 + term3

    @staticmethod
    def theta_put(S, K, r, q, sigma, T):
        """
        - Computes Theta (∂V/∂T) for a European put option (per year) under Black-Scholes
        - Time decay of the put option price
        - Theta (put) = -(S * exp(-q*T) * φ(d1) * sigma) / (2*sqrt(T))
                + r * K * exp(-r*T) * N(-d2) - q * S * exp(-q*T) * N(-d1)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Theta (put)                    : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd

        term1 = -(S * np.exp(-q * T) * EuropeanOptions.norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
        term2 =  r * K * np.exp(-r * T) * EuropeanOptions.norm_cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * EuropeanOptions.norm_cdf(-d1)

        return term1 + term2 + term3

    @staticmethod
    def rho_call(S, K, r, q, sigma, T):
        """
        - Computes Rho (∂V/∂r) for a European call option under Black-Scholes
        - Sensitivity to the risk-free rate
        - Rho (call) = K * T * exp(-r*T) * N(d2)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Rho (call)                     : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        _, d2 = dd

        return K * T * np.exp(-r * T) * EuropeanOptions.norm_cdf(d2)

    @staticmethod
    def rho_put(S, K, r, q, sigma, T):
        """
        - Computes Rho (∂V/∂r) for a European put option under Black-Scholes
        - Sensitivity to the risk-free rate
        - Rho (put) = -K * T * exp(-r*T) * N(-d2)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Rho (put)                      : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        _, d2 = dd

        return -K * T * np.exp(-r * T) * EuropeanOptions.norm_cdf(-d2)

    @staticmethod
    def phi_call(S, K, r, q, sigma, T):
        """
        - Computes Phi (∂V/∂q) for a European call option
        - Sensitivity to dividend yield q
        - Phi (call) = -T * S * exp(-q*T) * N(d1)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Phi (call)                     : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, _ = dd

        return -T * S * np.exp(-q * T) * EuropeanOptions.norm_cdf(d1)

    @staticmethod
    def phi_put(S, K, r, q, sigma, T):
        """
        - Computes Phi (∂V/∂q) for a European put option
        - Sensitivity to dividend yield
        - Phi (put) = +T * S * exp(-q*T) * N(-d1)

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Phi (put)                      : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, _ = dd

        return T * S * np.exp(-q * T) * EuropeanOptions.norm_cdf(-d1)

    @staticmethod
    def vanna(S, K, r, q, sigma, T):
        """
        - Computes Vanna = ∂²V / (∂S ∂sigma) for European call/put options
        - Measures how Delta (or Vega) changes with volatility (or spot)
        - Vanna = (Vega/S) * (-d2 / sigma) = -exp(-q*T) * φ(d1) * sqrt(T) * d2 / sigma

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Vanna                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd

        return -np.exp(-q * T) * EuropeanOptions.norm_pdf(d1) * np.sqrt(T) * (d2 / sigma)

    @staticmethod
    def vomma(S, K, r, q, sigma, T):
        """
        - Computes Vomma ∂²V / ∂sigma² for European call/put options
        - Sensitivity of Vega to volatility
        - Vomma = Vega * d1 * d2 / sigma, where Vega = S * exp(-q*T) * φ(d1) * sqrt(T)
        
        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Vomma                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd
        vg = EuropeanOptions.vega(S, K, r, q, sigma, T)

        return vg * (d1 * d2) / sigma

    @staticmethod
    def speed(S, K, r, q, sigma, T):
        """
        - Computes Speed = ∂Gamma / ∂S for European call/put options
        - Measures how Gamma changes with spot
        - Speed = -Gamma / S * (1 + d1 / (sigma * sqrt(T)))

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Speed                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, _ = dd
        g = EuropeanOptions.gamma(S, K, r, q, sigma, T)

        return -g / S * (1.0 + d1 / (sigma * np.sqrt(T)))

    @staticmethod
    def zomma(S, K, r, q, sigma, T):
        """
        - Computes Zomma = ∂Gamma / ∂sigma for European call/put options
        - Measures how Gamma changes with volatility
        - Zomma = Gamma * (d1 * d2 - 1) / sigma

        Inputs:
        - Current asset price S          : float or numpy array
        - Strike price K                 : float or numpy array
        - Risk-free interest rate r      : float
        - Dividend yield q               : float
        - Volatility sigma               : float
        - Time to maturity T             : float

        Outputs:
        - Zomma                          : float or numpy array
        """
        dd = EuropeanOptions.d1_d2(S, K, r, q, sigma, T)

        if isinstance(dd, str):
            return dd
        
        d1, d2 = dd
        g = EuropeanOptions.gamma(S, K, r, q, sigma, T)

        return g * (d1 * d2 - 1.0) / sigma




