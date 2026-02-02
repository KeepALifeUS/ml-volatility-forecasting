"""
Implied Volatility Extraction для Crypto Options & Volatility Indices

Реализация advanced implied volatility models:
- Options-based IV extraction (если доступны опционы)
- Volatility smile modeling (вся поверхность волатильности)
- Term structure analysis (временная структура волатильности)
- Crypto Volatility Index (VIX-style для криптовалют)
- Model-free implied volatility
- Risk-neutral density extraction

Features:
- Real-time options data processing
- Advanced volatility surface interpolation
- Greeks calculation и risk management
- Production-ready performance optimization
- Comprehensive error handling
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats, optimize, interpolate
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

# Настройка логирования  
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class OptionType(Enum):
    """Тип опциона"""
    CALL = "call"
    PUT = "put"

class VolatilityRegime(Enum):
    """Режим волатильности"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class OptionData:
    """Данные опциона"""
    symbol: str
    option_type: OptionType
    strike: float
    expiry: datetime
    price: float
    underlying_price: float
    risk_free_rate: float
    timestamp: datetime
    
    # Дополнительные поля
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def time_to_expiry(self) -> float:
        """Время до экспирации в годах"""
        if hasattr(self, '_time_to_expiry'):
            return self._time_to_expiry
        
        time_diff = self.expiry - self.timestamp
        return max(1/365, time_diff.total_seconds() / (365.25 * 24 * 3600))
    
    @property
    def moneyness(self) -> float:
        """Moneyness (S/K)"""
        return self.underlying_price / self.strike
    
    @property 
    def log_moneyness(self) -> float:
        """Log moneyness ln(S/K)"""
        return np.log(self.moneyness)

@dataclass
class ImpliedVolatilityResult:
    """Результат расчета implied volatility"""
    symbol: str
    timestamp: datetime
    option_data: OptionData
    implied_volatility: float
    
    # Метрики качества
    pricing_error: float
    vega: float
    gamma: float
    theta: float
    
    # Метод расчета
    method: str = "black_scholes"
    iterations: int = 0
    convergence_achieved: bool = True
    
    # Дополнительная информация
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    mid_iv: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolatilitySmile:
    """Результат моделирования volatility smile"""
    symbol: str
    expiry: datetime
    timestamp: datetime
    
    # Smile данные
    strikes: np.ndarray
    implied_volatilities: np.ndarray
    
    # Параметры модели
    atm_volatility: float
    skew: float
    kurtosis: float
    smile_parameters: Dict[str, float]
    
    # Качество подгонки
    r_squared: float
    rmse: float
    
    # Interpolation function
    iv_interpolator: Optional[Callable] = None
    
    # Risk-neutral moments
    risk_neutral_skewness: Optional[float] = None
    risk_neutral_kurtosis: Optional[float] = None

@dataclass
class CryptoVolatilityIndex:
    """Crypto Volatility Index (VIX-style)"""
    symbol: str
    timestamp: datetime
    
    # Основные метрики
    volatility_index: float
    regime: VolatilityRegime
    
    # Компоненты
    near_term_vol: float
    next_term_vol: float
    
    # Term structure
    term_structure: Dict[str, float]  # "30D", "60D", "90D", etc.
    
    # Подуровни
    call_vol_index: float
    put_vol_index: float
    
    # Метрики качества
    data_quality_score: float
    n_options_used: int
    
    # Сравнение с историческими уровнями
    percentile_1m: float
    percentile_3m: float
    percentile_1y: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    """
    Black-Scholes опционное ценообразование
    
    Args:
        S: Цена базового актива
        K: Страйк
        T: Время до экспирации
        r: Безрисковая ставка
        sigma: Волатильность
        option_type: Тип опциона
    """
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == OptionType.CALL else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == OptionType.CALL:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(0, price)

def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Vega"""
    if T <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Gamma"""
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def black_scholes_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
) -> float:
    """Black-Scholes Theta"""
    if T <= 0:
        return 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    
    if option_type == OptionType.CALL:
        second_term = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        second_term = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return first_term + second_term

class BaseImpliedVolatilityExtractor(ABC):
    """Базовый класс для извлечения implied volatility"""
    
    def __init__(self, symbol: str, name: str):
        self.symbol = symbol
        self.name = name
        self.extraction_history: List[ImpliedVolatilityResult] = []
        
        logger.info(f"🎯 Initialized {name} IV extractor for {symbol}")

    @abstractmethod
    async def extract_iv(self, option_data: OptionData) -> ImpliedVolatilityResult:
        """Извлечение implied volatility из данных опциона"""
        pass

    def _validate_option_data(self, option_data: OptionData) -> Tuple[bool, str]:
        """Валидация данных опциона"""
        if option_data.price <= 0:
            return False, "Option price must be positive"
        
        if option_data.underlying_price <= 0:
            return False, "Underlying price must be positive"
        
        if option_data.strike <= 0:
            return False, "Strike must be positive"
        
        if option_data.time_to_expiry <= 0:
            return False, "Time to expiry must be positive"
        
        # Проверка на arbitrage
        if option_data.option_type == OptionType.CALL:
            intrinsic = max(0, option_data.underlying_price - option_data.strike)
        else:
            intrinsic = max(0, option_data.strike - option_data.underlying_price)
        
        if option_data.price < intrinsic:
            return False, f"Option price {option_data.price} below intrinsic value {intrinsic}"
        
        return True, "Validation passed"

class ImpliedVolatilityExtractor(BaseImpliedVolatilityExtractor):
    """
    Standard Black-Scholes Implied Volatility Extractor
    
    Использует Newton-Raphson итерацию для нахождения implied volatility
    из market prices опционов через Black-Scholes формулу.
    """
    
    def __init__(self, symbol: str, max_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__(symbol, "Black-Scholes IV")
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    async def extract_iv(self, option_data: OptionData) -> ImpliedVolatilityResult:
        """Извлечение IV с Newton-Raphson методом"""
        try:
            # Валидация
            is_valid, message = self._validate_option_data(option_data)
            if not is_valid:
                raise ValueError(f"Invalid option data: {message}")
            
            # Асинхронное вычисление
            loop = asyncio.get_event_loop()
            iv_result = await loop.run_in_executor(
                None, 
                self._newton_raphson_iv, 
                option_data
            )
            
            result = ImpliedVolatilityResult(
                symbol=self.symbol,
                timestamp=datetime.now(),
                option_data=option_data,
                implied_volatility=iv_result["iv"],
                pricing_error=iv_result["error"],
                vega=iv_result["vega"],
                gamma=iv_result["gamma"], 
                theta=iv_result["theta"],
                method="newton_raphson",
                iterations=iv_result["iterations"],
                convergence_achieved=iv_result["converged"]
            )
            
            self.extraction_history.append(result)
            logger.info(f"✅ IV extracted: {iv_result['iv']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error extracting IV: {e}")
            raise

    def _newton_raphson_iv(self, option_data: OptionData) -> Dict[str, Any]:
        """Newton-Raphson поиск implied volatility"""
        
        # Начальное приближение
        if option_data.option_type == OptionType.CALL:
            intrinsic = max(0, option_data.underlying_price - option_data.strike)
        else:
            intrinsic = max(0, option_data.strike - option_data.underlying_price)
        
        # Heuristic для начального приближения
        time_value = option_data.price - intrinsic
        if time_value > 0 and option_data.time_to_expiry > 0:
            initial_iv = np.sqrt(2 * np.pi / option_data.time_to_expiry) * \
                        (time_value / option_data.underlying_price)
        else:
            initial_iv = 0.2  # 20% default
        
        iv = max(0.01, min(5.0, initial_iv))  # Ограничения
        
        # Newton-Raphson итерации
        for i in range(self.max_iterations):
            # Теоретическая цена
            theo_price = black_scholes_price(
                option_data.underlying_price,
                option_data.strike,
                option_data.time_to_expiry,
                option_data.risk_free_rate,
                iv,
                option_data.option_type
            )
            
            # Vega для производной
            vega = black_scholes_vega(
                option_data.underlying_price,
                option_data.strike,
                option_data.time_to_expiry,
                option_data.risk_free_rate,
                iv
            )
            
            # Ошибка цены
            price_error = theo_price - option_data.price
            
            # Проверка сходимости
            if abs(price_error) < self.tolerance:
                break
            
            # Проверка vega
            if abs(vega) < 1e-10:
                break
            
            # Newton-Raphson update
            iv_new = iv - price_error / vega
            iv = max(0.001, min(10.0, iv_new))  # Clamp IV
        
        # Финальные расчеты
        final_price = black_scholes_price(
            option_data.underlying_price,
            option_data.strike,
            option_data.time_to_expiry,
            option_data.risk_free_rate,
            iv,
            option_data.option_type
        )
        
        final_vega = black_scholes_vega(
            option_data.underlying_price,
            option_data.strike,
            option_data.time_to_expiry,
            option_data.risk_free_rate,
            iv
        )
        
        gamma = black_scholes_gamma(
            option_data.underlying_price,
            option_data.strike,
            option_data.time_to_expiry,
            option_data.risk_free_rate,
            iv
        )
        
        theta = black_scholes_theta(
            option_data.underlying_price,
            option_data.strike,
            option_data.time_to_expiry,
            option_data.risk_free_rate,
            iv,
            option_data.option_type
        )
        
        return {
            "iv": iv,
            "error": abs(final_price - option_data.price),
            "vega": final_vega,
            "gamma": gamma,
            "theta": theta,
            "iterations": i + 1,
            "converged": abs(final_price - option_data.price) < self.tolerance
        }

    async def extract_iv_bid_ask(self, option_data: OptionData) -> ImpliedVolatilityResult:
        """Извлечение IV для bid/ask spread"""
        if option_data.bid is None or option_data.ask is None:
            return await self.extract_iv(option_data)
        
        # IV для bid
        bid_option = OptionData(
            symbol=option_data.symbol,
            option_type=option_data.option_type,
            strike=option_data.strike,
            expiry=option_data.expiry,
            price=option_data.bid,
            underlying_price=option_data.underlying_price,
            risk_free_rate=option_data.risk_free_rate,
            timestamp=option_data.timestamp
        )
        
        # IV для ask
        ask_option = OptionData(
            symbol=option_data.symbol,
            option_type=option_data.option_type,
            strike=option_data.strike,
            expiry=option_data.expiry,
            price=option_data.ask,
            underlying_price=option_data.underlying_price,
            risk_free_rate=option_data.risk_free_rate,
            timestamp=option_data.timestamp
        )
        
        try:
            bid_result = await self.extract_iv(bid_option)
            ask_result = await self.extract_iv(ask_option)
            mid_result = await self.extract_iv(option_data)
            
            # Комбинированный результат
            result = mid_result
            result.bid_iv = bid_result.implied_volatility
            result.ask_iv = ask_result.implied_volatility
            result.mid_iv = mid_result.implied_volatility
            
            result.metadata.update({
                "bid_ask_spread_iv": ask_result.implied_volatility - bid_result.implied_volatility,
                "bid_convergence": bid_result.convergence_achieved,
                "ask_convergence": ask_result.convergence_achieved
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Bid-ask IV extraction failed, using mid: {e}")
            return await self.extract_iv(option_data)

class VolatilitySmileModel:
    """
    Volatility Smile Modeling
    
    Моделирование всей поверхности implied volatility как функции
    strike и time to expiry. Поддерживает разные параметрические модели.
    """
    
    def __init__(self, symbol: str, model_type: str = "sabr"):
        self.symbol = symbol
        self.model_type = model_type  # "sabr", "svi", "polynomial"
        self.fitted_smiles: Dict[datetime, VolatilitySmile] = {}
        
        logger.info(f"🎯 Volatility smile model initialized: {model_type}")

    async def fit_smile(
        self,
        options_data: List[OptionData],
        expiry: datetime,
        method: str = "least_squares"
    ) -> VolatilitySmile:
        """
        Подгонка volatility smile для конкретной экспирации
        
        Args:
            options_data: Список опционов для одной экспирации
            expiry: Дата экспирации
            method: Метод подгонки
        """
        try:
            logger.info(f"🔄 Fitting volatility smile for {expiry.date()}...")
            
            # Фильтрация данных по экспирации
            expiry_options = [opt for opt in options_data if opt.expiry.date() == expiry.date()]
            
            if len(expiry_options) < 3:
                raise ValueError(f"Insufficient options for smile fitting: {len(expiry_options)}")
            
            # Извлечение IV для всех опционов
            iv_extractor = ImpliedVolatilityExtractor(self.symbol)
            iv_results = []
            
            for option_data in expiry_options:
                try:
                    iv_result = await iv_extractor.extract_iv(option_data)
                    if iv_result.convergence_achieved:
                        iv_results.append(iv_result)
                except:
                    continue
            
            if len(iv_results) < 3:
                raise ValueError("Insufficient valid IV extractions")
            
            # Подготовка данных для подгонки
            strikes = np.array([r.option_data.strike for r in iv_results])
            ivs = np.array([r.implied_volatility for r in iv_results])
            underlying_price = iv_results[0].option_data.underlying_price
            
            # Moneyness для нормализации
            moneyness = strikes / underlying_price
            log_moneyness = np.log(moneyness)
            
            # Подгонка модели
            if self.model_type == "sabr":
                smile_params = await self._fit_sabr_model(log_moneyness, ivs)
            elif self.model_type == "svi":
                smile_params = await self._fit_svi_model(log_moneyness, ivs)
            else:
                smile_params = await self._fit_polynomial_model(log_moneyness, ivs)
            
            # Создание interpolator
            iv_interpolator = self._create_interpolator(strikes, ivs, smile_params)
            
            # Метрики качества
            predicted_ivs = iv_interpolator(strikes)
            r_squared = 1 - np.sum((ivs - predicted_ivs)**2) / np.sum((ivs - np.mean(ivs))**2)
            rmse = np.sqrt(np.mean((ivs - predicted_ivs)**2))
            
            # ATM volatility и параметры smile
            atm_strike = underlying_price
            atm_iv = float(iv_interpolator(atm_strike))
            
            # Skew и kurtosis
            skew = self._calculate_skew(strikes, ivs, underlying_price)
            kurtosis = self._calculate_kurtosis(strikes, ivs, underlying_price)
            
            # Risk-neutral moments
            rn_skew, rn_kurt = await self._calculate_risk_neutral_moments(
                strikes, ivs, underlying_price, iv_results[0].option_data.time_to_expiry
            )
            
            smile = VolatilitySmile(
                symbol=self.symbol,
                expiry=expiry,
                timestamp=datetime.now(),
                strikes=strikes,
                implied_volatilities=ivs,
                atm_volatility=atm_iv,
                skew=skew,
                kurtosis=kurtosis,
                smile_parameters=smile_params,
                r_squared=r_squared,
                rmse=rmse,
                iv_interpolator=iv_interpolator,
                risk_neutral_skewness=rn_skew,
                risk_neutral_kurtosis=rn_kurt
            )
            
            self.fitted_smiles[expiry] = smile
            logger.info(f"✅ Smile fitted: ATM IV={atm_iv:.4f}, R²={r_squared:.3f}")
            
            return smile
            
        except Exception as e:
            logger.error(f"❌ Error fitting smile: {e}")
            raise

    async def _fit_sabr_model(
        self, 
        log_moneyness: np.ndarray, 
        ivs: np.ndarray
    ) -> Dict[str, float]:
        """Подгонка SABR модели"""
        
        def sabr_iv(log_k, alpha, beta, rho, nu):
            """SABR implied volatility approximation"""
            if len(log_k) == 0:
                return np.array([])
                
            # Simplified SABR approximation для ATM
            f_k_avg = np.exp(log_k / 2)  # Geometric average
            
            # SABR formula (simplified)
            numerator = alpha * (1 + ((beta - 1)**2 / 24) * log_k**2)
            denominator = f_k_avg**(1 - beta)
            
            sabr_vol = numerator / denominator
            
            # Adjustments для skew (rho) и vol-of-vol (nu)
            skew_adj = (rho * beta * nu * alpha / 4) * log_k
            vol_vol_adj = (2 - 3 * rho**2) * nu**2 / 24
            
            return sabr_vol * (1 + skew_adj + vol_vol_adj)
        
        def objective(params):
            alpha, beta, rho, nu = params
            
            # Constraints
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1 or beta < 0 or beta > 1:
                return 1e6
            
            try:
                predicted = sabr_iv(log_moneyness, alpha, beta, rho, nu)
                return np.sum((ivs - predicted)**2)
            except:
                return 1e6
        
        # Initial guess
        initial_guess = [np.mean(ivs), 0.5, 0.0, 0.3]
        
        # Optimization
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: optimize.minimize(
                objective, 
                initial_guess, 
                method='L-BFGS-B',
                bounds=[(0.01, 2), (0, 1), (-0.99, 0.99), (0.01, 2)]
            )
        )
        
        alpha, beta, rho, nu = result.x
        
        return {
            "model": "sabr",
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "nu": nu,
            "success": result.success
        }

    async def _fit_svi_model(
        self,
        log_moneyness: np.ndarray,
        ivs: np.ndarray
    ) -> Dict[str, float]:
        """Подгонка SVI (Stochastic Volatility Inspired) модели"""
        
        def svi_iv(log_k, a, b, rho, m, sigma):
            """SVI parametrization"""
            return np.sqrt(a + b * (rho * (log_k - m) + np.sqrt((log_k - m)**2 + sigma**2)))
        
        def objective(params):
            a, b, rho, m, sigma = params
            
            # SVI constraints
            if a <= 0 or b < 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e6
            
            try:
                predicted = svi_iv(log_moneyness, a, b, rho, m, sigma)
                return np.sum((ivs - predicted)**2)
            except:
                return 1e6
        
        # Initial guess
        atm_var = np.mean(ivs)**2
        initial_guess = [atm_var, 0.1, 0.0, 0.0, 0.1]
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: optimize.minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=[(0.001, 1), (0, 1), (-0.99, 0.99), (-2, 2), (0.001, 1)]
            )
        )
        
        a, b, rho, m, sigma = result.x
        
        return {
            "model": "svi",
            "a": a,
            "b": b,
            "rho": rho,
            "m": m,
            "sigma": sigma,
            "success": result.success
        }

    async def _fit_polynomial_model(
        self,
        log_moneyness: np.ndarray,
        ivs: np.ndarray
    ) -> Dict[str, float]:
        """Подгонка полиномиальной модели"""
        
        # 2nd order polynomial: IV = a + b*k + c*k^2
        X = np.vstack([np.ones(len(log_moneyness)), log_moneyness, log_moneyness**2]).T
        
        loop = asyncio.get_event_loop()
        coefficients = await loop.run_in_executor(
            None,
            lambda: np.linalg.lstsq(X, ivs, rcond=None)[0]
        )
        
        a, b, c = coefficients
        
        return {
            "model": "polynomial",
            "a": a,
            "b": b,
            "c": c,
            "success": True
        }

    def _create_interpolator(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        params: Dict[str, float]
    ) -> Callable:
        """Создание interpolation функции"""
        
        if params["model"] == "polynomial":
            def poly_interpolator(k):
                if np.isscalar(k):
                    k = np.array([k])
                log_k = np.log(k / strikes[0])  # Normalize
                return params["a"] + params["b"] * log_k + params["c"] * log_k**2
            return poly_interpolator
        
        else:
            # Fallback: simple interpolation
            interpolator = interpolate.interp1d(
                strikes, ivs, 
                kind='cubic', 
                fill_value='extrapolate'
            )
            
            def safe_interpolator(k):
                if np.isscalar(k):
                    return float(interpolator(k))
                return interpolator(k)
            
            return safe_interpolator

    def _calculate_skew(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        underlying_price: float
    ) -> float:
        """Расчет volatility skew"""
        
        # Find strikes близкие к ATM
        atm_idx = np.argmin(np.abs(strikes - underlying_price))
        
        if len(strikes) < 3:
            return 0.0
        
        # Gradient around ATM
        if atm_idx > 0 and atm_idx < len(strikes) - 1:
            left_iv = ivs[atm_idx - 1]
            right_iv = ivs[atm_idx + 1]
            left_k = strikes[atm_idx - 1]
            right_k = strikes[atm_idx + 1]
            
            if right_k != left_k:
                skew = (right_iv - left_iv) / (right_k - left_k)
                return skew * underlying_price  # Normalize by underlying price
        
        # Fallback: overall gradient
        if len(strikes) > 1:
            gradient = np.gradient(ivs, strikes)
            return np.mean(gradient) * underlying_price
        
        return 0.0

    def _calculate_kurtosis(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        underlying_price: float
    ) -> float:
        """Расчет volatility kurtosis (convexity)"""
        
        if len(strikes) < 3:
            return 0.0
        
        # Second derivative (convexity)
        second_deriv = np.gradient(np.gradient(ivs, strikes), strikes)
        
        # Weight by distance from ATM
        weights = 1 / (1 + np.abs(strikes - underlying_price) / underlying_price)
        weighted_convexity = np.average(second_deriv, weights=weights)
        
        return weighted_convexity * (underlying_price ** 2)

    async def _calculate_risk_neutral_moments(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        underlying_price: float,
        time_to_expiry: float
    ) -> Tuple[float, float]:
        """Расчет risk-neutral skewness и kurtosis"""
        
        if len(strikes) < 5:
            return 0.0, 3.0  # Default normal distribution moments
        
        try:
            # Создаем более плотную сетку страйков для интеграции
            k_min, k_max = strikes.min(), strikes.max()
            k_dense = np.linspace(k_min, k_max, 100)
            
            # Interpolate IV
            iv_interp = interpolate.interp1d(strikes, ivs, kind='cubic', fill_value='extrapolate')
            iv_dense = iv_interp(k_dense)
            
            # Call prices для всех страйков
            call_prices = []
            for k, iv in zip(k_dense, iv_dense):
                price = black_scholes_price(
                    underlying_price, k, time_to_expiry, 0.0, iv, OptionType.CALL
                )
                call_prices.append(price)
            
            call_prices = np.array(call_prices)
            
            # Numerical derivatives для risk-neutral density
            dk = k_dense[1] - k_dense[0]
            first_deriv = np.gradient(call_prices, dk)
            second_deriv = np.gradient(first_deriv, dk)
            
            # Risk-neutral density: q(K) = exp(r*T) * d²C/dK²
            risk_neutral_density = second_deriv  # Simplified (r=0)
            
            # Нормализация density
            density_sum = np.sum(risk_neutral_density) * dk
            if density_sum > 0:
                risk_neutral_density = risk_neutral_density / density_sum
            
            # Центральные моменты
            mean_k = np.sum(k_dense * risk_neutral_density) * dk
            
            # Скорректированные моменты
            centered_k = k_dense - mean_k
            variance = np.sum(centered_k**2 * risk_neutral_density) * dk
            
            if variance > 0:
                third_moment = np.sum(centered_k**3 * risk_neutral_density) * dk
                fourth_moment = np.sum(centered_k**4 * risk_neutral_density) * dk
                
                skewness = third_moment / (variance ** 1.5)
                kurtosis = fourth_moment / (variance ** 2)
                
                return float(skewness), float(kurtosis)
        
        except Exception as e:
            logger.debug(f"Risk-neutral moments calculation failed: {e}")
        
        return 0.0, 3.0  # Default values

class CryptoVolatilityIndexCalculator:
    """
    Crypto Volatility Index Calculator (VIX-style)
    
    Рассчитывает volatility index на базе options prices,
    аналогично VIX для stock markets. Для криптовалют адаптирован
    под специфику рынка и доступность опционных данных.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.calculation_history: List[CryptoVolatilityIndex] = []
        
        logger.info(f"🎯 Crypto VIX calculator initialized for {symbol}")

    async def calculate_volatility_index(
        self,
        options_data: List[OptionData],
        target_days: List[int] = [30],
        risk_free_rate: float = 0.0
    ) -> CryptoVolatilityIndex:
        """
        Расчет crypto volatility index
        
        Args:
            options_data: Список всех доступных опционов
            target_days: Целевые сроки (дни до экспирации)
            risk_free_rate: Безрисковая ставка
        """
        try:
            logger.info(f"🔄 Calculating volatility index for {self.symbol}...")
            
            if len(options_data) < 5:
                raise ValueError("Insufficient options data for VIX calculation")
            
            # Группировка по экспирациям
            expiry_groups = {}
            for option in options_data:
                expiry_key = option.expiry.date()
                if expiry_key not in expiry_groups:
                    expiry_groups[expiry_key] = []
                expiry_groups[expiry_key].append(option)
            
            # Расчет для каждой экспирации
            expiry_volatilities = {}
            
            for expiry_date, expiry_options in expiry_groups.items():
                if len(expiry_options) < 3:
                    continue
                
                try:
                    vol = await self._calculate_model_free_volatility(expiry_options)
                    days_to_expiry = (expiry_date - datetime.now().date()).days
                    expiry_volatilities[days_to_expiry] = vol
                except:
                    continue
            
            if len(expiry_volatilities) == 0:
                raise ValueError("No valid expiry volatilities calculated")
            
            # Интерполяция к целевым срокам
            term_structure = {}
            
            for target_days_val in target_days:
                if target_days_val in expiry_volatilities:
                    term_structure[f"{target_days_val}D"] = expiry_volatilities[target_days_val]
                else:
                    # Интерполяция
                    days_list = sorted(expiry_volatilities.keys())
                    vols_list = [expiry_volatilities[d] for d in days_list]
                    
                    if len(days_list) >= 2:
                        interpolated = np.interp(target_days_val, days_list, vols_list)
                        term_structure[f"{target_days_val}D"] = interpolated
            
            # Основной index (30-дневный)
            main_vol_index = term_structure.get("30D", list(expiry_volatilities.values())[0])
            
            # Определение режима волатильности
            regime = self._determine_volatility_regime(main_vol_index)
            
            # Near/next term (если доступны)
            sorted_days = sorted(expiry_volatilities.keys())
            near_term_vol = expiry_volatilities[sorted_days[0]] if sorted_days else main_vol_index
            next_term_vol = expiry_volatilities[sorted_days[1]] if len(sorted_days) > 1 else near_term_vol
            
            # Раздельные расчеты для calls/puts
            call_vol, put_vol = await self._calculate_call_put_indices(options_data)
            
            # Качество данных
            data_quality = self._assess_data_quality(options_data)
            
            # Исторические перцентили
            percentiles = await self._calculate_historical_percentiles(main_vol_index)
            
            crypto_vix = CryptoVolatilityIndex(
                symbol=self.symbol,
                timestamp=datetime.now(),
                volatility_index=main_vol_index,
                regime=regime,
                near_term_vol=near_term_vol,
                next_term_vol=next_term_vol,
                term_structure=term_structure,
                call_vol_index=call_vol,
                put_vol_index=put_vol,
                data_quality_score=data_quality,
                n_options_used=len(options_data),
                percentile_1m=percentiles.get("1m", 50.0),
                percentile_3m=percentiles.get("3m", 50.0),
                percentile_1y=percentiles.get("1y", 50.0),
                metadata={
                    "calculation_method": "model_free",
                    "n_expiries": len(expiry_groups),
                    "risk_free_rate": risk_free_rate
                }
            )
            
            self.calculation_history.append(crypto_vix)
            logger.info(f"✅ Crypto VIX calculated: {main_vol_index:.4f}")
            
            return crypto_vix
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility index: {e}")
            raise

    async def _calculate_model_free_volatility(
        self,
        expiry_options: List[OptionData]
    ) -> float:
        """
        Model-free implied volatility calculation
        
        Использует весь спектр страйков для расчета volatility
        без предположений о конкретной модели ценообразования.
        """
        
        if len(expiry_options) < 3:
            raise ValueError("Insufficient options for model-free calculation")
        
        # Группировка по страйкам
        strike_data = {}
        underlying_price = expiry_options[0].underlying_price
        time_to_expiry = expiry_options[0].time_to_expiry
        
        for option in expiry_options:
            strike = option.strike
            if strike not in strike_data:
                strike_data[strike] = {"call": None, "put": None}
            
            if option.option_type == OptionType.CALL:
                strike_data[strike]["call"] = option
            else:
                strike_data[strike]["put"] = option
        
        # Forward price estimation (put-call parity)
        forward_estimates = []
        for strike, options in strike_data.items():
            call_opt = options["call"] 
            put_opt = options["put"]
            
            if call_opt and put_opt:
                # F = K + exp(r*T) * (C - P)
                forward = strike + np.exp(0.0 * time_to_expiry) * (call_opt.price - put_opt.price)
                forward_estimates.append(forward)
        
        if forward_estimates:
            forward_price = np.median(forward_estimates)
        else:
            forward_price = underlying_price
        
        # Выбор ATM страйка
        strikes = list(strike_data.keys())
        atm_strike = min(strikes, key=lambda k: abs(k - forward_price))
        
        # Model-free variance calculation
        variance_contribution = 0.0
        total_weight = 0.0
        
        sorted_strikes = sorted(strikes)
        
        for i, strike in enumerate(sorted_strikes):
            options = strike_data[strike]
            
            # Выбор опциона (call или put в зависимости от moneyness)
            if strike < atm_strike:
                # OTM puts
                option = options["put"]
                if not option:
                    continue
            else:
                # ATM и OTM calls
                option = options["call"] 
                if not option:
                    option = options["put"]  # Fallback
                if not option:
                    continue
            
            # Delta K (spacing between strikes)
            if i == 0:
                delta_k = sorted_strikes[1] - strike if len(sorted_strikes) > 1 else strike * 0.1
            elif i == len(sorted_strikes) - 1:
                delta_k = strike - sorted_strikes[i-1]
            else:
                delta_k = (sorted_strikes[i+1] - sorted_strikes[i-1]) / 2
            
            # Contribution: (2*exp(r*T) / T) * (ΔK/K²) * Q(K)
            contribution = (2 / time_to_expiry) * (delta_k / (strike**2)) * option.price
            variance_contribution += contribution
            total_weight += delta_k / (strike**2)
        
        if total_weight == 0:
            raise ValueError("No valid contributions to model-free variance")
        
        # Центральный member adjustment
        central_term = (forward_price / atm_strike - 1)**2 / time_to_expiry
        
        model_free_variance = variance_contribution - central_term
        model_free_volatility = np.sqrt(max(0, model_free_variance))
        
        return model_free_volatility

    def _determine_volatility_regime(self, vol_index: float) -> VolatilityRegime:
        """Определение режима волатильности"""
        
        # Пороги для криптовалют (выше чем для traditional assets)
        if vol_index < 0.3:  # 30%
            return VolatilityRegime.LOW
        elif vol_index < 0.6:  # 60%
            return VolatilityRegime.NORMAL
        elif vol_index < 1.0:  # 100%
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    async def _calculate_call_put_indices(
        self,
        options_data: List[OptionData]
    ) -> Tuple[float, float]:
        """Раздельные indices для calls и puts"""
        
        calls = [opt for opt in options_data if opt.option_type == OptionType.CALL]
        puts = [opt for opt in options_data if opt.option_type == OptionType.PUT]
        
        try:
            call_vol = await self._calculate_model_free_volatility(calls) if len(calls) >= 3 else 0.0
        except:
            call_vol = 0.0
        
        try:
            put_vol = await self._calculate_model_free_volatility(puts) if len(puts) >= 3 else 0.0
        except:
            put_vol = 0.0
        
        return call_vol, put_vol

    def _assess_data_quality(self, options_data: List[OptionData]) -> float:
        """Оценка качества опционных данных"""
        
        quality_scores = []
        
        # 1. Coverage (покрытие страйков)
        strikes = [opt.strike for opt in options_data]
        underlying_price = options_data[0].underlying_price
        
        strike_range = max(strikes) - min(strikes)
        coverage_score = min(1.0, strike_range / underlying_price)  # Нормализация
        quality_scores.append(coverage_score)
        
        # 2. Liquidity (volume/OI если доступны)
        liquidity_scores = []
        for opt in options_data:
            if opt.volume is not None and opt.open_interest is not None:
                liquidity = (opt.volume + opt.open_interest) / 2
                liquidity_scores.append(min(1.0, liquidity / 100))  # Normalize to max 100
        
        if liquidity_scores:
            avg_liquidity = np.mean(liquidity_scores)
            quality_scores.append(avg_liquidity)
        
        # 3. Bid-Ask spread quality
        spread_scores = []
        for opt in options_data:
            if opt.bid is not None and opt.ask is not None and opt.ask > opt.bid:
                spread = (opt.ask - opt.bid) / ((opt.ask + opt.bid) / 2)
                spread_score = max(0, 1 - spread)  # Lower spread = better quality
                spread_scores.append(spread_score)
        
        if spread_scores:
            avg_spread_quality = np.mean(spread_scores)
            quality_scores.append(avg_spread_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.5

    async def _calculate_historical_percentiles(
        self,
        current_vol: float
    ) -> Dict[str, float]:
        """Расчет исторических перцентилей"""
        
        # В реальной реализации здесь будет обращение к базе данных
        # Для демонстрации возвращаем mock значения
        
        historical_vols = [vol.volatility_index for vol in self.calculation_history[-252:]]  # Last year
        
        if len(historical_vols) < 10:
            return {"1m": 50.0, "3m": 50.0, "1y": 50.0}
        
        # Перцентили
        percentiles = {}
        
        # 1 месяц
        recent_1m = [vol.volatility_index for vol in self.calculation_history[-30:]]
        if recent_1m:
            percentiles["1m"] = (np.sum(np.array(recent_1m) < current_vol) / len(recent_1m)) * 100
        else:
            percentiles["1m"] = 50.0
        
        # 3 месяца
        recent_3m = [vol.volatility_index for vol in self.calculation_history[-90:]]
        if recent_3m:
            percentiles["3m"] = (np.sum(np.array(recent_3m) < current_vol) / len(recent_3m)) * 100
        else:
            percentiles["3m"] = 50.0
        
        # 1 год
        if historical_vols:
            percentiles["1y"] = (np.sum(np.array(historical_vols) < current_vol) / len(historical_vols)) * 100
        else:
            percentiles["1y"] = 50.0
        
        return percentiles

    def generate_vix_report(self, crypto_vix: CryptoVolatilityIndex) -> str:
        """Генерация отчета по Crypto VIX"""
        
        report = f"""
🎯 Crypto Volatility Index Report for {self.symbol}
{'='*60}

Current VIX Level: {crypto_vix.volatility_index:.2%}
Volatility Regime: {crypto_vix.regime.value.upper()}

Term Structure:
{chr(10).join([f"  {term}: {vol:.2%}" for term, vol in crypto_vix.term_structure.items()])}

Market Sentiment:
- Call Volatility:  {crypto_vix.call_vol_index:.2%}
- Put Volatility:   {crypto_vix.put_vol_index:.2%}
- Put/Call Ratio:   {crypto_vix.put_vol_index/crypto_vix.call_vol_index:.2f}

Historical Context:
- 1 Month Percentile:  {crypto_vix.percentile_1m:.0f}%
- 3 Month Percentile:  {crypto_vix.percentile_3m:.0f}%
- 1 Year Percentile:   {crypto_vix.percentile_1y:.0f}%

Data Quality: {crypto_vix.data_quality_score:.1%}
Options Used: {crypto_vix.n_options_used}

Interpretation:
{self._interpret_vix_level(crypto_vix)}
        """
        
        return report

    def _interpret_vix_level(self, crypto_vix: CryptoVolatilityIndex) -> str:
        """Интерпретация уровня VIX"""
        
        level = crypto_vix.volatility_index
        
        if level < 0.3:
            return "- Low volatility environment\n- Potential for volatility expansion\n- Consider long volatility strategies"
        elif level < 0.6:
            return "- Normal volatility levels\n- Balanced risk environment\n- Standard position sizing appropriate"
        elif level < 1.0:
            return "- Elevated volatility\n- Higher risk environment\n- Reduce position sizes, increase hedging"
        else:
            return "- Extreme volatility\n- Crisis-level fear\n- Consider defensive positioning"

# Export всех классов
__all__ = [
    "OptionType",
    "VolatilityRegime",
    "OptionData",
    "ImpliedVolatilityResult",
    "VolatilitySmile",
    "CryptoVolatilityIndex",
    "BaseImpliedVolatilityExtractor", 
    "ImpliedVolatilityExtractor",
    "VolatilitySmileModel",
    "CryptoVolatilityIndexCalculator",
    "black_scholes_price",
    "black_scholes_vega",
    "black_scholes_gamma",
    "black_scholes_theta"
]

logger.info("🔥 Implied Volatility module loaded successfully!")