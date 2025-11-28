#ifndef NLFITTER_JACOBIAN_AUTODIFF_HPP
#define NLFITTER_JACOBIAN_AUTODIFF_HPP

#include "nlfitter/core/types.hpp"
#include "nlfitter/core/dataset.hpp"
#include "nlfitter/core/model.hpp"
#include <memory>
#include <cmath>
#include <stdexcept>

namespace nlfitter {
namespace jacobian {

/**
 * @brief 双数（Dual Number）用于前向模式自动微分
 * 
 * 双数形式: a + b*ε, 其中 ε² = 0
 * 运算规则:
 *   (a + b*ε) + (c + d*ε) = (a+c) + (b+d)*ε
 *   (a + b*ε) * (c + d*ε) = (a*c) + (a*d + b*c)*ε
 *   f(a + b*ε) = f(a) + b*f'(a)*ε
 */
template<typename Real = double>
class Dual {
public:
    Real value;      // 函数值
    Real derivative; // 导数值
    
    // 构造函数
    Dual() : value(0), derivative(0) {}
    Dual(Real v) : value(v), derivative(0) {}
    Dual(Real v, Real d) : value(v), derivative(d) {}
    
    // 算术运算符
    Dual operator+(const Dual& other) const {
        return Dual(value + other.value, derivative + other.derivative);
    }
    
    Dual operator-(const Dual& other) const {
        return Dual(value - other.value, derivative - other.derivative);
    }
    
    Dual operator*(const Dual& other) const {
        return Dual(value * other.value,
                   value * other.derivative + derivative * other.value);
    }
    
    Dual operator/(const Dual& other) const {
        return Dual(value / other.value,
                   (derivative * other.value - value * other.derivative) / 
                   (other.value * other.value));
    }
    
    Dual operator+(Real scalar) const {
        return Dual(value + scalar, derivative);
    }
    
    Dual operator-(Real scalar) const {
        return Dual(value - scalar, derivative);
    }
    
    Dual operator*(Real scalar) const {
        return Dual(value * scalar, derivative * scalar);
    }
    
    Dual operator/(Real scalar) const {
        return Dual(value / scalar, derivative / scalar);
    }
    
    // 取负
    Dual operator-() const {
        return Dual(-value, -derivative);
    }
    
    // 复合赋值运算符
    Dual& operator+=(const Dual& other) {
        value += other.value;
        derivative += other.derivative;
        return *this;
    }
    
    Dual& operator-=(const Dual& other) {
        value -= other.value;
        derivative -= other.derivative;
        return *this;
    }
    
    Dual& operator*=(const Dual& other) {
        Real new_deriv = value * other.derivative + derivative * other.value;
        value *= other.value;
        derivative = new_deriv;
        return *this;
    }
    
    Dual& operator/=(const Dual& other) {
        Real new_deriv = (derivative * other.value - value * other.derivative) / 
                        (other.value * other.value);
        value /= other.value;
        derivative = new_deriv;
        return *this;
    }
};

// 标量与双数的运算
template<typename Real>
Dual<Real> operator+(Real scalar, const Dual<Real>& dual) {
    return Dual<Real>(scalar + dual.value, dual.derivative);
}

template<typename Real>
Dual<Real> operator-(Real scalar, const Dual<Real>& dual) {
    return Dual<Real>(scalar - dual.value, -dual.derivative);
}

template<typename Real>
Dual<Real> operator*(Real scalar, const Dual<Real>& dual) {
    return Dual<Real>(scalar * dual.value, scalar * dual.derivative);
}

template<typename Real>
Dual<Real> operator/(Real scalar, const Dual<Real>& dual) {
    return Dual<Real>(scalar / dual.value,
                     -scalar * dual.derivative / (dual.value * dual.value));
}

// 数学函数
template<typename Real>
Dual<Real> exp(const Dual<Real>& x) {
    Real exp_val = std::exp(x.value);
    return Dual<Real>(exp_val, x.derivative * exp_val);
}

template<typename Real>
Dual<Real> log(const Dual<Real>& x) {
    return Dual<Real>(std::log(x.value), x.derivative / x.value);
}

template<typename Real>
Dual<Real> sqrt(const Dual<Real>& x) {
    Real sqrt_val = std::sqrt(x.value);
    return Dual<Real>(sqrt_val, x.derivative / (2.0 * sqrt_val));
}

template<typename Real>
Dual<Real> pow(const Dual<Real>& x, Real exponent) {
    Real pow_val = std::pow(x.value, exponent);
    return Dual<Real>(pow_val, 
                     x.derivative * exponent * std::pow(x.value, exponent - 1));
}

template<typename Real>
Dual<Real> sin(const Dual<Real>& x) {
    return Dual<Real>(std::sin(x.value), x.derivative * std::cos(x.value));
}

template<typename Real>
Dual<Real> cos(const Dual<Real>& x) {
    return Dual<Real>(std::cos(x.value), -x.derivative * std::sin(x.value));
}

template<typename Real>
Dual<Real> tan(const Dual<Real>& x) {
    Real cos_val = std::cos(x.value);
    return Dual<Real>(std::tan(x.value), x.derivative / (cos_val * cos_val));
}

template<typename Real>
Dual<Real> abs(const Dual<Real>& x) {
    return Dual<Real>(std::abs(x.value), 
                     x.value >= 0 ? x.derivative : -x.derivative);
}

/**
 * @brief 支持自动微分的模型包装器
 * 
 * 用户需要提供一个模板函数，该函数可以接受 Dual 类型的参数
 */
template<typename FuncType>
class AutoDiffModel : public core::Model {
public:
    AutoDiffModel(FuncType func, size_t num_params)
        : core::Model(num_params)
        , m_func(func) {
        m_has_analytical_jacobian = true; // 自动微分提供精确导数
    }
    
    double evaluate(const core::VectorXd& x, const core::VectorXd& params) const override {
        // 将参数转换为普通 double 进行计算
        return m_func(x, params);
    }
    
    core::VectorXd jacobian(const core::VectorXd& x, 
                           const core::VectorXd& params) const override {
        const size_t n_params = params.size();
        core::VectorXd jac(n_params);
        
        // 对每个参数进行前向模式自动微分
        for (size_t i = 0; i < n_params; ++i) {
            // 创建双数参数向量
            std::vector<Dual<double>> dual_params(n_params);
            for (size_t j = 0; j < n_params; ++j) {
                if (i == j) {
                    // 对第 i 个参数，设置导数为 1
                    dual_params[j] = Dual<double>(params(j), 1.0);
                } else {
                    dual_params[j] = Dual<double>(params(j), 0.0);
                }
            }
            
            // 计算函数值及导数
            Dual<double> result = m_func(x, dual_params);
            jac(i) = result.derivative;
        }
        
        return jac;
    }
    
private:
    FuncType m_func;
};

/**
 * @brief 自动微分计算器
 * 
 * 通过双数自动微分计算雅可比矩阵
 */
class AutoDiff {
public:
    /**
     * @brief 构造函数
     * @param model 模型指针
     */
    explicit AutoDiff(std::shared_ptr<core::Model> model);
    
    /**
     * @brief 计算雅可比矩阵
     * @param data 数据集
     * @param params 参数
     * @return 雅可比矩阵 (n_data × n_params)
     */
    core::MatrixXd compute(const core::Dataset& data,
                          const core::VectorXd& params) const;
    
    /**
     * @brief 计算单点的雅可比向量
     * @param x 输入点
     * @param params 参数
     * @return 雅可比向量 (1 × n_params)
     */
    core::VectorXd computePoint(const core::VectorXd& x,
                               const core::VectorXd& params) const;
    
private:
    std::shared_ptr<core::Model> m_model;
};

/**
 * @brief 创建支持自动微分的模型
 * 
 * @param func 模型函数，接受 VectorXd 和 vector<Dual<double>> 参数
 * @param num_params 参数个数
 * @return 模型智能指针
 */
template<typename FuncType>
std::shared_ptr<core::Model> makeAutoDiffModel(FuncType func, size_t num_params) {
    return std::make_shared<AutoDiffModel<FuncType>>(func, num_params);
}

} // namespace jacobian
} // namespace nlfitter

#endif // NLFITTER_JACOBIAN_AUTODIFF_HPP