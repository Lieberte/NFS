#include "nlfitter/core/model.hpp"
#include <stdexcept>
#include <sstream>

namespace nlfitter {
namespace core {

// ============================================================================
// Model 实现
// ============================================================================

Model::Model(size_t num_params)
    : m_num_params(num_params)
    , m_has_analytical_jacobian(false) {
    
    if (num_params == 0) {
        throw std::invalid_argument("Number of parameters must be positive");
    }
}

Model::~Model() = default;

size_t Model::numParameters() const {
    return m_num_params;
}

bool Model::hasAnalyticalJacobian() const {
    return m_has_analytical_jacobian;
}

VectorXd Model::jacobian(const VectorXd& x, const VectorXd& params) const {
    throw std::runtime_error("Analytical Jacobian not implemented for this model");
}

MatrixXd Model::jacobianMatrix(const Dataset& data, const VectorXd& params) const {
    const size_t n = data.size();
    MatrixXd J(n, m_num_params);
    
    for (size_t i = 0; i < n; ++i) {
        if (data.isMultivariate()) {
            J.row(i) = jacobian(data.xVector(i), params);
        } else {
            VectorXd x(1);
            x(0) = data.x(i);
            J.row(i) = jacobian(x, params);
        }
    }
    
    return J;
}

VectorXd Model::residuals(const Dataset& data, const VectorXd& params) const {
    const size_t n = data.size();
    VectorXd r(n);
    
    for (size_t i = 0; i < n; ++i) {
        double y_pred;
        if (data.isMultivariate()) {
            y_pred = evaluate(data.xVector(i), params);
        } else {
            VectorXd x(1);
            x(0) = data.x(i);
            y_pred = evaluate(x, params);
        }
        
        double residual = data.y(i) - y_pred;
        
        // 应用权重（如果有）
        if (data.hasWeights()) {
            residual *= std::sqrt(data.weight(i));
        }
        
        r(i) = residual;
    }
    
    return r;
}

VectorXd Model::predict(const Dataset& data, const VectorXd& params) const {
    const size_t n = data.size();
    VectorXd predictions(n);
    
    for (size_t i = 0; i < n; ++i) {
        if (data.isMultivariate()) {
            predictions(i) = evaluate(data.xVector(i), params);
        } else {
            VectorXd x(1);
            x(0) = data.x(i);
            predictions(i) = evaluate(x, params);
        }
    }
    
    return predictions;
}

void Model::validateParameters(const VectorXd& params) const {
    if (params.size() != m_num_params) {
        std::ostringstream oss;
        oss << "Parameter size mismatch: expected " << m_num_params 
            << ", got " << params.size();
        throw std::invalid_argument(oss.str());
    }
}

std::vector<std::string> Model::parameterNames() const {
    // 默认参数名
    std::vector<std::string> names(m_num_params);
    for (size_t i = 0; i < m_num_params; ++i) {
        names[i] = "p" + std::to_string(i);
    }
    return names;
}

std::string Model::description() const {
    return "Generic model with " + std::to_string(m_num_params) + " parameters";
}

// ============================================================================
// PolynomialModel 实现
// ============================================================================

PolynomialModel::PolynomialModel(size_t degree)
    : Model(degree + 1)
    , m_degree(degree) {
    
    m_has_analytical_jacobian = true;
}

double PolynomialModel::evaluate(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Polynomial model requires univariate input");
    }
    
    const double xi = x(0);
    double result = 0.0;
    double x_power = 1.0;
    
    // y = p0 + p1*x + p2*x^2 + ... + pn*x^n
    for (size_t i = 0; i <= m_degree; ++i) {
        result += params(i) * x_power;
        x_power *= xi;
    }
    
    return result;
}

VectorXd PolynomialModel::jacobian(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Polynomial model requires univariate input");
    }
    
    const double xi = x(0);
    VectorXd jac(m_num_params);
    double x_power = 1.0;
    
    // ∂y/∂pi = x^i
    for (size_t i = 0; i <= m_degree; ++i) {
        jac(i) = x_power;
        x_power *= xi;
    }
    
    return jac;
}

std::vector<std::string> PolynomialModel::parameterNames() const {
    std::vector<std::string> names(m_num_params);
    for (size_t i = 0; i <= m_degree; ++i) {
        names[i] = "a" + std::to_string(i);
    }
    return names;
}

std::string PolynomialModel::description() const {
    return "Polynomial model of degree " + std::to_string(m_degree);
}

// ============================================================================
// ExponentialModel 实现
// ============================================================================

ExponentialModel::ExponentialModel()
    : Model(3) {
    
    m_has_analytical_jacobian = true;
}

double ExponentialModel::evaluate(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Exponential model requires univariate input");
    }
    
    const double xi = x(0);
    const double A = params(0);
    const double k = params(1);
    const double c = params(2);
    
    // y = A * exp(k*x) + c
    return A * std::exp(k * xi) + c;
}

VectorXd ExponentialModel::jacobian(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Exponential model requires univariate input");
    }
    
    const double xi = x(0);
    const double A = params(0);
    const double k = params(1);
    
    VectorXd jac(3);
    
    const double exp_kx = std::exp(k * xi);
    
    // ∂y/∂A = exp(k*x)
    jac(0) = exp_kx;
    
    // ∂y/∂k = A*x*exp(k*x)
    jac(1) = A * xi * exp_kx;
    
    // ∂y/∂c = 1
    jac(2) = 1.0;
    
    return jac;
}

std::vector<std::string> ExponentialModel::parameterNames() const {
    return {"A", "k", "c"};
}

std::string ExponentialModel::description() const {
    return "Exponential model: y = A*exp(k*x) + c";
}

// ============================================================================
// GaussianModel 实现
// ============================================================================

GaussianModel::GaussianModel()
    : Model(3) {
    
    m_has_analytical_jacobian = true;
}

double GaussianModel::evaluate(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Gaussian model requires univariate input");
    }
    
    const double xi = x(0);
    const double A = params(0);      // 幅度
    const double mu = params(1);     // 中心
    const double sigma = params(2);  // 标准差
    
    // y = A * exp(-(x-mu)^2 / (2*sigma^2))
    const double z = (xi - mu) / sigma;
    return A * std::exp(-0.5 * z * z);
}

VectorXd GaussianModel::jacobian(const VectorXd& x, const VectorXd& params) const {
    validateParameters(params);
    
    if (x.size() != 1) {
        throw std::invalid_argument("Gaussian model requires univariate input");
    }
    
    const double xi = x(0);
    const double A = params(0);
    const double mu = params(1);
    const double sigma = params(2);
    
    VectorXd jac(3);
    
    const double z = (xi - mu) / sigma;
    const double exp_term = std::exp(-0.5 * z * z);
    
    // ∂y/∂A = exp(-(x-mu)^2 / (2*sigma^2))
    jac(0) = exp_term;
    
    // ∂y/∂mu = A * (x-mu)/sigma^2 * exp(...)
    jac(1) = A * z / sigma * exp_term;
    
    // ∂y/∂sigma = A * (x-mu)^2/sigma^3 * exp(...)
    jac(2) = A * z * z / sigma * exp_term;
    
    return jac;
}

std::vector<std::string> GaussianModel::parameterNames() const {
    return {"A", "mu", "sigma"};
}

std::string GaussianModel::description() const {
    return "Gaussian model: y = A*exp(-(x-mu)^2/(2*sigma^2))";
}

} // namespace core
} // namespace nlfitter