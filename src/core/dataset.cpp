#include "nlfitter/core/dataset.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace nlfitter {
namespace core {

// ============================================================================
// Dataset 实现
// ============================================================================

Dataset::Dataset()
    : m_num_points(0)
    , m_has_weights(false) {
}

Dataset::Dataset(const VectorXd& x, const VectorXd& y)
    : m_x(x)
    , m_y(y)
    , m_num_points(x.size())
    , m_has_weights(false) {
    
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    
    if (x.size() == 0) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
}

Dataset::Dataset(const VectorXd& x, const VectorXd& y, const VectorXd& weights)
    : m_x(x)
    , m_y(y)s
    , m_weights(weights)
    , m_num_points(x.size())
    , m_has_weights(true) {
    
    if (x.size() != y.size() || x.size() != weights.size()) {
        throw std::invalid_argument("x, y, and weights must have the same size");
    }
    
    if (x.size() == 0) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
    
    // 检查权重是否都为正
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights(i) <= 0.0) {
            throw std::invalid_argument("All weights must be positive");
        }
    }
}

Dataset::Dataset(const MatrixXd& X, const VectorXd& y)
    : m_X(X)
    , m_y(y)
    , m_num_points(X.rows())
    , m_has_weights(false)
    , m_is_multivariate(true) {
    
    if (X.rows() != y.size()) {
        throw std::invalid_argument("Number of rows in X must match size of y");
    }
    
    if (X.rows() == 0) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
}

Dataset::Dataset(const MatrixXd& X, const VectorXd& y, const VectorXd& weights)
    : m_X(X)
    , m_y(y)
    , m_weights(weights)
    , m_num_points(X.rows())
    , m_has_weights(true)
    , m_is_multivariate(true) {
    
    if (X.rows() != y.size() || X.rows() != weights.size()) {
        throw std::invalid_argument("Dimensions mismatch");
    }
    
    if (X.rows() == 0) {
        throw std::invalid_argument("Dataset cannot be empty");
    }
    
    // 检查权重
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights(i) <= 0.0) {
            throw std::invalid_argument("All weights must be positive");
        }
    }
}

void Dataset::setWeights(const VectorXd& weights) {
    if (weights.size() != m_num_points) {
        throw std::invalid_argument("Weights size mismatch");
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights(i) <= 0.0) {
            throw std::invalid_argument("All weights must be positive");
        }
    }
    
    m_weights = weights;
    m_has_weights = true;
}

const VectorXd& Dataset::getX() const {
    if (m_is_multivariate) {
        throw std::runtime_error("Use getXMatrix() for multivariate data");
    }
    return m_x;
}

const VectorXd& Dataset::getY() const {
    return m_y;
}

const MatrixXd& Dataset::getXMatrix() const {
    if (!m_is_multivariate) {
        throw std::runtime_error("Use getX() for univariate data");
    }
    return m_X;
}

const VectorXd& Dataset::getWeights() const {
    if (!m_has_weights) {
        throw std::runtime_error("Dataset does not have weights");
    }
    return m_weights;
}

bool Dataset::hasWeights() const {
    return m_has_weights;
}

bool Dataset::isMultivariate() const {
    return m_is_multivariate;
}

size_t Dataset::size() const {
    return m_num_points;
}

size_t Dataset::numFeatures() const {
    if (m_is_multivariate) {
        return m_X.cols();
    }
    return 1;
}

double Dataset::x(size_t i) const {
    if (i >= m_num_points) {
        throw std::out_of_range("Index out of range");
    }
    if (m_is_multivariate) {
        throw std::runtime_error("Use xVector() for multivariate data");
    }
    return m_x(i);
}

double Dataset::y(size_t i) const {
    if (i >= m_num_points) {
        throw std::out_of_range("Index out of range");
    }
    return m_y(i);
}

VectorXd Dataset::xVector(size_t i) const {
    if (i >= m_num_points) {
        throw std::out_of_range("Index out of range");
    }
    if (!m_is_multivariate) {
        // 单变量情况，返回一维向量
        VectorXd result(1);
        result(0) = m_x(i);
        return result;
    }
    return m_X.row(i);
}

double Dataset::weight(size_t i) const {
    if (i >= m_num_points) {
        throw std::out_of_range("Index out of range");
    }
    if (!m_has_weights) {
        return 1.0;  // 默认权重为 1
    }
    return m_weights(i);
}

void Dataset::normalize() {
    // 归一化 y 值到 [0, 1] 范围
    double min_y = m_y.minCoeff();
    double max_y = m_y.maxCoeff();
    
    if (std::abs(max_y - min_y) < 1e-15) {
        throw std::runtime_error("Cannot normalize: all y values are the same");
    }
    
    m_y = (m_y.array() - min_y) / (max_y - min_y);
    
    // 保存归一化参数
    m_normalization_params.y_min = min_y;
    m_normalization_params.y_max = max_y;
    m_normalization_params.is_normalized = true;
}

void Dataset::standardize() {
    // 标准化 y 值到均值为 0，标准差为 1
    double mean_y = m_y.mean();
    double std_y = std::sqrt((m_y.array() - mean_y).square().mean());
    
    if (std_y < 1e-15) {
        throw std::runtime_error("Cannot standardize: y has zero variance");
    }
    
    m_y = (m_y.array() - mean_y) / std_y;
    
    // 保存标准化参数
    m_normalization_params.y_mean = mean_y;
    m_normalization_params.y_std = std_y;
    m_normalization_params.is_standardized = true;
}

Dataset Dataset::subset(size_t start, size_t end) const {
    if (start >= m_num_points || end > m_num_points || start >= end) {
        throw std::out_of_range("Invalid subset range");
    }
    
    const size_t n = end - start;
    
    if (m_is_multivariate) {
        MatrixXd X_sub = m_X.block(start, 0, n, m_X.cols());
        VectorXd y_sub = m_y.segment(start, n);
        
        if (m_has_weights) {
            VectorXd w_sub = m_weights.segment(start, n);
            return Dataset(X_sub, y_sub, w_sub);
        }
        return Dataset(X_sub, y_sub);
    } else {
        VectorXd x_sub = m_x.segment(start, n);
        VectorXd y_sub = m_y.segment(start, n);
        
        if (m_has_weights) {
            VectorXd w_sub = m_weights.segment(start, n);
            return Dataset(x_sub, y_sub, w_sub);
        }
        return Dataset(x_sub, y_sub);
    }
}

std::pair<Dataset, Dataset> Dataset::trainTestSplit(double test_ratio, unsigned int seed) const {
    if (test_ratio <= 0.0 || test_ratio >= 1.0) {
        throw std::invalid_argument("test_ratio must be in (0, 1)");
    }
    
    const size_t test_size = static_cast<size_t>(m_num_points * test_ratio);
    const size_t train_size = m_num_points - test_size;
    
    // 生成随机索引
    std::vector<size_t> indices(m_num_points);
    for (size_t i = 0; i < m_num_points; ++i) {
        indices[i] = i;
    }
    
    // 使用指定种子打乱索引
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // 创建训练集和测试集
    std::vector<size_t> train_idx(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> test_idx(indices.begin() + train_size, indices.end());
    
    return {createSubsetFromIndices(train_idx), createSubsetFromIndices(test_idx)};
}

Dataset Dataset::createSubsetFromIndices(const std::vector<size_t>& indices) const {
    const size_t n = indices.size();
    
    if (m_is_multivariate) {
        MatrixXd X_sub(n, m_X.cols());
        VectorXd y_sub(n);
        VectorXd w_sub(n);
        
        for (size_t i = 0; i < n; ++i) {
            X_sub.row(i) = m_X.row(indices[i]);
            y_sub(i) = m_y(indices[i]);
            if (m_has_weights) {
                w_sub(i) = m_weights(indices[i]);
            }
        }
        
        if (m_has_weights) {
            return Dataset(X_sub, y_sub, w_sub);
        }
        return Dataset(X_sub, y_sub);
    } else {
        VectorXd x_sub(n);
        VectorXd y_sub(n);
        VectorXd w_sub(n);
        
        for (size_t i = 0; i < n; ++i) {
            x_sub(i) = m_x(indices[i]);
            y_sub(i) = m_y(indices[i]);
            if (m_has_weights) {
                w_sub(i) = m_weights(indices[i]);
            }
        }
        
        if (m_has_weights) {
            return Dataset(x_sub, y_sub, w_sub);
        }
        return Dataset(x_sub, y_sub);
    }
}

DatasetStats Dataset::computeStats() const {
    DatasetStats stats;
    
    // y 的统计量
    stats.y_mean = m_y.mean();
    stats.y_std = std::sqrt((m_y.array() - stats.y_mean).square().mean());
    stats.y_min = m_y.minCoeff();
    stats.y_max = m_y.maxCoeff();
    
    // x 的统计量（单变量情况）
    if (!m_is_multivariate) {
        stats.x_mean = VectorXd(1);
        stats.x_std = VectorXd(1);
        stats.x_min = VectorXd(1);
        stats.x_max = VectorXd(1);
        
        stats.x_mean(0) = m_x.mean();
        stats.x_std(0) = std::sqrt((m_x.array() - stats.x_mean(0)).square().mean());
        stats.x_min(0) = m_x.minCoeff();
        stats.x_max(0) = m_x.maxCoeff();
    } else {
        // 多变量情况
        const size_t n_features = m_X.cols();
        stats.x_mean = VectorXd(n_features);
        stats.x_std = VectorXd(n_features);
        stats.x_min = VectorXd(n_features);
        stats.x_max = VectorXd(n_features);
        
        for (size_t j = 0; j < n_features; ++j) {
            VectorXd col = m_X.col(j);
            stats.x_mean(j) = col.mean();
            stats.x_std(j) = std::sqrt((col.array() - stats.x_mean(j)).square().mean());
            stats.x_min(j) = col.minCoeff();
            stats.x_max(j) = col.maxCoeff();
        }
    }
    
    return stats;
}

// ============================================================================
// DatasetStats 实现
// ============================================================================

void DatasetStats::print(std::ostream& os) const {
    os << "=== Dataset Statistics ===" << std::endl;
    os << "Y statistics:" << std::endl;
    os << "  Mean: " << y_mean << std::endl;
    os << "  Std:  " << y_std << std::endl;
    os << "  Min:  " << y_min << std::endl;
    os << "  Max:  " << y_max << std::endl;
    
    os << "X statistics:" << std::endl;
    for (size_t i = 0; i < x_mean.size(); ++i) {
        os << "  Feature " << i << ":" << std::endl;
        os << "    Mean: " << x_mean(i) << std::endl;
        os << "    Std:  " << x_std(i) << std::endl;
        os << "    Min:  " << x_min(i) << std::endl;
        os << "    Max:  " << x_max(i) << std::endl;
    }
}

} // namespace core
} // namespace nlfitter