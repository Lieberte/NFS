#ifndef NLFITTER_JACOBIAN_NUMERICAL_DIFF_HPP
#define NLFITTER_JACOBIAN_NUMERICAL_DIFF_HPP

#include "nlfitter/core/types.hpp"
#include "nlfitter/core/dataset.hpp"
#include "nlfitter/core/model.hpp"
#include <memory>

namespace nlfitter {
namespace jacobian {

/**
 * @brief 数值微分方法
 */
enum class DiffMethod {
    Forward,     // 前向差分 f'(x) ≈ (f(x+h) - f(x)) / h
    Backward,    // 后向差分 f'(x) ≈ (f(x) - f(x-h)) / h
    Central,     // 中心差分 f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    Complex      // 复步长法 f'(x) = Im(f(x+ih)) / h (最精确)
};

/**
 * @brief 数值微分配置
 */
struct NumericalDiffConfig {
    DiffMethod method = DiffMethod::Central;
    double step_size = 1e-8;           // 微分步长
    bool relative_step = true;         // 使用相对步长
    double min_step = 1e-12;           // 最小步长
    double max_step = 1e-4;            // 最大步长
    
    NumericalDiffConfig() = default;
};

/**
 * @brief 数值微分计算器
 */
class NumericalDiff {
public:
    /**
     * @brief 构造函数
     * @param model 模型指针
     * @param config 配置
     */
    NumericalDiff(std::shared_ptr<core::Model> model,
                  const NumericalDiffConfig& config = NumericalDiffConfig());
    
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
    
    /**
     * @brief 设置配置
     */
    void setConfig(const NumericalDiffConfig& config);
    
    /**
     * @brief 获取配置
     */
    const NumericalDiffConfig& getConfig() const { return m_config; }
    
private:
    /**
     * @brief 计算步长
     * @param param_value 参数值
     * @return 步长
     */
    double computeStepSize(double param_value) const;
    
    /**
     * @brief 前向差分
     */
    double forwardDifference(const core::VectorXd& x,
                            const core::VectorXd& params,
                            size_t param_idx,
                            double h) const;
    
    /**
     * @brief 后向差分
     */
    double backwardDifference(const core::VectorXd& x,
                             const core::VectorXd& params,
                             size_t param_idx,
                             double h) const;
    
    /**
     * @brief 中心差分
     */
    double centralDifference(const core::VectorXd& x,
                            const core::VectorXd& params,
                            size_t param_idx,
                            double h) const;
    
    /**
     * @brief 复步长法
     */
    double complexStepDifference(const core::VectorXd& x,
                                const core::VectorXd& params,
                                size_t param_idx,
                                double h) const;
    
    std::shared_ptr<core::Model> m_model;
    NumericalDiffConfig m_config;
};

} // namespace jacobian
} // namespace nlfitter

#endif // NLFITTER_JACOBIAN_NUMERICAL_DIFF_HPP