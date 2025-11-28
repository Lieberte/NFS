#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <functional>
#include <memory>
#include <string>

namespace nlfitter {

using Scalar = double;
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<double>;

using ParameterVector = Vector;
using FeatureVector = Vector;
using ObservationVector = Vector;
using ResidualVector = Vector;
using JacobianMatrix = Matrix;
using SparseJacobian = SparseMatrix;
using WeightMatrix = Vector;

using ModelFunction = std::function<ObservationVector(
    const FeatureVector& x, 
    const ParameterVector& params)>;

using JacobianFunction = std::function<JacobianMatrix(
    const FeatureVector& x,
    const ParameterVector& params)>;

using ResidualFunction = std::function<ResidualVector(
    const FeatureVector& x,
    const ObservationVector& y,
    const ParameterVector& params)>;

struct ConvergenceCriteria {
    double parameter_tolerance = 1e-8;
    double gradient_tolerance = 1e-8;
    double residual_tolerance = 1e-8;
    double residual_absolute_tolerance = 1e-10;
    int max_iterations = 100;
    int max_function_evaluations = 1000;
    bool verbose = false;
    int verbose_frequency = 1;
};

struct SolverConfig {
    ConvergenceCriteria convergence;
    
    double finite_diff_step = 1e-7;
    
    enum class DiffScheme {
        Forward,
        Central
    } diff_scheme = DiffScheme::Central;
    
    double regularization = 0.0;
    
    enum class RegularizationType {
        None,
        L2,
        L1,
        Elastic
    } regularization_type = RegularizationType::None;
    
    std::unique_ptr<class LossFunction> loss_function;
    
    bool use_parallel = true;
    int num_threads = -1;
    
    double initial_trust_radius = 1.0;
    double max_trust_radius = 1e10;
    double line_search_backtrack_factor = 0.5;
    int max_line_search_iterations = 20;
    
    double initial_damping = 1e-3;
    double damping_increase_factor = 10.0;
    double damping_decrease_factor = 0.1;
    double min_damping = 1e-15;
    double max_damping = 1e15;
    
    bool use_sparse = false;
    std::shared_ptr<SparseMatrix> sparsity_pattern;
};

struct IterationInfo {
    int iteration = 0;
    double residual_norm = 0.0;
    double previous_residual_norm = 0.0;
    double residual_change = 0.0;
    double parameter_change = 0.0;
    double gradient_norm = 0.0;
    double step_norm = 0.0;
    double damping_parameter = 0.0;
    double trust_radius = 0.0;
    double gain_ratio = 0.0;
    int num_function_evaluations = 0;
    int num_jacobian_evaluations = 0;
    double elapsed_time = 0.0;
};

enum class ConvergenceStatus {
    NotStarted,
    Running,
    ParameterConverged,
    GradientConverged,
    ResidualConverged,
    AbsoluteResidualConverged,
    MaxIterations,
    MaxFunctionEvaluations,
    NumericalError,
    SingularJacobian,
    UserStopped,
    Failed
};

inline const char* to_string(ConvergenceStatus status) {
    switch (status) {
        case ConvergenceStatus::NotStarted: return "Not Started";
        case ConvergenceStatus::Running: return "Running";
        case ConvergenceStatus::ParameterConverged: return "Converged (Parameter change < tolerance)";
        case ConvergenceStatus::GradientConverged: return "Converged (Gradient norm < tolerance)";
        case ConvergenceStatus::ResidualConverged: return "Converged (Residual change < tolerance)";
        case ConvergenceStatus::AbsoluteResidualConverged: return "Converged (Absolute residual < tolerance)";
        case ConvergenceStatus::MaxIterations: return "Stopped (Maximum iterations reached)";
        case ConvergenceStatus::MaxFunctionEvaluations: return "Stopped (Maximum function evaluations reached)";
        case ConvergenceStatus::NumericalError: return "Failed (Numerical error: NaN or Inf detected)";
        case ConvergenceStatus::SingularJacobian: return "Failed (Singular Jacobian matrix)";
        case ConvergenceStatus::UserStopped: return "Stopped by user";
        case ConvergenceStatus::Failed: return "Failed (Unknown reason)";
        default: return "Unknown Status";
    }
}

inline bool is_converged(ConvergenceStatus status) {
    return status == ConvergenceStatus::ParameterConverged ||
           status == ConvergenceStatus::GradientConverged ||
           status == ConvergenceStatus::ResidualConverged ||
           status == ConvergenceStatus::AbsoluteResidualConverged;
}

inline bool is_error(ConvergenceStatus status) {
    return status == ConvergenceStatus::NumericalError ||
           status == ConvergenceStatus::SingularJacobian ||
           status == ConvergenceStatus::Failed;
}

using IterationCallback = std::function<bool(const IterationInfo& info)>;

} // namespace nlfitter