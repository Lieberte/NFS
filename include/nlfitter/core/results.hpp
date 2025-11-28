#pragma once

#include "types.hpp"
#include <vector>
#include <chrono>

namespace nlfitter {

class FitResult {
public:
    FitResult() = default;
    
    void set_parameters(const ParameterVector& params) {
        parameters_ = params;
    }
    
    void set_status(ConvergenceStatus status) {
        status_ = status;
    }
    
    void set_num_iterations(int n) {
        num_iterations_ = n;
    }
    
    void set_num_function_evaluations(int n) {
        num_function_evaluations_ = n;
    }
    
    void set_num_jacobian_evaluations(int n) {
        num_jacobian_evaluations_ = n;
    }
    
    void set_final_residual_norm(double norm) {
        final_residual_norm_ = norm;
    }
    
    void set_initial_residual_norm(double norm) {
        initial_residual_norm_ = norm;
    }
    
    void set_elapsed_time(double seconds) {
        elapsed_time_ = seconds;
    }
    
    void set_covariance(const Matrix& cov) {
        covariance_ = cov;
        has_covariance_ = true;
    }
    
    void add_iteration_info(const IterationInfo& info) {
        iteration_history_.push_back(info);
    }
    
    const ParameterVector& parameters() const { return parameters_; }
    ConvergenceStatus status() const { return status_; }
    int num_iterations() const { return num_iterations_; }
    int num_function_evaluations() const { return num_function_evaluations_; }
    int num_jacobian_evaluations() const { return num_jacobian_evaluations_; }
    double final_residual_norm() const { return final_residual_norm_; }
    double initial_residual_norm() const { return initial_residual_norm_; }
    double elapsed_time() const { return elapsed_time_; }
    
    bool success() const { return is_converged(status_); }
    
    bool has_covariance() const { return has_covariance_; }
    const Matrix& covariance() const { return covariance_; }
    
    Vector parameter_errors() const {
        if (!has_covariance_) {
            return Vector::Zero(parameters_.size());
        }
        return covariance_.diagonal().cwiseSqrt();
    }
    
    const std::vector<IterationInfo>& iteration_history() const {
        return iteration_history_;
    }
    
    double relative_residual_reduction() const {
        if (initial_residual_norm_ == 0.0) return 0.0;
        return (initial_residual_norm_ - final_residual_norm_) / initial_residual_norm_;
    }
    
    std::string summary() const {
        std::ostringstream oss;
        oss << "Fit Result Summary:\n";
        oss << "  Status: " << to_string(status_) << "\n";
        oss << "  Success: " << (success() ? "Yes" : "No") << "\n";
        oss << "  Iterations: " << num_iterations_ << "\n";
        oss << "  Function evaluations: " << num_function_evaluations_ << "\n";
        oss << "  Jacobian evaluations: " << num_jacobian_evaluations_ << "\n";
        oss << "  Initial residual: " << initial_residual_norm_ << "\n";
        oss << "  Final residual: " << final_residual_norm_ << "\n";
        oss << "  Relative reduction: " << (relative_residual_reduction() * 100) << "%\n";
        oss << "  Elapsed time: " << elapsed_time_ << " seconds\n";
        oss << "  Parameters: [";
        for (int i = 0; i < parameters_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << parameters_[i];
        }
        oss << "]\n";
        if (has_covariance_) {
            Vector errors = parameter_errors();
            oss << "  Errors: [";
            for (int i = 0; i < errors.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << errors[i];
            }
            oss << "]\n";
        }
        return oss.str();
    }
    
private:
    ParameterVector parameters_;
    ConvergenceStatus status_ = ConvergenceStatus::NotStarted;
    int num_iterations_ = 0;
    int num_function_evaluations_ = 0;
    int num_jacobian_evaluations_ = 0;
    double final_residual_norm_ = 0.0;
    double initial_residual_norm_ = 0.0;
    double elapsed_time_ = 0.0;
    Matrix covariance_;
    bool has_covariance_ = false;
    std::vector<IterationInfo> iteration_history_;
};

} // namespace nlfitter