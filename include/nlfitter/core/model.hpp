#pragma once

#include "types.hpp"
#include "dataset.hpp"
#include <string>
#include <stdexcept>

namespace nlfitter {

class Model {
public:
    virtual ~Model() = default;
    
    virtual std::string name() const = 0;
    
    virtual int num_parameters() const = 0;
    
    virtual ObservationVector evaluate(const FeatureVector& x, const ParameterVector& params) const = 0;
    
    virtual JacobianMatrix jacobian(const FeatureVector& x, const ParameterVector& params) const {
        return compute_jacobian_numerical(x, params);
    }
    
    virtual bool has_analytical_jacobian() const { return false; }
    
    virtual ParameterVector get_initial_parameters() const {
        return ParameterVector::Zero(num_parameters());
    }
    
    virtual void validate_parameters(const ParameterVector& params) const {
        if (params.size() != num_parameters()) {
            throw std::invalid_argument("Parameter size mismatch");
        }
    }
    
    ResidualVector compute_residual(const FeatureVector& x, 
                                   const ObservationVector& y,
                                   const ParameterVector& params) const {
        return y - evaluate(x, params);
    }
    
    ResidualVector compute_residuals(const Dataset& data, const ParameterVector& params) const {
        int total_size = data.size() * data.observation_dimension();
        ResidualVector residuals(total_size);
        
        for (size_t i = 0; i < data.size(); ++i) {
            ResidualVector r = compute_residual(data.x(i), data.y(i), params);
            residuals.segment(i * data.observation_dimension(), data.observation_dimension()) = r;
        }
        
        return residuals;
    }
    
    JacobianMatrix compute_jacobian_numerical(const FeatureVector& x, 
                                             const ParameterVector& params,
                                             double h = 1e-7) const {
        int n_params = params.size();
        ObservationVector y0 = evaluate(x, params);
        int n_outputs = y0.size();
        
        JacobianMatrix J(n_outputs, n_params);
        ParameterVector params_perturbed = params;
        
        for (int j = 0; j < n_params; ++j) {
            double h_j = h * std::max(1.0, std::abs(params[j]));
            
            params_perturbed[j] = params[j] + h_j;
            ObservationVector y_plus = evaluate(x, params_perturbed);
            
            params_perturbed[j] = params[j] - h_j;
            ObservationVector y_minus = evaluate(x, params_perturbed);
            
            J.col(j) = (y_plus - y_minus) / (2.0 * h_j);
            
            params_perturbed[j] = params[j];
        }
        
        return J;
    }
    
    JacobianMatrix compute_full_jacobian(const Dataset& data, 
                                        const ParameterVector& params) const {
        int total_size = data.size() * data.observation_dimension();
        int n_params = num_parameters();
        
        JacobianMatrix J(total_size, n_params);
        
        for (size_t i = 0; i < data.size(); ++i) {
            JacobianMatrix J_i = jacobian(data.x(i), params);
            J.block(i * data.observation_dimension(), 0, 
                   data.observation_dimension(), n_params) = J_i;
        }
        
        return J;
    }
    
protected:
    Model() = default;
};

} // namespace nlfitter