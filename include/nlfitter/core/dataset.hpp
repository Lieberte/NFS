#pragma once

#include "types.hpp"
#include <vector>
#include <stdexcept>

namespace nlfitter {

class Dataset {
public:
    Dataset() = default;
    
    Dataset(const std::vector<FeatureVector>& x, 
            const std::vector<ObservationVector>& y)
        : x_data_(x), y_data_(y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("X and Y must have same size");
        }
        if (x.empty()) {
            throw std::invalid_argument("Dataset cannot be empty");
        }
        feature_dim_ = x[0].size();
        observation_dim_ = y[0].size();
        
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i].size() != feature_dim_) {
                throw std::invalid_argument("All feature vectors must have same dimension");
            }
            if (y[i].size() != observation_dim_) {
                throw std::invalid_argument("All observation vectors must have same dimension");
            }
        }
    }
    
    Dataset(const Matrix& X, const Matrix& Y) {
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument("X and Y must have same number of rows");
        }
        if (X.rows() == 0) {
            throw std::invalid_argument("Dataset cannot be empty");
        }
        
        feature_dim_ = X.cols();
        observation_dim_ = Y.cols();
        
        x_data_.resize(X.rows());
        y_data_.resize(Y.rows());
        
        for (int i = 0; i < X.rows(); ++i) {
            x_data_[i] = X.row(i);
            y_data_[i] = Y.row(i);
        }
    }
    
    void add_sample(const FeatureVector& x, const ObservationVector& y) {
        if (x_data_.empty()) {
            feature_dim_ = x.size();
            observation_dim_ = y.size();
        } else {
            if (x.size() != feature_dim_) {
                throw std::invalid_argument("Feature dimension mismatch");
            }
            if (y.size() != observation_dim_) {
                throw std::invalid_argument("Observation dimension mismatch");
            }
        }
        x_data_.push_back(x);
        y_data_.push_back(y);
    }
    
    void set_weights(const std::vector<double>& weights) {
        if (weights.size() != size()) {
            throw std::invalid_argument("Weights size must match dataset size");
        }
        weights_ = Vector(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            weights_[i] = weights[i];
        }
    }
    
    void set_weights(const Vector& weights) {
        if (weights.size() != static_cast<int>(size())) {
            throw std::invalid_argument("Weights size must match dataset size");
        }
        weights_ = weights;
    }
    
    size_t size() const { return x_data_.size(); }
    int feature_dimension() const { return feature_dim_; }
    int observation_dimension() const { return observation_dim_; }
    
    const FeatureVector& x(size_t i) const { return x_data_[i]; }
    const ObservationVector& y(size_t i) const { return y_data_[i]; }
    
    const std::vector<FeatureVector>& x_data() const { return x_data_; }
    const std::vector<ObservationVector>& y_data() const { return y_data_; }
    
    bool has_weights() const { return weights_.size() > 0; }
    const Vector& weights() const { return weights_; }
    
    Matrix get_X_matrix() const {
        Matrix X(x_data_.size(), feature_dim_);
        for (size_t i = 0; i < x_data_.size(); ++i) {
            X.row(i) = x_data_[i];
        }
        return X;
    }
    
    Matrix get_Y_matrix() const {
        Matrix Y(y_data_.size(), observation_dim_);
        for (size_t i = 0; i < y_data_.size(); ++i) {
            Y.row(i) = y_data_[i];
        }
        return Y;
    }
    
    void clear() {
        x_data_.clear();
        y_data_.clear();
        weights_.resize(0);
        feature_dim_ = 0;
        observation_dim_ = 0;
    }
    
private:
    std::vector<FeatureVector> x_data_;
    std::vector<ObservationVector> y_data_;
    Vector weights_;
    int feature_dim_ = 0;
    int observation_dim_ = 0;
};

} // namespace nlfitter