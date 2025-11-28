```
nonlinear-fitter/
├── CMakeLists.txt # 主构建文件
├── README.md # 项目说明
├── LICENSE # 开源协议
├── .gitignore # Git 忽略文件
├── .clang-format # 代码格式化配置
│
├── include/ # 公共头文件
│ └── nlfitter/
│ ├── core/
│ │ ├── types.hpp # 基础类型定义
│ │ ├── dataset.hpp # 数据集
│ │ ├── model.hpp # 模型接口
│ │ └── result.hpp # 拟合结果
│ │
│ ├── solvers/
│ │ ├── solver_base.hpp # 求解器基类
│ │ ├── levenberg_marquardt.hpp
│ │ ├── trust_region.hpp
│ │ └── gauss_newton.hpp
│ │
│ ├── optimizers/
│ │ ├── line_search.hpp
│ │ ├── trust_region_policy.hpp
│ │ └── regularization.hpp
│ │
│ ├── jacobian/
│ │ ├── numerical_diff.hpp
│ │ ├── autodiff.hpp # 自动微分
│ │ └── sparse_jacobian.hpp
│ │
│ ├── loss/
│ │ ├── loss_function.hpp
│ │ ├── squared_loss.hpp
│ │ ├── huber_loss.hpp
│ │ └── robust_losses.hpp
│ │
│ ├── constraints/
│ │ ├── bounds.hpp
│ │ └── general_constraints.hpp
│ │
│ ├── analysis/
│ │ ├── uncertainty.hpp # 不确定度分析
│ │ ├── diagnostics.hpp # 诊断工具
│ │ └── cross_validation.hpp
│ │
│ └── nlfitter.hpp # 主入口头文件
│
├── src/ # 实现文件
│ ├── core/
│ │ ├── dataset.cpp
│ │ └── model.cpp
│ │
│ ├── solvers/
│ │ ├── levenberg_marquardt.cpp
│ │ ├── trust_region.cpp
│ │ └── gauss_newton.cpp
│ │
│ ├── jacobian/
│ │ ├── numerical_diff.cpp
│ │ └── autodiff.cpp
│ │
│ ├── loss/
│ │ └── robust_losses.cpp
│ │
│ └── analysis/
│ ├── uncertainty.cpp
│ └── diagnostics.cpp
│
├── examples/ # 示例代码
│ ├── CMakeLists.txt
│ ├── basic_fitting.cpp # 基础拟合示例
│ ├── polynomial_fit.cpp # 多项式拟合
│ ├── exponential_fit.cpp # 指数拟合
│ ├── custom_model.cpp # 自定义模型
│ ├── robust_fitting.cpp # 鲁棒拟合
│ ├── constrained_fit.cpp # 约束优化
│ └── uncertainty_analysis.cpp # 不确定度分析
│
├── tests/ # 单元测试
│ ├── CMakeLists.txt
│ ├── test_main.cpp
│ ├── core/
│ │ ├── test_dataset.cpp
│ │ └── test_model.cpp
│ ├── solvers/
│ │ ├── test_lm.cpp
│ │ └── test_trust_region.cpp
│ └── jacobian/
│ ├── test_numerical_diff.cpp
│ └── test_autodiff.cpp
│
├── benchmarks/ # 性能基准测试
│ ├── CMakeLists.txt
│ ├── bench_jacobian.cpp
│ └── bench_solvers.cpp
│
├── docs/ # 文档
│ ├── api/ # API 文档
│ ├── tutorials/ # 教程
│ │ ├── 01_getting_started.md
│ │ ├── 02_custom_models.md
│ │ └── 03_advanced_topics.md
│ └── theory/ # 理论说明
│ ├── algorithms.md
│ └── numerical_methods.md
│
├── third_party/ # 第三方依赖
│ └── CMakeLists.txt
│
└── scripts/ # 辅助脚本
├── format.sh # 代码格式化
├── build.sh # 构建脚本
└── test.sh # 测试脚本
```