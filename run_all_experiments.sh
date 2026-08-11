#!/bin/bash

################################################################################
# GRAFT - Run All Experiments
# 
# This script runs all three experiments sequentially:
#   1. Baseline Comparison (Experiment_1.py)
#   2. Ablation Study (Experiment_2.py)
#   3. Noise Robustness (Experiment_3.py)
#
# Each experiment's output is logged to a separate file.
################################################################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to check if required datasets exist
check_datasets() {
    print_info "Checking for required datasets..."
    
    missing_datasets=()
    
    if [ ! -f "flchain_final.csv" ]; then
        missing_datasets+=("flchain_final.csv")
    fi
    
    if [ ! -f "nwtco.csv" ]; then
        missing_datasets+=("nwtco.csv")
    fi
    
    if [ ${#missing_datasets[@]} -ne 0 ]; then
        print_error "Missing required datasets:"
        for dataset in "${missing_datasets[@]}"; do
            echo "  - $dataset"
        done
        echo ""
        print_info "Please download from: https://vincentarelbundock.github.io/Rdatasets/articles/data.html"
        echo ""
        echo "Quick download commands:"
        echo "  wget -O flchain_final.csv https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv"
        echo "  wget -O nwtco.csv https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv"
        echo ""
        print_info "Note: aids.csv is already included in the repository"
        echo ""
        return 1
    fi
    
    print_success "All required datasets found"
    return 0
}

# Function to check if Python and required packages are available
check_requirements() {
    print_info "Checking Python environment..."
    
    # Check Python version
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8 or higher."
        return 1
    fi
    
    python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $python_version found"
    
    # Check key packages
    print_info "Checking required packages..."
    
    packages=("torch" "numpy" "pandas" "lifelines" "pycox" "sklearn")
    missing_packages=()
    
    for package in "${packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        print_error "Missing required packages:"
        for package in "${missing_packages[@]}"; do
            echo "  - $package"
        done
        echo ""
        print_info "Please install missing packages. See README.md for details."
        return 1
    fi
    
    print_success "All required packages found"
    
    # Check GPU availability
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        print_success "GPU detected: $gpu_name"
        print_info "Estimated total runtime: ~3-4 hours"
    else
        print_warning "No GPU detected. Training will use CPU."
        print_info "Estimated total runtime: ~12-18 hours"
    fi
    
    return 0
}

# Function to run an experiment with error handling
run_experiment() {
    local exp_name=$1
    local exp_file=$2
    local log_file=$3
    
    print_header "$exp_name"
    print_info "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    print_info "Output will be logged to: $log_file"
    echo ""
    
    start_time=$(date +%s)
    
    # Run the experiment and capture output
    if python "$exp_file" 2>&1 | tee "$log_file"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$((duration % 60))
        
        echo ""
        print_success "$exp_name completed successfully!"
        print_info "Duration: ${hours}h ${minutes}m ${seconds}s"
        print_info "Results saved to: $log_file"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$((duration % 60))
        
        echo ""
        print_error "$exp_name failed!"
        print_info "Duration before failure: ${hours}h ${minutes}m ${seconds}s"
        print_info "Check log file for details: $log_file"
        return 1
    fi
}

# Main execution
main() {
    clear
    
    print_header "GRAFT: Running All Experiments"
    echo ""
    print_info "This script will run 3 experiments sequentially:"
    echo "  1. Experiment 1: Baseline Comparison (~60-90 min with GPU)"
    echo "  2. Experiment 2: Ablation Study (~60-90 min with GPU)"
    echo "  3. Experiment 3: Noise Robustness (~90-120 min with GPU)"
    echo ""
    echo ""
    
    # Pre-flight checks
    if ! check_requirements; then
        print_error "Requirements check failed. Please fix the issues above."
        exit 1
    fi
    
    echo ""
    
    if ! check_datasets; then
        print_error "Dataset check failed. Please download missing datasets."
        exit 1
    fi
    
    echo ""
    print_warning "Press Ctrl+C within 5 seconds to cancel..."
    sleep 5
    
    echo ""
    
    # Track overall start time
    overall_start=$(date +%s)
    
    # Track experiment results
    exp1_success=false
    exp2_success=false
    exp3_success=false
    
    # Run Experiment 1
    if run_experiment "Experiment 1: Baseline Comparison" "Experiment_1.py" "experiment_1_output.log"; then
        exp1_success=true
    else
        print_error "Stopping execution due to failure in Experiment 1"
        exit 1
    fi
    
    echo ""
    echo ""
    
    # Run Experiment 2
    if run_experiment "Experiment 2: Ablation Study" "Experiment_2.py" "experiment_2_output.log"; then
        exp2_success=true
    else
        print_error "Stopping execution due to failure in Experiment 2"
        exit 1
    fi
    
    echo ""
    echo ""
    
    # Run Experiment 3
    if run_experiment "Experiment 3: Noise Robustness" "Experiment_3.py" "experiment_3_output.log"; then
        exp3_success=true
    else
        print_error "Stopping execution due to failure in Experiment 3"
        exit 1
    fi
    
    # Calculate overall duration
    overall_end=$(date +%s)
    overall_duration=$((overall_end - overall_start))
    overall_hours=$((overall_duration / 3600))
    overall_minutes=$(((overall_duration % 3600) / 60))
    overall_seconds=$((overall_duration % 60))
    
    # Print summary
    echo ""
    echo ""
    print_header "EXECUTION SUMMARY"
    echo ""
    print_info "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    print_info "Total duration: ${overall_hours}h ${overall_minutes}m ${overall_seconds}s"
    echo ""
    
    if $exp1_success; then
        print_success "Experiment 1: Baseline Comparison - SUCCESS"
    else
        print_error "Experiment 1: Baseline Comparison - FAILED"
    fi
    
    if $exp2_success; then
        print_success "Experiment 2: Ablation Study - SUCCESS"
    else
        print_error "Experiment 2: Ablation Study - FAILED"
    fi
    
    if $exp3_success; then
        print_success "Experiment 3: Noise Robustness - SUCCESS"
    else
        print_error "Experiment 3: Noise Robustness - FAILED"
    fi
    
    echo ""
    print_info "Log files:"
    echo "  - experiment_1_output.log"
    echo "  - experiment_2_output.log"
    echo "  - experiment_3_output.log"
    
    echo ""
    print_info "Generated plots:"
    if [ -f "ablation_cindex_summary.png" ]; then
        echo "  ✓ ablation_cindex_summary.png"
    else
        echo "  ✗ ablation_cindex_summary.png (missing)"
    fi
    
    if [ -f "ablation_ibs_summary.png" ]; then
        echo "  ✓ ablation_ibs_summary.png"
    else
        echo "  ✗ ablation_ibs_summary.png (missing)"
    fi
    
    if [ -f "noise_robustness_cindex_summary.png" ]; then
        echo "  ✓ noise_robustness_cindex_summary.png"
    else
        echo "  ✗ noise_robustness_cindex_summary.png (missing)"
    fi
    
    if [ -f "noise_robustness_ibs_summary.png" ]; then
        echo "  ✓ noise_robustness_ibs_summary.png"
    else
        echo "  ✗ noise_robustness_ibs_summary.png (missing)"
    fi
    
    echo ""
    
    if $exp1_success && $exp2_success && $exp3_success; then
        print_success "All experiments completed successfully!"
        echo ""
        print_info "Next steps:"
        echo "  1. Review the log files for detailed results"
        exit 0
    else
        print_error "Some experiments failed. Check the logs for details."
        exit 1
    fi
}

# Run main function
main
