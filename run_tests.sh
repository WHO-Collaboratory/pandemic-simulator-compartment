#!/bin/bash
# Convenience script to run tests with common options

set -e

echo "================================"
echo "Running Pandemic Simulator Tests"
echo "================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Parse command line arguments
FAST=false
COVERAGE=false
VERBOSE=false
PATTERN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fast        Skip slow tests"
            echo "  --coverage    Generate coverage report"
            echo "  --verbose     Verbose output"
            echo "  --pattern     Run tests matching pattern (e.g., 'ABC' or 'test_models.py')"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh --fast             # Run only fast tests"
            echo "  ./run_tests.sh --coverage         # Run with coverage report"
            echo "  ./run_tests.sh --pattern ABC      # Run ABC model tests"
            echo "  ./run_tests.sh -v --coverage      # Verbose with coverage"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$FAST" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
    echo "Running fast tests only..."
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=compartment --cov-report=html --cov-report=term"
    echo "Running with coverage..."
fi

if [ -n "$PATTERN" ]; then
    PYTEST_CMD="$PYTEST_CMD -k $PATTERN"
    echo "Running tests matching: $PATTERN"
fi

# Run tests
echo ""
echo "Command: $PYTEST_CMD"
echo ""

$PYTEST_CMD

# Show coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "================================"
    echo "Coverage report generated:"
    echo "  HTML: htmlcov/index.html"
    echo "================================"
fi
