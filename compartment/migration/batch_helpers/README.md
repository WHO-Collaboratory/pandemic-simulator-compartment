# Batch Helpers

This directory contains utility modules and helper functions that support batch processing operations for epidemiological simulations.

## Purpose

The batch helpers provide common functionality for:
- **GraphQL Operations** (`gql.py`) - Querying simulation job data from GraphQL endpoints
- **S3 Operations** (`s3.py`) - Writing simulation results and data to AWS S3 storage
- **Simulation Helpers** (`simulation_helpers.py`) - Common utilities for batch simulation setup and logging
- **GraphQL Queries** (`graphql_queries.py`) - Predefined GraphQL query templates

## Key Components

- **Environment Configuration**: Utilities for reading simulation parameters from environment variables
- **Logging Setup**: Standardized logging configuration for AWS Batch environments
- **Data Persistence**: Helper functions for writing simulation outputs to cloud storage
- **API Integration**: GraphQL client utilities for fetching simulation job configurations

## Usage

These helpers are designed to reduce code duplication across batch processing programs and provide consistent interfaces for common operations like data retrieval, storage, and logging in cloud-based simulation environments.
