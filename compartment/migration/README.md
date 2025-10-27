# Migration Directory

This directory contains code currently being migrated from the private Pandemic Simulator monorepo to form a package usable both locally and within the Pandemic Simulator app.

## Directory Structure

- **`batch_helpers/`** - Utilities and helpers for interacting with AWS services in the pandemic simulator project. These dependencies will be removed from packages and made available as extensions.
- **`compartment/`** - The current core components of the compartment model in a flat implementation. These will be refactored to a more hierarchical organization which promotes faster local use.
