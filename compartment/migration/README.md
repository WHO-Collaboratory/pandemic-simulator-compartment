# Migration Directory

This directory contains code that is currently being migrated from a separate private project monorepo into this open repository dedicated to a compartment modelling package which can be used both locally and through the pandemic simulator app.

## Directory Structure

- **`batch_helpers/`** - Utilities and helpers for interacting with AWS services in the pandemic simulator project. These dependencies will be removed from packages and made available as extensions, as the app will still need many of them.
- **`compartment/`** - The current core components of the compartment model in a flat implementation. These stand to be refactored to a more hierarchical organization which should promote faster local use.

## Note

This code represents the current state during the migration process and may be subject to significant changes as it is integrated into the new framework architecture.
