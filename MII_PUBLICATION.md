# MII model publication (EventBridge)

This repository’s **only** publication-side responsibility toward the PanSim web application is to emit a single EventBridge event after the shared-services artifact pipeline succeeds (S3 object present and shared-services DynamoDB mapping written). Downstream behavior—per-environment provisioning, validation, rollback, DLQ handling, and admin enable/disable—is implemented entirely in AWS (see MII-PUB-003 design in platform requirements).

## Workflow

The `disease-pipeline` workflow, on semver tags in `WHO-Collaboratory/pandemic-simulator-compartment`, runs in order: smoke tests → container image push to ECR → artifact generation, S3 upload, shared-services DDB write → **`events:PutEvents`** to the shared-services custom bus.

## GitHub configuration

| Secret | Purpose |
|--------|---------|
| `MODEL_PUBLICATION_EVENT_BUS_NAME` | Name of the shared-services EventBridge **custom** event bus (same bus the provisioner rules target). |

The GitHub OIDC role used by Actions must already allow `events:PutEvents` on that bus (platform IAM).

## Event shape

Events use `source` `mii.publication`, `detail-type` `ModelVersionPublished`, and `detail` JSON containing `name`, `semver`, `image_uri`, `artifact_hash`, and `artifact_s3_key`, as defined in the MII-PUB-003 design document.
