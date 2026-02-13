# doc-auto-pilot Directory Structure

This repository is organized into source layers, API interfaces, configuration, deployment assets, test suites, and operational docs.

## CMD command used to create directories

```cmd
mkdir src\common src\infrastructure\base src\infrastructure\gcp src\infrastructure\local src\infrastructure\database\models src\layer1_ingestion src\layer2_analysis src\layer3_orchestration\workflows src\layer3_orchestration\agents src\layer3_orchestration\tools src\layer3_orchestration\prompts src\layer4_delivery src\api\routes src\api\schemas config\schemas scripts\migrations\postgres tests\unit tests\integration tests\e2e docs\architecture docs\setup docs\runbooks deployments\docker deployments\kubernetes data logs
```

## Directory tree

```text
C:.
+---config
|   \---schemas
+---data
+---deployments
|   +---docker
|   \---kubernetes
+---docs
|   +---architecture
|   +---runbooks
|   \---setup
+---logs
+---scripts
|   \---migrations
|       \---postgres
+---src
|   +---api
|   |   +---routes
|   |   \---schemas
|   +---common
|   +---infrastructure
|   |   +---base
|   |   +---database
|   |   |   \---models
|   |   +---gcp
|   |   \---local
|   +---layer1_ingestion
|   +---layer2_analysis
|   +---layer3_orchestration
|   |   +---agents
|   |   +---prompts
|   |   +---tools
|   |   \---workflows
|   \---layer4_delivery
\---tests
    +---e2e
    +---integration
    \---unit
```
