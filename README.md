# Multinode Finetuning POC using Kubeflow

## Overview

This repository demonstrates a **Proof-of-Concept (POC)** for multinode, multi-GPU fine-tuning using **Kubeflow Training Operator** and **PyTorch Distributed (torchrun)**.

The goal is to validate distributed training infrastructure rather than optimize model performance.

---

## Scope

### ✅ What this POC covers

- Kubernetes orchestration for **multi-node, multi-GPU fine-tuning**
- Job configuration using:
  - TRL (Transformer Reinforcement Learning)
  - HuggingFace Transformers
  - Kubeflow PyTorchJob
- Validation setup:
  - Model: `meta-llama/Llama-3.2-1B`
  - Dataset: `tatsu-lab/alpaca`
  - Cluster: **2 nodes × 2 GPUs per node**

---

### ❌ What this POC does NOT cover

- ML optimization:
  - Hyperparameter tuning
  - Checkpointing
  - Metrics tracking
- State management:
  - Suspend / Resume / Restart
  - Gang scheduling
- Fault tolerance

---

## Project Structure

```
examples/multinode-sft-trainer/
│
├── train.py
├── entrypoint.sh
├── Dockerfile
├── manifests/
│   ├── pytorchjob.yaml
│   └── hf-secret.yaml
```

---

## Configurable Parameters

- `--model_name`
- `--dataset_name`
- `--epochs`
- `GPUS_PER_NODE`
- `WORKER_REPLICAS`

---

## Setup

```bash
kubectl apply -k github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1
```

```bash
export IMAGE="your-docker-image"
export GPUS_PER_NODE="2"
export WORKER_REPLICAS="1"

envsubst < manifests/pytorchjob.yaml | kubectl apply -f -
```

---

## Validation

```bash
kubectl get pytorchjob
kubectl get pods
```

---

## Notes

- Multinode training incurs communication overhead.
- NCCL handles inter-node and intra-node communication.

---

## License

MIT
