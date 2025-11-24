---  
layout: post
title: "Streamlined multi-node serving with Ray symmetric-run"
author: "Richard Liaw (Anyscale/Ray), Kaichao You (vLLM)"
image: /assets/logos/vllm-logo-text-light.png
---


Ray now has a new command: `ray symmetric-run`. This command makes it possible to launch the **same entrypoint command** on every node in a Ray cluster, simplifying the workflow to spawn vLLM servers with multi-node models on HPC setups or when using parallel ssh tools like `mpssh`.

In this blog, we’ll talk about the issues with the current way of spawning vLLM servers with Ray; we’ll walk through a motivating example; and finally we’ll discuss how the new `symmetric-run` API from Ray will improve the launching experience.

Ray recently joined the Pytorch foundation. As part of this donation, both vLLM and Ray teams are working together to build deep alignment to advance the next generation of AI infrastructure.

<p align="center">
<picture>
<img src="/assets/figures/2025-11-25-ray-symmetric-run/symmetric-run.png" width="100%">
</picture><br>
Figure 1. Overview of `symmetric-run` command from Ray.
</p>

### Context

vLLM users approaching Ray often bring expectations from their existing tools and workflows. Developers doing interactive work on barebones clusters expect to quickly launch commands across multiple hosts with tools like `mpssh` or `pssh`, using a single command parameterized by “rank”. HPC users familiar with SLURM and PBS expect *symmetric execution* — a single program entrypoint that runs simultaneously across all nodes, similar to MPI applications.

However, Ray's recommended job execution pattern follows a different philosophy with specific roles for head and worker nodes.

The program entrypoint executes on the head node, which then orchestrates and delegates work to worker nodes. The runtime lifecycle is separate from job execution, requiring explicit cluster management. This means users need two distinct command sets—one to establish the cluster with proper head/worker roles, and another to actually run their work.

### Motivating Example

Let’s walk through an example of launching a distributed job on 2 separate machines. We assume the machines are bare in that we can’t use other solutions like Ray’s cluster launcher or KubeRay.

To run a Ray job on multiple machines in this situation, users will need to first start Ray on the head node:

```shell
ray start --block
```

And then on worker nodes, they’ll need to connect back to the head node:

```shell
# worker node, terminal 1:
ray start --block --address='ip:6379'
```

After the nodes are setup, then they’ll need to start a separate terminal on the head node to run the job:

```shell
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

And then in termination, they’ll need to run `ray stop` on each of the nodes:

```shell
ray stop
```

Running vLLM in this configuration typically requires a fair amount of trial and error. A common failure mode occurs when certain environment variables, like `VLLM_HOST_IP`, are missing. When this happens, users will need to shut down the Ray cluster, set the environment variable on \`ray start\`, and go through the rest of the above steps yet again.

For users expecting symmetric, single-command execution, this creates significant friction in their development and deployment workflows.

Ray now offers a simple solution: **`ray symmetric-run`**, a new way to run Ray jobs across all nodes in a cluster.

### What does `symmetric-run` do?

`ray symmetric-run` makes it possible to launch the **same entrypoint command** on every node in a Ray cluster. The script automatically handles Ray setup, job execution, and teardown, allowing users can have a similar experience to other tools like `mpirun` or `torchrun`.

Taking the same example as above, with `symmetric-run`, you can simply do:

```shell
# in SLURM sbatch script or via mpssh

ray symmetric-run \
  --address <head_node_address>:6379 \
  --min-nodes 2 \
  --num-gpus 8 \
  -- vllm serve Qwen/Qwen3-32B --tensor-parallel-size 8 --pipeline-parallel-size 2
```

Each node will execute the same command, but the behavior underneath the hood will be different per node. In particular, the worker node will only execute the Ray cluster initialization, whereas the head node will:

1. Starts Ray in `--head` mode.
2. Waits for four nodes to register.
3. Run your user command (`vllm serve Qwen/Qwen3-32B …`).
4. Shuts down Ray when done.

The workers simply run `ray start --address head-node:6379` and wait until the job ends, after which it will self-destruct. No extra SSH orchestration or startup scripts needed.

If you need to provide an environment variable, you can simply append it to the start of the command, and `symmetric-run` will automatically propagate the variable to the Ray runtime:

```shell
ENV=VAR ray symmetric-run --address 127.0.0.1:6379 -- python test.py
```

### Conclusion

Ray’s new symmetric run utility simplifies running Ray and vLLM programs on HPC or parallel SSH setups. Try out Ray Symmetric Run today: [https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)

If you run into any issues, please open a ticket on Github: [https://github.com/ray-project/ray/](https://github.com/ray-project/ray/)

To chat with other members of the community, join the [vLLM slack](https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack) and [Ray slack](https://www.ray.io/join-slack)\!
