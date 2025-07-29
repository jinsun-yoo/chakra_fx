# Chakra_FX: Generating Distributed ML Workload "Representation" from PyTorch Source Code, with only 1 GPU

This project aims to answer the question: ASTRA-sim, or many other tools can take in a Chakra graph and do many stuff (simulation, etc) with it. *How do I (easily) get these Chakra graphs (without having to obtain hundreds of GPUs and running the workload on the GPUs)?*

Here's the key idea: PyTorch's `torch.compile` traces the Python source code and creates a graph based representation (called FXGraphs) in compile time. We simply take that graph and convert it into a Chakra graph. While the default behavior is to provide this graph to low-level compilers (such as Inductor or NvFuser), PyTorch also exposes an API through which developers can write custom compilers (i.e. custom modifications to this graph). This repository writes a 'custom compiler' that, instead of compiling and giving the compiled result to PyTorch, simply takes the FX Graph, converts it into a Chakra Graph, stores it locally, and gracefully exits.

For a detailed setup guide, please refer to [USER_GUIDE.md](USER_GUIDE.md)

Here is the directory structure:
```
profile_fxgraph.py #
|
--src
|
|
|
