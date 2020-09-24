# Literature review

# Technologies

## Pytorch

Pytorch is a DL framework that was inspired by Torch which was written in Lua. The two key features of Pytorch are:

- Imperative programming
- Dynamic Computation Graphing

The contrast to imperative programming would be symbolic (Siraj, YT) programming (or functional programming). Tensorflow is functional in nature so it has a lot of lazy evaluation built in [citation required]. Symbolic programs are more efficient since it can reuse variables and memory (Siraj, YT).

Pytorch is define by run and that means that it is building the graph as the program is running, whereas Tensorflow builds the graph first. Tensorflow first assembles a graph and uses a session to execute ops in the graph. This can be useful if we want to distribute the model on something like Spark for distributed cross validation.

Static graphs work well for fixed-size networks like Feed Forward NN and CNNs.

Dynamic graphing is useful whenever the amount of work that needs to be done is variable. It works well for things like RNN. It is possible through Tensorflow since it has support for "primitives" but it is still static and this aspect feels more natural in Pytorch.

This also makes debugging really easy since it is a written line of code that fails as opposed to being burried somewhere in that static graph.

Tensorflow is better suited for beginners and production since it was built with that usecase in mind.
Pytorch may be better for research.

# References

- Siraj, YT [[source](https://www.youtube.com/watch?v=nbJ-2G2GXL0)]

# Methods

- [Jetson segnet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/segnet-console-2.md#segmenting-images-from-the-command-line)
