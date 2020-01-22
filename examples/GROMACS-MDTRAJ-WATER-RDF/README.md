# About

This example demonstrates the use of freud to analyze the outputs of a simulation run with GROMACS.
Specifically, we compute the RDF of a system of TIP4P water.
The system is set up using the GROMACS configuration files provided [here](https://www.svedruziclab.com/tutorials/gromacs/1-tip4pew-water/) under the CC BY-SA 4.0 License.
The only modification is that the tau\_p parameter for the second equilibration run has been doubled because GROMACS now warns against possibly coordinated fluctuations between the two algorithms if the damping constants are identical.

TODO: Properly attribute everything as required by the license.
