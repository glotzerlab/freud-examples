# About

This example demonstrates the use of freud to analyze the outputs of a simulation run with GROMACS.
Specifically, we compute the RDF of a system of TIP4P water.
The system is set up using the GROMACS configuration files provided [here](https://www.svedruziclab.com/tutorials/gromacs/1-tip4pew-water/) under the [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
These files were created by Wes Barnett in the lab of Professor Zeljko M. Svedruzic at the University of Rijeka Department of BIomedical Tehcnology, Copyright 2015-2019 Zeljko Svedruzic.
The only modification is that the tau\_p parameter for the second equilibration run has been doubled because GROMACS now warns against possibly coordinated fluctuations between the two algorithms if the damping constants are identical.
These modified files are also made available under the CC BY-SA 4.0 License.
