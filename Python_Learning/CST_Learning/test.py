"""
Project Name: CST Automation
File Name: test.py
Author: Liu Ming
Created: March 8th, 2025
Description:
This script automates the creation and simulation of a microwave project in CST Studio Suite. 
It sets up the design environment, creates a new project, configures the solver and units, 
adds a cylindrical structure, sets background and boundary conditions, runs the solver, 
and retrieves the mode frequencies from the results.
Preparation:
1. Ensure CST Studio Suite 2021 is installed on your system.
2. Update the `cst_lib_path` variable to point to the CST python libraries directory.
3. Ensure the output directory specified in the `prj.save` method exists.
Usage:
Run this script using a Python interpreter. Ensure that the CST Studio Suite application 
is accessible and the CST python libraries are correctly referenced.
Change Log:
- March 8th, 2025: Initial creation of the script by Liu Ming.
"""

import sys
cst_lib_path = r"C:\CST Studio Suite 2021\AMD64 python_cst_libraries"
sys.path.append(cst_lib_path)
import cst
import cst.interface
import cst.results
import time

# Create a new design environment
de = cst.interface.DesignEnvironment()

# Create a new Microwave Studio project
prj = de.new_mws()

# Save the project
prj.save(r"C:\Users\LiuMing\Downloads\cylinder_project.cst")

# Set the solver to Eigenmode
prj.modeler.add_to_history("Set Solver Type", "ChangeSolverType \"HF Eigenmode\"")

# Set Unit
GeometryUnit = "mm"
FrequencyUnit = "GHz"
TimeUnit = "ns"
TemperatureUnit = "Kelvin"
VoltageUnit = "V"
CurrentUnit = "A"
ResistanceUnit = "Ohm"
ConductanceUnit = "Siemens"
CapacitanceUnit = "PikoF"
InductanceUnit = "NanoH"

UnitVba = f"""With Units 
    .Geometry "{GeometryUnit}" 
    .Frequency "{FrequencyUnit}" 
    .Time "{TimeUnit}" 
    .TemperatureUnit "{TemperatureUnit}" 
    .Voltage "{VoltageUnit}" 
    .Current "{CurrentUnit}" 
    .Resistance "{ResistanceUnit}" 
    .Conductance "{ConductanceUnit}" 
    .Capacitance "{CapacitanceUnit}" 
    .Inductance "{InductanceUnit}" 
    .SetResultUnit "frequency", "frequency", "" 
End With"""
prj.modeler.add_to_history("Set Unit", UnitVba)

# Set frequency range
FrequencyRangeMin = 0
FrequencyRangeMax = 10

FrequencyRangeVba = f"""Solver.FrequencyRange "{FrequencyRangeMin}", "{FrequencyRangeMax}" """
prj.modeler.add_to_history("Set Frequency Range", FrequencyRangeVba)

# Add a cylinder
Cylinder_outer_radius = 100
Cylinder_inner_radius = 0
Cylinder_axis = "z"
Cylinder_zrange = [-50, 50]
Cylinder_xcenter = 0
Cylinder_ycenter = 0
Cylinder_segments = 0

CavityShapeVba = f"""With Cylinder 
    .Reset 
    .Name "Cavity" 
    .Component "component1" 
    .Material "Vacuum" 
    .OuterRadius "{str(Cylinder_outer_radius)}" 
    .InnerRadius "{str(Cylinder_inner_radius)}" 
    .Axis "{Cylinder_axis}" 
    .Zrange "{str(Cylinder_zrange[0])}", "{str(Cylinder_zrange[1])}" 
    .Xcenter "{str(Cylinder_xcenter)}" 
    .Ycenter "{str(Cylinder_ycenter)}" 
    .Segments "{str(Cylinder_segments)}" 
    .Create 
End With"""
prj.modeler.add_to_history("Add Cylinder", CavityShapeVba)

# Set background conditions
XminSpace = 0.0
XmaxSpace = 0.0
YminSpace = 0.0
YmaxSpace = 0.0
ZminSpace = 0.0
ZmaxSpace = 0.0
Rho = 0.0
ThermalType = "Normal"
ThermalConductivity = 0
SpecificHeat = 0
DynamicViscosity = 0
Emissivity = 0
MetabolicRate = 0.0
VoxelConvection = 0.0
BloodFlow = 0
MechanicsType = "Unused"
FrqType = "all"
MaterialType = "Pec"
MaterialUnitFrequency = "Hz"
MaterialUnitGeometry = "m"
MaterialUnitTime = "s"
MaterialUnitTemperature = "Kelvin"
Epsilon = 1.0
Mu = 1.0
ReferenceCoordSystem = "Global"
CoordSystemType = "Cartesian"
NLAnisotropy = "False"
NLAStackingFactor = 1
NLADirectionX = 1
NLADirectionY = 0
NLADirectionZ = 0
Colour = (0.6, 0.6, 0.6)
Wireframe = "False"
Reflection = "False"
Allowoutline = "True"
Transparentoutline = "False"
Transparency = 0

BackgroundVba = f"""With Background 
    .ResetBackground 
    .XminSpace "{XminSpace}" 
    .XmaxSpace "{XmaxSpace}" 
    .YminSpace "{YminSpace}" 
    .YmaxSpace "{YmaxSpace}" 
    .ZminSpace "{ZminSpace}" 
    .ZmaxSpace "{ZmaxSpace}" 
    .ApplyInAllDirections "False" 
End With 
With Material 
    .Reset 
    .Rho "{Rho}" 
    .ThermalType "{ThermalType}" 
    .ThermalConductivity "{ThermalConductivity}" 
    .SpecificHeat "{SpecificHeat}", "J/K/kg" 
    .DynamicViscosity "{DynamicViscosity}" 
    .Emissivity "{Emissivity}" 
    .MetabolicRate "{MetabolicRate}" 
    .VoxelConvection "{VoxelConvection}" 
    .BloodFlow "{BloodFlow}" 
    .MechanicsType "{MechanicsType}" 
    .FrqType "{FrqType}" 
    .Type "{MaterialType}" 
    .MaterialUnit "Frequency", "{MaterialUnitFrequency}" 
    .MaterialUnit "Geometry", "{MaterialUnitGeometry}" 
    .MaterialUnit "Time", "{MaterialUnitTime}" 
    .MaterialUnit "Temperature", "{MaterialUnitTemperature}" 
    .Epsilon "{Epsilon}" 
    .Mu "{Mu}" 
    .ReferenceCoordSystem "{ReferenceCoordSystem}" 
    .CoordSystemType "{CoordSystemType}" 
    .NLAnisotropy "{NLAnisotropy}" 
    .NLAStackingFactor "{NLAStackingFactor}" 
    .NLADirectionX "{NLADirectionX}" 
    .NLADirectionY "{NLADirectionY}" 
    .NLADirectionZ "{NLADirectionZ}" 
    .Colour "{Colour[0]}", "{Colour[1]}", "{Colour[2]}" 
    .Wireframe "{Wireframe}" 
    .Reflection "{Reflection}" 
    .Allowoutline "{Allowoutline}" 
    .Transparentoutline "{Transparentoutline}" 
    .Transparency "{Transparency}" 
    .ChangeBackgroundMaterial 
End With"""
prj.modeler.add_to_history("Set Background Conditions", BackgroundVba)

# Set boundary conditions
BoundaryVba = """With Boundary 
    .Xmin "electric" 
    .Xmax "electric" 
    .Ymin "electric" 
    .Ymax "electric" 
    .Zmin "electric" 
    .Zmax "electric" 
    .Xsymmetry "none" 
    .Ysymmetry "none" 
    .Zsymmetry "none" 
    .ApplyInAllDirections "True" 
End With"""
prj.modeler.add_to_history("Set Boundary Conditions", BoundaryVba)

# Run the solver
prj.modeler.run_solver()
while prj.modeler.is_solver_running():
    time.sleep(1)

# Save the project and include results
prj.save(r"C:\Users\LiuMing\Downloads\cylinder_project.cst", include_results=True)

# Close the project and design environment
prj.close()
de.close()

# Load the project file and get results
pf = cst.results.ProjectFile(r"C:\Users\LiuMing\Downloads\cylinder_project.cst")
results_module = pf.get_3d()

# Get the mode frequencies
mode_frequencies_result = results_module.get_result_item("1D Results\\Mode Frequencies\\Mode 1", run_id=0)
data = mode_frequencies_result.get_data()
for i in range(len(data)):
    print(data[i])