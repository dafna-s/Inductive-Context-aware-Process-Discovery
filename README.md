# iductive-data-aware-process-discovery
This work combine control-flow and data perspectives under a single roof by extending inductive process discovery.

## Getting Started

### Prerequisites:
1. Anaconda 3.
2. INtelliJ https://www.jetbrains.com/idea/download/#section=windows.

### Installing the *EDU-ProM* library:

In order to use the inductive miner from our phyton project it is required to use the EDU-ProM library.
The usaged with the library is applyed by the 'MyScript.bat' script file that is been called from the
phyton project. In this script file we call the EDU-ProM java project.
It is required to update the 'MyScript.bat' file with the currect jar files path from your local machine.

1. Follow the instuctions in https://github.com/bpm-technion/EDU-ProM to install the EDU-ProM 
   (Use the 'iductive-data-aware-process-discovery' folder as installation directory).

2. Run the application.

3. From the 'Run' tab open the '.jar' file that is been created.
   For example - 
   "C:\Program Files\Java\jdk1.8.0_181\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.2.4\lib\idea_rt.jar=51558:C:\Program Files\JetBrains\IntelliJ IDEA 2018.2.4\bin" -Dfile.encoding=UTF-8 -classpath C:\Users\odeds\AppData\Local\Temp\classpath710228862.jar org.eduprom.Main
   open the file 'C:\Users\odeds\AppData\Local\Temp\classpath710228862.jar'.

4. Copy the contant of the file (exclude the 'Class-Path: ') and paste it to the MyScript.bat file in the required location.

5. Copy the execute command (exclude the -classpath parameter) and paste it to the MyScript.bat file in the required location.
   For example - 
   "C:\Program Files\Java\jdk1.8.0_181\bin\java.exe" "-javaagent:C:\Program Files\JetBrains\IntelliJ IDEA 2018.2.4\lib\idea_rt.jar=51558:C:\Program Files\JetBrains\IntelliJ IDEA 2018.2.4\bin" -Dfile.encoding=UTF-8 org.eduprom.Main

6. Run the 'MyScript.bat' from the command line in order to verify it is working. 

### Installing the *pm4py* library:

In order to use the inductive miner from our phyton project it is required to use the pm4py library.

1.Install the library from http://pm4py.org/ 
  (Use the 'iductive-data-aware-process-discovery' folder as installation directory).
  
### Installing the *iductive-data-aware-process-discovery* project:

1. Open the PyCharm.

2. Open the Project in the PyCharm.

3. Run the application.

4. TOTO::

## The Paper
The paper is been published at *Process Mining Conference 2019 (https://icpmconference.org/)*

## The Team
*iductive-data-aware-process-discovery* was developed by [Dafna Schumacher](dafna.s@campus.technion.ac.il).
