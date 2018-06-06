Team;
Markus KOLLER - 1710299012 
Leonhardt SCHWARZ - 1710299019

Aufgaben;
Alle Aufgaben wurden mit CPP implementiert. Öffnen/Starten des Projektes in Visual Studio 2017: Datei/Öffnen/CMake... und CMakeList.txt auswählen
	-Exercise 1;
		Aufgabe wurde vollständig für RGB-Bilder gelöst, das Ergebnisbild befindet sich im CMake-directory (zB: C:\Users\MK\CMakeBuilds\50286fdc-45ca-eb34-81f8-d50e7c783897\build\x86-Debug\).
		Ergebnisbilder stimmen
		--Inputs;
			lenna.tga (225x225px)
			Inquisition.tga (2000x3100px) - nicht in Abgabe (23MB Filesize)
			lizard.tga (4444x3136px) - nicht in Abgabe (40MB Filesize)
			Theta: π/4
		--Performance;
			lenna.tga;
				GPU (Avg): 443ms
				CPU only (Avg): 134ms
			Inquisition.tga; 
				GPU (Avg): 13262ms
				CPU only (Avg): 13864ms
			lizard.tga;
				GPU (Avg): 31929ms
				CPU only (Avg): 32586ms
		--Aufgetretene Probleme;
			Herausfinden wie OpenCL überhaupt funktioniert
			Herausfinden wo die builderrors vom Kernel Compiler eingesehen werden können
			Testen mit großen Bildern sehr zeitaufwändig
	-Exercise 2;
		--Inputs;

		--Performance;

		--Aufgetretene Probleme;
	-Exercise 3;
		--Inputs;

		--Performance;

		--Aufgetretene Probleme;