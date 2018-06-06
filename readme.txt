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
			Theta: π/4 (45 Grad)
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
		Aufgabe wurde vollständig gelöst, für jeden Algorithmus wurde ein eigener Kernel erstellt. Zur Kontrolle wurde ein sequentielle Algorithmus erstellt.
		--Inputs/Kernels;
			naive_parallel_prefixsum(Naive Parallel Scan)
			naive_parallel_prefixsum2(Double-buffered Naive Parallel Scan)
			blelloch_scan(Work-Efficient Parallel Scan after Blelloch)

			Vektorgrößen: 256 (work group size), 1024 (*4), 4096 (*16)
		--Performance;
			naive_parallel_prefixsum(Naive Parallel Scan - inclusive);
				256: correct (1ms)
				1024: INCORRECT (1ms) => An Workgroupgrenze wird letzter Wert nicht weitergegeben
				4096: INCORRECT (2ms) => An Workgroupgrenze wird letzter Wert nicht weitergegeben
				Ab einer gewissen Inputlänge werden die genutzten Buffer zu groß(cl_out_of_resources)			
			naive_parallel_prefixsum2(Naive Parallel Scan with double buffer - inclusive);
				256: Output ist eine vector mit Nullen?
				1024: INCORRECT (1ms) => An Workgroupgrenze wird letzter Wert nicht weitergegeben
				4096: INCORRECT (2ms) => An Workgroupgrenze wird letzter Wert nicht weitergegeben	
				Ab einer gewissen Inputlänge werden die genutzten Buffer zu groß(cl_out_of_resources)
			blelloch_scan(Work-Efficient Parallel Scan after Blelloch - exclusive);
				256: correct (1ms)
				1024: correct (2ms)
				4096: correct (2ms)
				256*1024: correct (9ms)
				blelloch ist etwas langsamer als der naive approach, liefert dafür richtige ergebnisse
		--Aufgetretene Probleme;
			Algorithmen verstehen (viel Zeit & Rumprobieren nötig)
			Einige IndexOutOfBounds-Fehler
			Tlw. Fehler die nur in NVIDIA auftraten
			Kurzzeitig wurde das erste Element des nächsten Blocks mit dem letzten Element des vorherigen Blocks gefüllt ("off-by-one error")
			naive_parallel_prefixsum2 hat bei einer inputlänge von 256 nur 0er zurückgeliefert
	-Exercise 3;
		--Inputs;

		--Performance;

		--Aufgetretene Probleme;